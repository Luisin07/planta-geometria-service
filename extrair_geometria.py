"""
Pipeline consolidado — extração de geometria estrutural a partir de um DXF real.

Reúne tudo que foi validado manualmente hoje em um único script:
  1. Leitura de coordenadas reais (linhas, arcos, polylines, blocos)
  2. Detecção automática de escala/unidade (nunca confia cegamente no $INSUNITS)
  3. Extração de paredes (tenta layer nomeado primeiro; se não achar, usa
     grafo de conectividade + poda de galho solto)
  4. Detecção de porta (via bloco tipo "PUERTA"/"DOOR"/"PORTA", e via arco de
     giro solto no modelspace)

Uso:
    python extrair_geometria.py caminho/para/arquivo.dxf

Saída (na mesma pasta do script):
    <nome>_relatorio.json   -> escala detectada, confiança, métodos usados
    <nome>_paredes.json     -> segmentos de parede (coordenadas reais)
    <nome>_portas.json      -> portas detectadas (posição, método, largura estimada)
    <nome>_resultado.png    -> desenho para conferência visual humana

IMPORTANTE: a escala é sempre uma estimativa. Confira o campo "confianca" no
relatório. Se vier "baixa", confirme manualmente antes de confiar no resultado.
"""

import sys
import os
import json
import math
from collections import defaultdict, Counter

import ezdxf
import matplotlib.pyplot as plt

LARGURA_PORTA_REFERENCIA_M = 0.8  # premissa: porta real fica perto de 0.8m de largura
LAYER_PAREDE_PADROES = ["A-WALL", "WALL", "PAREDE", "PARED", "MUR", "MURO", "MAUER"]
NOME_BLOCO_PORTA_PADROES = ["PUERTA", "DOOR", "PORTA", "TUR"]


# ---------------------------------------------------------------------------
# 1. Leitura de entidades
# ---------------------------------------------------------------------------

def carregar_entidades(doc):
    msp = doc.modelspace()
    linhas, arcos, polylines = [], [], []

    for e in msp:
        if e.dxftype() == "LINE":
            linhas.append({
                "layer": e.dxf.layer,
                "start": (e.dxf.start.x, e.dxf.start.y),
                "end": (e.dxf.end.x, e.dxf.end.y),
            })
        elif e.dxftype() == "ARC":
            arcos.append({
                "layer": e.dxf.layer,
                "center": (e.dxf.center.x, e.dxf.center.y),
                "radius": e.dxf.radius,
                "start_angle": e.dxf.start_angle,
                "end_angle": e.dxf.end_angle,
            })
        elif e.dxftype() in ("LWPOLYLINE", "POLYLINE"):
            try:
                pontos = [(p[0], p[1]) for p in e.get_points()]
            except AttributeError:
                pontos = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            polylines.append({
                "layer": e.dxf.layer,
                "pontos": pontos,
                "fechada": e.closed if hasattr(e, "closed") else False,
            })

    return linhas, arcos, polylines


# ---------------------------------------------------------------------------
# 2. Detecção de escala (nunca confia só no cabeçalho do arquivo)
# ---------------------------------------------------------------------------

def _angulo_total(a):
    total = (a["end_angle"] - a["start_angle"]) % 360
    return total


def _selecionar_raio_representativo(raios):
    """Entre vários raios candidatos (que podem incluir muito ruído de
    hachura/parafuso/detalhe pequeno), acha o maior salto proporcional na
    distribuição ordenada e retorna a média do grupo acima desse salto.
    Isso é mais robusto que mediana simples quando o ruído pequeno domina
    em quantidade sobre o valor real (a porta)."""
    raios = sorted(r for r in raios if r > 0)
    if not raios:
        return None
    if len(raios) == 1:
        return raios[0]
    maior_salto = 0
    idx_corte = len(raios) - 1
    for i in range(len(raios) - 1):
        razao = raios[i + 1] / raios[i]
        if razao > maior_salto:
            maior_salto = razao
            idx_corte = i + 1
    cluster_superior = raios[idx_corte:]
    return sum(cluster_superior) / len(cluster_superior)



def detectar_escala(doc, arcos_modelspace):
    """Retorna (fator_para_metros, confianca, explicacao).
    fator_para_metros: multiplique qualquer coordenada bruta por esse valor
    para obter metros reais.
    """
    msp = doc.modelspace()

    # Método 1 (confiança alta): bloco de porta contendo um arco de giro
    raios_reais = []
    for e in msp:
        if e.dxftype() != "INSERT":
            continue
        nome = e.dxf.name.upper()
        if not any(p in nome for p in NOME_BLOCO_PORTA_PADROES):
            continue
        try:
            bloco = doc.blocks.get(e.dxf.name)
        except Exception:
            continue
        for be in bloco:
            if be.dxftype() == "ARC" and 70 <= _angulo_total({
                "start_angle": be.dxf.start_angle, "end_angle": be.dxf.end_angle
            }) <= 100:
                escala_media = (abs(e.dxf.xscale) + abs(e.dxf.yscale)) / 2
                raios_reais.append(be.dxf.radius * escala_media)

    if raios_reais:
        raio_repr = _selecionar_raio_representativo(raios_reais)
        fator = LARGURA_PORTA_REFERENCIA_M / raio_repr
        return fator, "alta", (
            f"bloco de porta ({len(raios_reais)} instâncias encontradas, "
            f"raio representativo {raio_repr:.4f} unidades do arquivo)"
        )

    # Método 2 (confiança média): arco solto no modelspace com ângulo de porta
    candidatos = [a["radius"] for a in arcos_modelspace if 70 <= _angulo_total(a) <= 100 and a["radius"] > 0]
    if candidatos:
        raio_repr = _selecionar_raio_representativo(candidatos)
        fator = LARGURA_PORTA_REFERENCIA_M / raio_repr
        return fator, "média", (
            f"arco solto no modelspace ({len(candidatos)} candidatos, "
            f"raio representativo {raio_repr:.4f} unidades do arquivo, "
            f"filtrado por maior salto na distribuição)"
        )

    # Método 3 (confiança baixa): cabeçalho do arquivo -- historicamente não confiável
    mapa_para_metros = {0: 1.0, 1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 9: 0.1}
    codigo = doc.header.get('$INSUNITS', 0)
    fator = mapa_para_metros.get(codigo, 1.0)
    return fator, "baixa", (
        f"nenhum bloco/arco de porta encontrado -- usando \\$INSUNITS={codigo} do "
        f"cabeçalho do arquivo. ESSE CAMPO JÁ SE PROVOU ERRADO ANTES. Confirme manualmente."
    )


# ---------------------------------------------------------------------------
# 3. Extração de parede
# ---------------------------------------------------------------------------

def achar_layer_parede(linhas):
    contagem = Counter(l["layer"] for l in linhas)
    for layer, qtd in contagem.most_common():
        layer_upper = layer.upper()
        if qtd >= 10 and any(p in layer_upper for p in LAYER_PAREDE_PADROES):
            return layer, qtd
    return None, 0


def _comprimento(l):
    x1, y1 = l["start"]
    x2, y2 = l["end"]
    return math.hypot(x2 - x1, y2 - y1)


def extrair_paredes_por_grafo(linhas, fator_para_metros, corte_metros=0.08, tolerancia_metros=0.002):
    """Retorna (paredes_finais, paredes_amplas).
    paredes_finais: depois da poda de galho solto -- bom para o envelope geral.
    paredes_amplas: antes da poda -- preserva jogs pequenos de parede (como
    recortes de porta), útil para checar proximidade de porta."""
    corte_bruto = corte_metros / fator_para_metros
    tolerancia_bruta = tolerancia_metros / fator_para_metros

    filtradas = [l for l in linhas if _comprimento(l) >= corte_bruto]

    def chave(ponto):
        return (round(ponto[0] / tolerancia_bruta), round(ponto[1] / tolerancia_bruta))

    pai = {}

    def find(x):
        while pai.setdefault(x, x) != x:
            x = pai[x]
        return x

    def une(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            pai[ra] = rb

    for l in filtradas:
        une(chave(l["start"]), chave(l["end"]))

    grupos = defaultdict(list)
    for l in filtradas:
        grupos[find(chave(l["start"]))].append(l)

    if not grupos:
        return []

    def bbox_area(g):
        xs = [p for l in g for p in (l["start"][0], l["end"][0])]
        ys = [p for l in g for p in (l["start"][1], l["end"][1])]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    candidato = max(grupos.values(), key=bbox_area)
    paredes_amplas = candidato

    edges = {}
    eid = 0
    for l in candidato:
        a, b = chave(l["start"]), chave(l["end"])
        if a == b:
            continue
        edges[eid] = (a, b, l)
        eid += 1

    vivas = set(edges.keys())
    mudou = True
    while mudou:
        mudou = False
        grau = defaultdict(int)
        for i in vivas:
            a, b, _ = edges[i]
            grau[a] += 1
            grau[b] += 1
        remover = [i for i in vivas if grau[edges[i][0]] == 1 or grau[edges[i][1]] == 1]
        if remover:
            vivas -= set(remover)
            mudou = True

    paredes_finais = [edges[i][2] for i in vivas]
    return paredes_finais, paredes_amplas


def extrair_paredes(linhas, fator_para_metros):
    layer, qtd = achar_layer_parede(linhas)
    if layer:
        paredes = [l for l in linhas if l["layer"] == layer]
        return paredes, paredes, f"layer nomeado '{layer}' ({qtd} linhas)"
    paredes_finais, paredes_amplas = extrair_paredes_por_grafo(linhas, fator_para_metros)
    return paredes_finais, paredes_amplas, "grafo de conectividade + poda de galho solto (nenhum layer de parede claro foi encontrado)"


# ---------------------------------------------------------------------------
# 3b. Detecção de estruturas desconectadas (candidatas a multiunidade/multi-vista)
# ---------------------------------------------------------------------------

TOLERANCIA_CONECTIVIDADE_M = 0.1  # calibrado com dado real (10/08): distância entre
    # parede realmente conectada é ~0 (ruído de ponto flutuante); a faixa 0.05-0.3m
    # não muda a contagem de grupos, indicando que é uma zona segura -- só acima de
    # ~1m começa a fundir estruturas que provavelmente são mesmo separadas.
MIN_SEGMENTOS_ESTRUTURA_SIGNIFICATIVA = 5  # abaixo disso, no dado real testado, os
    # grupos eram claramente ruído (traço solto, elemento isolado) -- salto visível
    # na distribuição de tamanho entre grupos de 4 e de 1 segmento.
GAP_METROS_SEPARACAO_FORTE = 5.0  # separação >= isso entre duas estruturas
    # significativas é sinal forte de unidade/prédio distinto (dado de referência:
    # 19,2m confirmado em arquivo real com duas unidades lado a lado). Gap menor
    # que isso (dado real testado: 0,9-2m) é mais provável ser cômodo/ala dentro
    # da MESMA estrutura, não prova suficiente sozinha.


def detectar_estruturas_desconectadas(paredes, fator_para_metros,
                                       tolerancia_m=TOLERANCIA_CONECTIVIDADE_M,
                                       min_segmentos=MIN_SEGMENTOS_ESTRUTURA_SIGNIFICATIVA):
    """
    Agrupa segmentos de parede em componentes fisicamente conectados (uma
    parede "toca" outra se a distância real entre elas, em metros, é menor
    que a tolerância -- cobre tanto ponta-com-ponta quanto parede interna
    encostando no meio de uma parede externa).

    Isso NÃO decide sozinho "é multiunidade" -- devolve os dados (quantos
    grupos significativos existem, envelope de cada um, menor distância
    entre os maiores) para quem chamar decidir o nível de alerta. Dois
    sinais são necessários juntos pra sinal forte, um só não basta:
    (1) mais de um grupo com segmentos suficientes pra ser estrutura de
    verdade, não ruído -- E (2) esses grupos ficarem fisicamente longe uns
    dos outros (metros, não centímetros) -- um desenho multi-vista normal
    (vários cômodos, cortes, plantas de andar diferente na mesma prancha)
    já fragmenta em dezenas de grupos com gaps pequenos (~1-2m no dado
    real testado), sem ser prova de unidade duplicada.
    """
    from shapely.geometry import LineString, MultiLineString

    linhas_m = [
        LineString([
            (p["start"][0] * fator_para_metros, p["start"][1] * fator_para_metros),
            (p["end"][0] * fator_para_metros, p["end"][1] * fator_para_metros),
        ])
        for p in paredes
    ]
    n = len(linhas_m)
    if n == 0:
        return {"grupos_significativos": [], "alerta": "nenhum"}

    pai = list(range(n))

    def find(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def unir(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            pai[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if linhas_m[i].distance(linhas_m[j]) <= tolerancia_m:
                unir(i, j)

    grupos_idx = {}
    for i in range(n):
        r = find(i)
        grupos_idx.setdefault(r, []).append(i)

    grupos_significativos = []
    for indices in grupos_idx.values():
        if len(indices) < min_segmentos:
            continue
        xs = [c for i in indices for c in (linhas_m[i].coords[0][0], linhas_m[i].coords[1][0])]
        ys = [c for i in indices for c in (linhas_m[i].coords[0][1], linhas_m[i].coords[1][1])]
        grupos_significativos.append({
            "segmentos_qtd": len(indices),
            "indices_paredes": indices,
            "envelope_m": {
                "min_x": round(min(xs), 2), "max_x": round(max(xs), 2),
                "min_y": round(min(ys), 2), "max_y": round(max(ys), 2),
            },
            "_uniao": MultiLineString([linhas_m[i] for i in indices]),
        })

    grupos_significativos.sort(key=lambda g: g["segmentos_qtd"], reverse=True)

    maior_gap_m = None
    if len(grupos_significativos) >= 2:
        menor_gap_entre_par = min(
            grupos_significativos[i]["_uniao"].distance(grupos_significativos[j]["_uniao"])
            for i in range(len(grupos_significativos))
            for j in range(i + 1, len(grupos_significativos))
        )
        maior_gap_m = round(menor_gap_entre_par, 2)

    for g in grupos_significativos:
        del g["_uniao"]  # objeto shapely não serializa em JSON, só serviu pro cálculo de distância

    if len(grupos_significativos) <= 1:
        alerta = "nenhum"
    elif maior_gap_m is not None and maior_gap_m >= GAP_METROS_SEPARACAO_FORTE:
        alerta = "forte"  # gap grande confirmado -- muito provável unidade/prédio distinto
    elif len(grupos_significativos) >= 4:
        alerta = "moderado"  # muitos grupos desconectados, mesmo sem um gap enorme -- vale checar
    else:
        alerta = "baixo"  # poucos grupos, gap pequeno -- provavelmente só cômodos/ala da mesma estrutura

    return {
        "grupos_significativos": grupos_significativos,
        "grupos_qtd": len(grupos_significativos),
        "menor_gap_entre_maiores_m": maior_gap_m,
        "alerta": alerta,
    }


# ---------------------------------------------------------------------------
# 4. Detecção de porta
# ---------------------------------------------------------------------------

def _dist_ponto_segmento(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _centro_vao_a_partir_da_insercao(x, y, rotacao_graus, largura_m, fator_para_metros):
    """O ponto de inserção do bloco de porta é o ponto de referência do bloco
    (tipicamente o dobradiça/canto), NÃO o centro do vão. Desloca esse ponto
    pela metade da largura, ao longo do eixo local X do bloco (convenção mais
    comum em blocos PUERTA/DOOR: X local = direção da largura do vão).

    Isso é extração determinística, não heurística: reaproveita dado que já
    vem do próprio INSERT (rotação) e da própria detecção de largura (arco de
    giro), sem inventar nada sobre a geometria.

    ATENÇÃO: se depois de testar em dado real a porta aparecer deslocada
    PERPENDICULAR ao vão em vez de ao longo dele, o bloco usado nesse arquivo
    segue a convenção contrária (Y local = largura) -- nesse caso trocar
    cos/sin por -sin/cos abaixo.
    """
    if largura_m is None:
        return [x, y]
    deslocamento_bruto = (largura_m / 2) / fator_para_metros
    rad = math.radians(rotacao_graus)
    dx = math.cos(rad) * deslocamento_bruto
    dy = math.sin(rad) * deslocamento_bruto
    return [x + dx, y + dy]


def detectar_portas(doc, arcos, paredes, fator_para_metros, margem_envelope_m=0.15, paredes_envelope=None):
    """paredes: lista usada para checar proximidade (normalmente paredes_amplas,
    que preserva jogs/recortes de porta).
    paredes_envelope: lista usada só para definir os limites reais da planta
    (normalmente a lista final, já podada). Se não for passada, usa `paredes`
    (comportamento antigo, sem a proteção extra)."""
    msp = doc.modelspace()
    portas = []

    if paredes_envelope is None:
        paredes_envelope = paredes

    for e in msp:
        if e.dxftype() != "INSERT":
            continue
        nome = e.dxf.name.upper()
        if any(p in nome for p in NOME_BLOCO_PORTA_PADROES):
            x, y, _ = e.dxf.insert
            rotacao_graus = getattr(e.dxf, "rotation", 0.0)
            largura_m = None
            try:
                bloco = doc.blocks.get(e.dxf.name)
                escala_media = (abs(e.dxf.xscale) + abs(e.dxf.yscale)) / 2
                raios = [
                    be.dxf.radius * escala_media
                    for be in bloco
                    if be.dxftype() == "ARC" and 70 <= _angulo_total({
                        "start_angle": be.dxf.start_angle, "end_angle": be.dxf.end_angle
                    }) <= 100
                ]
                if raios:
                    largura_m = max(raios) * fator_para_metros
            except Exception:
                pass

            posicao_centro = _centro_vao_a_partir_da_insercao(
                x, y, rotacao_graus, largura_m, fator_para_metros
            )

            portas.append({
                "metodo": "bloco",
                "nome_bloco": e.dxf.name,
                "posicao": posicao_centro,
                "posicao_insercao_bruta": [x, y],  # mantido para depuração/comparação visual
                "rotacao_graus": rotacao_graus,
                "largura_estimada_m": largura_m,
            })

    if paredes and paredes_envelope:
        # Envelope da planta REAL (lista final/podada), com margem pequena --
        # evita que um arco detectado "case" por coincidência com anotação/
        # legenda distante que sobrou na lista ampla de conectividade (bug
        # real: legenda ficou "perto" do arco em distância mas fora da
        # estrutura de verdade). Margem pequena (15cm) porque a planta pode
        # ser pequena (poucos metros) -- margem grande demais deixa de filtrar.
        margem_bruta = margem_envelope_m / fator_para_metros
        xs_env = [p for l in paredes_envelope for p in (l["start"][0], l["end"][0])]
        ys_env = [p for l in paredes_envelope for p in (l["start"][1], l["end"][1])]
        min_x, max_x = min(xs_env) - margem_bruta, max(xs_env) + margem_bruta
        min_y, max_y = min(ys_env) - margem_bruta, max(ys_env) + margem_bruta

        for a in arcos:
            if not (70 <= _angulo_total(a) <= 100):
                continue
            largura_m = a["radius"] * fator_para_metros
            if not (0.3 <= largura_m <= 1.5):
                continue  # não é plausível como largura de porta real
            cx, cy = a["center"]
            if not (min_x <= cx <= max_x and min_y <= cy <= max_y):
                continue  # fora do envelope real da planta -- provável falso positivo
            dist_min = min(_dist_ponto_segmento((cx, cy), l["start"], l["end"]) for l in paredes)
            if dist_min * fator_para_metros < 0.15:
                portas.append({
                    "metodo": "arco",
                    "posicao": [cx, cy],
                    "largura_estimada_m": round(largura_m, 3),
                    "distancia_parede_m": round(dist_min * fator_para_metros, 4),
                })

    return portas


# ---------------------------------------------------------------------------
# 5. Detecção de objeto/móvel via texto (MTEXT/TEXT)
# ---------------------------------------------------------------------------

def _texto_entidade(e):
    """TEXT guarda o conteúdo em e.dxf.text; MTEXT precisa de plain_text()
    pra remover códigos de formatação (\\P, fontes, etc)."""
    if e.dxftype() == "TEXT":
        return e.dxf.text
    if e.dxftype() == "MTEXT":
        try:
            return e.plain_text()
        except AttributeError:
            return e.text
    return ""


def _posicao_entidade_texto(e):
    if e.dxftype() == "TEXT":
        return (e.dxf.insert.x, e.dxf.insert.y)
    if e.dxftype() == "MTEXT":
        return (e.dxf.insert.x, e.dxf.insert.y)
    return None


def _altura_texto(e):
    """Altura da fonte, em unidades brutas do arquivo -- sinal de verdade
    do desenho técnico (nome de cômodo costuma ser desenhado maior que
    anotação de detalhe/instrução), em vez de adivinhar pelo CONTEÚDO do
    texto (ex: 'mais longo'), que se provou não confiável -- um texto de
    instrução comprido pode ser mais longo que o nome do cômodo em si."""
    if e.dxftype() == "TEXT":
        return e.dxf.height
    if e.dxftype() == "MTEXT":
        return e.dxf.char_height
    return 0


def detectar_objetos_por_texto(doc, envelope, margem_m, fator_para_metros):
    """Varre TEXT/MTEXT do modelspace e trata cada rótulo como um objeto/móvel
    endereçável (ex: TOILET, MIRROR, GRAB RAIL), desde que caia dentro do
    envelope da planta + uma margem -- isso descarta rótulo de anotação/cota
    que fica longe, na margem da prancha (duplicata de legenda, carimbo etc.),
    sem precisar de lista fixa de nomes válidos.

    Retorna lista de {"nome": str, "posicao": [x, y]} em coordenadas BRUTAS
    do arquivo (o chamador multiplica pelo fator de escala, igual já faz com
    parede e porta).
    """
    min_x, max_x, min_y, max_y = envelope
    margem_bruta = margem_m / fator_para_metros

    min_x -= margem_bruta
    max_x += margem_bruta
    min_y -= margem_bruta
    max_y += margem_bruta

    msp = doc.modelspace()
    objetos = []

    for e in msp:
        if e.dxftype() not in ("TEXT", "MTEXT"):
            continue

        texto = (_texto_entidade(e) or "").strip()
        if not texto:
            continue

        pos = _posicao_entidade_texto(e)
        if pos is None:
            continue
        x, y = pos

        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            continue  # fora do envelope + margem -- provável anotação/rótulo de prancha

        try:
            altura = _altura_texto(e)
        except Exception:
            altura = 0  # fallback seguro: se o DXF não tiver o atributo, não quebra o pipeline

        objetos.append({
            "nome": texto.upper(),
            "posicao": [x, y],
            "altura_texto": altura,
        })

    return objetos


# ---------------------------------------------------------------------------
# 6. Visualização e main
# ---------------------------------------------------------------------------

def desenhar(paredes, portas, arcos, saida_png):
    fig, ax = plt.subplots(figsize=(12, 12))

    for l in paredes:
        x = [l["start"][0], l["end"][0]]
        y = [l["start"][1], l["end"][1]]
        ax.plot(x, y, color="red", linewidth=2)

    for p in portas:
        x, y = p["posicao"]
        ax.plot(x, y, "go", markersize=10)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{len(paredes)} segmentos de parede, {len(portas)} portas detectadas")
    plt.savefig(saida_png, dpi=200, bbox_inches="tight")


def desenhar_planta_tecnica(segmentos_m, objetos_fisicos, nome_comodo, envelope_m, buffer,
                             espessura_m=0.15):
    """
    Desenho técnico da planta baixa em metros reais, pensado para ser
    entregue ao cliente -- diferente de `desenhar()` acima, que é debug
    interno (linha fina + bolinha, título de contagem de segmentos).

    Diferenças propositais:
    - Parede vira uma FAIXA preenchida com espessura real (não uma linha
      fina), calculada como um retângulo perpendicular à direção do
      segmento -- mesma noção de espessura usada no modelo 3D
      (ESPESSURA_PAREDE_M), só que desenhada em 2D.
    - `segmentos_m` já deve vir CORTADO nas portas (mesma função
      `dividir_paredes_pelas_portas` usada para gerar o modelo 3D) -- o
      vão da porta aparece sozinho, como um espaço vazio entre dois
      segmentos, sem precisar de lógica própria pra "desenhar porta".
    - Objeto/móvel vira um retângulo pequeno rotulado com o nome.
    - Nome do cômodo aparece como texto acima do desenho, se detectado.
    - Barra de escala de 1 metro no canto, pra dar noção de proporção
      sem depender de o observador saber ler coordenada.
    - Sem eixo, sem título de debug -- pensado para tela/PDF de cliente.

    `buffer` é um objeto tipo arquivo (ex: io.BytesIO) -- essa função não
    grava em disco, quem chama decide o destino (arquivo local, resposta
    HTTP, etc).
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    meia_esp = espessura_m / 2

    for seg in segmentos_m:
        (x1, y1), (x2, y2) = seg
        dx, dy = x2 - x1, y2 - y1
        comprimento = math.hypot(dx, dy)
        if comprimento == 0:
            continue
        nx, ny = -dy / comprimento, dx / comprimento  # normal unitária ao segmento
        cantos_x = [x1 + nx * meia_esp, x2 + nx * meia_esp, x2 - nx * meia_esp, x1 - nx * meia_esp]
        cantos_y = [y1 + ny * meia_esp, y2 + ny * meia_esp, y2 - ny * meia_esp, y1 - ny * meia_esp]
        ax.fill(cantos_x, cantos_y, facecolor="#4a4a4a", edgecolor="#2a2a2a", linewidth=0.6, zorder=2)

    for o in objetos_fisicos:
        x, y, nome = o["x"], o["y"], o.get("nome", "")
        ax.add_patch(plt.Rectangle((x - 0.15, y - 0.15), 0.3, 0.3,
                                    facecolor="#c9a876", edgecolor="#7a6142", linewidth=0.6, zorder=3))
        if nome:
            ax.text(x, y - 0.28, nome, fontsize=6, ha="center", va="top", color="#333333", zorder=4)

    # Nome de cômodo automático REMOVIDO do desenho por decisão consciente
    # (04/08): duas heurísticas diferentes (string mais longa, depois
    # altura de fonte) falharam por motivos reais e não óbvios -- sinal de
    # que é problema mais caro do que vale agora, e não era requisito
    # explícito do cliente (ele pediu parede/porta/proporção, não título
    # automático). O campo nome_comodo continua disponível no JSON de
    # /processar-planta pra quem quiser usar manualmente.

    # Barra de escala (1 metro) -- posicionada RELATIVA à posição real das
    # paredes, não em coordenada absoluta fixa. Bug anterior: barra fixa em
    # x=0 assumia que toda planta começa perto da origem, o que quebrou em
    # arquivo com coordenada absoluta longe de (0,0) -- o enquadramento
    # tinha que abranger a barra (em x=0) E a planta (em x=380, nesse
    # caso), espremendo a planta inteira a ponto de ficar invisível.
    xs_paredes = [pt[0] for seg in segmentos_m for pt in seg]
    ys_paredes = [pt[1] for seg in segmentos_m for pt in seg]
    if xs_paredes:
        escala_x0 = min(xs_paredes)
        escala_y = min(ys_paredes) - 0.6
    else:
        escala_x0, escala_y = 0, -0.45
    ax.plot([escala_x0, escala_x0 + 1], [escala_y, escala_y], color="black", linewidth=2, solid_capstyle="butt")
    ax.text(escala_x0 + 0.5, escala_y - 0.18, "1 m", fontsize=8, ha="center", va="top")

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Uso: python extrair_geometria.py caminho/para/arquivo.dxf")
        sys.exit(1)

    caminho = sys.argv[1]
    base = os.path.splitext(os.path.basename(caminho))[0]

    doc = ezdxf.readfile(caminho)
    linhas, arcos, polylines = carregar_entidades(doc)

    fator, confianca, explicacao = detectar_escala(doc, arcos)
    print(f"Escala detectada: 1 unidade do arquivo = {fator:.6f} metros")
    print(f"Confiança: {confianca}")
    print(f"Como: {explicacao}")

    paredes, paredes_amplas, metodo_parede = extrair_paredes(linhas, fator)
    print(f"\nParedes extraídas: {len(paredes)} segmentos (versão limpa, para o envelope geral)")
    print(f"Método: {metodo_parede}")

    portas = detectar_portas(doc, arcos, paredes_amplas, fator, paredes_envelope=paredes)
    print(f"\nPortas detectadas: {len(portas)}")
    for p in portas:
        print(f"  {p}")

    xs = [pt for l in paredes for pt in (l["start"][0], l["end"][0])]
    ys = [pt for l in paredes for pt in (l["start"][1], l["end"][1])]
    largura_m = (max(xs) - min(xs)) * fator if xs else 0
    altura_m = (max(ys) - min(ys)) * fator if ys else 0
    print(f"\nEnvelope estimado: {largura_m:.2f}m x {altura_m:.2f}m")

    relatorio = {
        "arquivo": caminho,
        "escala_fator_para_metros": fator,
        "escala_confianca": confianca,
        "escala_explicacao": explicacao,
        "paredes_metodo": metodo_parede,
        "paredes_qtd": len(paredes),
        "portas_qtd": len(portas),
        "envelope_largura_m": round(largura_m, 2),
        "envelope_altura_m": round(altura_m, 2),
    }

    with open(f"{base}_relatorio.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    with open(f"{base}_paredes.json", "w", encoding="utf-8") as f:
        json.dump(paredes, f, indent=2)
    with open(f"{base}_portas.json", "w", encoding="utf-8") as f:
        json.dump(portas, f, indent=2)

    desenhar(paredes, portas, arcos, f"{base}_resultado.png")
    print(f"\nArquivos gerados: {base}_relatorio.json, {base}_paredes.json, {base}_portas.json, {base}_resultado.png")


if __name__ == "__main__":
    main()