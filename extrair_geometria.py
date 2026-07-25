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


def detectar_portas(doc, arcos, paredes, fator_para_metros):
    msp = doc.modelspace()
    portas = []

    for e in msp:
        if e.dxftype() != "INSERT":
            continue
        nome = e.dxf.name.upper()
        if any(p in nome for p in NOME_BLOCO_PORTA_PADROES):
            x, y, _ = e.dxf.insert
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
            portas.append({
                "metodo": "bloco",
                "nome_bloco": e.dxf.name,
                "posicao": [x, y],
                "largura_estimada_m": largura_m,
            })

    if paredes:
        for a in arcos:
            if not (70 <= _angulo_total(a) <= 100):
                continue
            largura_m = a["radius"] * fator_para_metros
            if not (0.3 <= largura_m <= 1.5):
                continue  # não é plausível como largura de porta real
            cx, cy = a["center"]
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
# 5. Visualização e main
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

    portas = detectar_portas(doc, arcos, paredes_amplas, fator)
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
