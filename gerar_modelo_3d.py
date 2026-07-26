"""
Pipeline consolidado -- Camada 2, parte 2: geometria 2D validada -> modelo 3D.

Recebe as saídas do extrair_geometria.py (paredes.json e portas.json) e
produz um modelo 3D real (.glb) com:
  1. Parede com espessura (não é mais linha, é volume)
  2. Extrusão em altura
  3. Vão de porta cortado no lugar certo
  4. Laje de piso simples, pra dar contexto visual e evitar peças "flutuando"

Uso:
    python gerar_modelo_3d.py paredes.json portas.json saida.glb [espessura_m] [altura_m]

Todas as coordenadas de entrada devem estar na MESMA unidade (normalmente já
convertidas pra metros pelo extrair_geometria.py -- confira o fator de escala
no relatório antes de usar).

Decisão de arquitetura importante (não mudar sem motivo forte):
  Cada segmento de parede vira uma CAIXA INDIVIDUAL (não um polígono único
  com furo). Isso evita um bug real de triangulação (earcut trava em
  polígono-com-furo complexo) que apareceu ao testar em arquivo real no
  dia 25/07. Custo: pequena sobra de volume nos cantos (paredes se
  sobrepõem um pouco). Benefício: robustez -- sempre gera um sólido válido.
"""

import sys
import json
import numpy as np
from shapely.geometry import LineString, Point
import trimesh


def _comprimento(l):
    x1, y1 = l["start"]
    x2, y2 = l["end"]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _dividir_segmento_com_porta(p1, p2, t_hinge, largura, comprimento):
    """Recorta o vão de porta de um segmento de parede, devolvendo 0, 1 ou 2
    sub-segmentos (0 se a porta ocupa a parede inteira)."""
    candidatos = []
    for a in (t_hinge, t_hinge - largura):
        b = a + largura
        if a >= -1e-6 and b <= comprimento + 1e-6:
            candidatos.append((max(a, 0), min(b, comprimento)))
    if not candidatos:
        a = max(0, min(t_hinge, comprimento - largura))
        candidatos = [(a, min(a + largura, comprimento))]
    a, b = candidatos[0]
    direcao = (np.array(p2) - np.array(p1)) / comprimento
    subsegs = []
    if a > 1e-6:
        subsegs.append((tuple(p1), tuple(np.array(p1) + direcao * a)))
    if b < comprimento - 1e-6:
        subsegs.append((tuple(np.array(p1) + direcao * b), tuple(p2)))
    return subsegs


def dividir_paredes_pelas_portas(paredes, portas, limiar_proximidade_m=0.15):
    """Associa cada porta à parede mais próxima e recorta o vão."""
    linhas_shapely = [LineString([l["start"], l["end"]]) for l in paredes]

    portas_por_segmento = {}
    for porta in portas:
        largura = porta.get("largura_estimada_m")
        if not largura:
            continue
        p = Point(porta["posicao"])
        idx = min(range(len(linhas_shapely)), key=lambda i: linhas_shapely[i].distance(p))
        if linhas_shapely[idx].distance(p) <= limiar_proximidade_m:
            portas_por_segmento.setdefault(idx, []).append(porta)

    segmentos_finais = []
    for i, l in enumerate(paredes):
        p1, p2 = l["start"], l["end"]
        comprimento = linhas_shapely[i].length
        if i in portas_por_segmento and comprimento > 0:
            porta = portas_por_segmento[i][0]  # só a primeira, se houver mais de uma no mesmo trecho
            t_hinge = linhas_shapely[i].project(Point(porta["posicao"]))
            subsegs = _dividir_segmento_com_porta(p1, p2, t_hinge, porta["largura_estimada_m"], comprimento)
            segmentos_finais.extend(subsegs)
        else:
            segmentos_finais.append((tuple(p1), tuple(p2)))

    return segmentos_finais, len(portas_por_segmento)


def gerar_porta_3d(porta, espessura_parede_m, altura_porta_m=2.1):
    """Cria um painel simples (caixa) representando a folha da porta, para
    que ela exista como geometria própria e possa virar um nó separado no
    .glb (necessário para seleção/coloração individual).

    LIMITAÇÃO CONHECIDA: sem rotação confiável para porta detectada por
    arco (só bloco tem rotação hoje), o painel é gerado sem rotação --
    pode não ficar alinhado com a parede em portas detectadas por arco.
    Suficiente como placeholder visual/nó endereçável, não como geometria
    arquitetonicamente precisa ainda.
    """
    largura = porta.get("largura_estimada_m") or 0.8
    x, y = porta["posicao"]
    painel = trimesh.creation.box(extents=[largura, espessura_parede_m * 0.6, altura_porta_m])
    painel.apply_translation([x, y, altura_porta_m / 2])
    return painel


def gerar_paredes_3d(segmentos, espessura_m, altura_m):
    """Extruda cada segmento como uma caixa individual (robusto a furo complexo)."""
    meshes = []
    for p1, p2 in segmentos:
        seg = LineString([p1, p2])
        if seg.length < 0.01:
            continue
        footprint = seg.buffer(espessura_m / 2, cap_style="square", join_style="mitre")
        m = trimesh.creation.extrude_polygon(footprint, height=altura_m)
        meshes.append(m)
    return trimesh.util.concatenate(meshes)


def gerar_piso(paredes, espessura_piso_m=0.1):
    """Laje simples cobrindo o envelope da planta -- dá contexto visual."""
    xs = [p for l in paredes for p in (l["start"][0], l["end"][0])]
    ys = [p for l in paredes for p in (l["start"][1], l["end"][1])]
    piso = trimesh.creation.box(extents=[max(xs) - min(xs), max(ys) - min(ys), espessura_piso_m])
    piso.apply_translation([(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, -espessura_piso_m / 2])
    return piso


def main():
    if len(sys.argv) < 5:
        print("Uso: python gerar_modelo_3d.py paredes.json portas.json saida.glb FATOR_PARA_METROS [espessura_m=0.15] [altura_m=2.7]")
        print("\nFATOR_PARA_METROS é OBRIGATÓRIO -- é o mesmo valor que o extrair_geometria.py")
        print("reportou em 'escala_fator_para_metros'. NÃO existe valor padrão de propósito:")
        print("cada arquivo DWG pode estar numa unidade diferente (mm, m, polegada...), e assumir")
        print("metros silenciosamente foi exatamente o bug que apareceu hoje testando no banheiro.")
        sys.exit(1)

    caminho_paredes, caminho_portas, caminho_saida = sys.argv[1:4]
    fator_para_metros = float(sys.argv[4])
    espessura_m = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
    altura_m = float(sys.argv[6]) if len(sys.argv) > 6 else 2.7

    with open(caminho_paredes) as f:
        paredes_bruto = json.load(f)
    with open(caminho_portas) as f:
        portas_bruto = json.load(f)

    # converte tudo pra metros ANTES de qualquer cálculo geométrico
    paredes = [
        {"start": [p * fator_para_metros for p in l["start"]],
         "end": [p * fator_para_metros for p in l["end"]]}
        for l in paredes_bruto
    ]
    portas = []
    for p in portas_bruto:
        p2 = dict(p)
        p2["posicao"] = [c * fator_para_metros for c in p["posicao"]]
        portas.append(p2)

    print(f"Fator aplicado: 1 unidade do arquivo = {fator_para_metros} metros")

    segmentos, qtd_portas_cortadas = dividir_paredes_pelas_portas(paredes, portas)
    print(f"Paredes: {len(paredes)} segmentos originais -> {len(segmentos)} após recorte de porta")
    print(f"Portas efetivamente cortadas: {qtd_portas_cortadas} de {len(portas)} detectadas")

    mesh_paredes = gerar_paredes_3d(segmentos, espessura_m, altura_m)
    piso = gerar_piso(paredes)
    modelo_completo = trimesh.util.concatenate([mesh_paredes, piso])

    print(f"\nMesh final: {len(modelo_completo.vertices)} vértices")
    print(f"Watertight (parede isolada, antes de exportar): {mesh_paredes.is_watertight}")

    modelo_completo.export(caminho_saida)
    print(f"\nExportado: {caminho_saida}")


if __name__ == "__main__":
    main()
