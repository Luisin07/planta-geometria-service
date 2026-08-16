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


# ------------------------------------------------------------------
# MATERIAIS_BASE (16/08) -- Etapa B do fix de "modelo 3D sem contraste"
# (a Lovable identificou, com o .glb inspecionado direto, que 17 dos 18
# primitivos saíam sem material nenhum -- todos caindo no material
# "Default" branco liso do glTF, indistinguíveis entre si e do fundo).
# Etapa A (já em produção do lado do app) resolvia isso no
# pós-processamento do Lovable (glb-tint.server.ts); esta é a Etapa B,
# combinada com eles: o próprio serviço passa a exportar já com
# material por tipo de nó. Quando isso estiver validado ponta a ponta,
# o pós-processamento do app vira no-op sozinho (só aplica quando o
# primitivo ainda não tem material -- não removemos nada do lado deles).
#
# NOTA: os valores de cor abaixo são um ponto de partida razoável, não
# os valores exatos que a Lovable usou no MATERIAIS_BASE deles -- para
# paridade visual entre o .glb "cru" deste serviço e o que o app já
# gerava antes, os hex precisam ser sincronizados com eles depois
# (pendência registrada no documento de continuidade, não decisão
# unilateral deste lado).
# ------------------------------------------------------------------
MATERIAIS_BASE = {
    "paredes": {"cor_rgb": (214, 214, 219), "roughness": 0.85},  # cinza claro levemente saturado
    "piso": {"cor_rgb": (120, 98, 84), "roughness": 0.6},         # tom mais escuro/quente
    "porta": {"cor_rgb": (150, 110, 70), "roughness": 0.5},       # tom de madeira neutro
    "objeto": {"cor_rgb": (176, 190, 197), "roughness": 0.4},     # tom de destaque neutro
}


def aplicar_material_base(malha, categoria):
    """Atribui um material PBR uniforme (sem textura) a uma malha gerada
    localmente (caixa/extrusão), pela categoria do nó. NÃO usar em malha
    carregada de asset real (ex: biblioteca_objetos/*.glb) -- isso
    sobrescreveria a textura própria do asset. Chamado só nos
    fallbacks de geometria genérica."""
    cfg = MATERIAIS_BASE[categoria]
    r, g, b = cfg["cor_rgb"]
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[r / 255, g / 255, b / 255, 1.0],
        roughnessFactor=cfg["roughness"],
        metallicFactor=0.0,
    )
    malha.visual = trimesh.visual.TextureVisuals(material=material)
    return malha


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


def dividir_paredes_pelas_portas(paredes, portas, limiar_proximidade_m=0.5):
    """
    NOTA (10/08): limiar era 0.15m -- calibrado pra parede desenhada como
    linha única (porta encostada na linha central, distância ~0). Achado
    real, testando contra arquivo com parede desenhada como DUAS linhas
    paralelas (cada face, 0.15m de espessura -- Two-story-house-410202.dxf):
    todas as 5 portas detectadas por bloco ficavam entre 0.2m e 0.4m da
    parede mais próxima, ACIMA do limiar antigo -- ou seja, o corte nunca
    acontecia, silenciosamente, sem nenhum aviso. Aumentado pra 0.5m,
    com margem sobre o pior caso observado (0.4m). Ainda conservador o
    suficiente pra não associar porta a parede errada em planta densa
    (a escolha de QUAL parede é sempre "a mais próxima", o limiar só
    decide se aceita ou descarta esse match -- alargar não muda qual
    parede é escolhida, só evita descartar um match legítimo por engano).
    """
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


def criar_caixa_com_uv(tamanho=(1.0, 1.0, 1.0), centro=(0.0, 0.0, 0.0)):
    """
    Caixa 3D com UV mapping independente por face (Camada 3c, validado
    isoladamente em 05/08 antes de integrar aqui -- ver documento de
    retomada seção 11).

    Por que não usar trimesh.creation.box() direto: a caixa padrão
    compartilha vértices entre faces adjacentes (só 8 vértices no total)
    -- isso impede UV independente por face, porque o vértice de um
    canto pertenceria a 3 faces ao mesmo tempo, cada uma precisando de
    um UV diferente ali. Aqui cada face ganha seus 4 vértices próprios
    (24 no total), fisicamente no mesmo lugar mas logicamente
    independentes -- cada face pode ter sua própria textura completa
    (UV de 0,0 a 1,1).

    `tamanho`: (dx, dy, dz) -- pode ser não-cúbico, objetos reais da
    cena raramente são cubos perfeitos (ex: espelho é fino e alto).
    `centro`: posição real do objeto na cena -- normalmente o centro da
    bounding box do objeto original que está sendo substituído (a
    posição costuma estar embutida nos vértices, não no transform do
    grafo da cena, que geralmente vem identidade).
    """
    dx, dy, dz = tamanho
    sx, sy, sz = dx / 2, dy / 2, dz / 2

    faces_vertices = {
        "+X": [(sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz), (sx, -sy, sz)],
        "-X": [(-sx, sy, -sz), (-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz)],
        "+Y": [(sx, sy, -sz), (-sx, sy, -sz), (-sx, sy, sz), (sx, sy, sz)],
        "-Y": [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz)],
        "+Z": [(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)],
        "-Z": [(-sx, sy, -sz), (sx, sy, -sz), (sx, -sy, -sz), (-sx, -sy, -sz)],
    }

    vertices, faces, uv = [], [], []
    for cantos in faces_vertices.values():
        base = len(vertices)
        vertices.extend(cantos)
        uv.extend([(0, 0), (1, 0), (1, 1), (0, 1)])
        faces.append([base + 0, base + 1, base + 2])
        faces.append([base + 0, base + 2, base + 3])

    vertices = np.array(vertices, dtype=float) + np.array(centro, dtype=float)
    malha = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=int), process=False)
    return malha, np.array(uv, dtype=float)


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