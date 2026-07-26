"""
API do serviço de geometria -- empacota extrair_geometria.py e
gerar_modelo_3d.py como um serviço web, pronto pra hospedar.

ESCOPO ATUAL (deliberado, não esconder): aceita DXF diretamente.
Conversão DWG->DXF ainda é manual (ODA File Converter), fora deste
serviço. Ver nota no final do arquivo sobre por quê.

Uso local (teste antes de hospedar):
    pip install fastapi uvicorn python-multipart
    uvicorn servico_geometria:app --reload --port 8000

Endpoint principal:
    POST /processar-planta
    Input: arquivo .dxf (multipart/form-data, campo "arquivo")
    Output: JSON no contrato PlantaProcessada (agregados + objetos
    endereçáveis, cada um com campo "node" explícito) + URL do modelo .glb
"""

import os
import uuid
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import ezdxf
import trimesh

import extrair_geometria as eg
import gerar_modelo_3d as g3d

app = FastAPI(title="Serviço de Geometria -- Conversor CAD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajustar pra origem real do Lovable em produção
    allow_methods=["*"],
    allow_headers=["*"],
)

PASTA_MODELOS = os.path.join(tempfile.gettempdir(), "modelos_3d")
os.makedirs(PASTA_MODELOS, exist_ok=True)

ESPESSURA_PAREDE_M = 0.15
ALTURA_PAREDE_M = 2.7


@app.get("/saude")
def saude():
    """Endpoint simples pra confirmar que o serviço está de pé."""
    return {"status": "ok"}


@app.post("/processar-planta")
async def processar_planta(arquivo: UploadFile = File(...)):
    if not arquivo.filename.lower().endswith(".dxf"):
        raise HTTPException(
            status_code=400,
            detail="Só arquivos .dxf são aceitos por enquanto. Converta o DWG "
                   "para DXF antes de enviar (ODA File Converter).",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        shutil.copyfileobj(arquivo.file, tmp)
        caminho_dxf = tmp.name

    try:
        doc = ezdxf.readfile(caminho_dxf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Não consegui ler o DXF: {e}")

    linhas, arcos, polylines = eg.carregar_entidades(doc)
    fator, confianca, explicacao = eg.detectar_escala(doc, arcos)
    paredes, paredes_amplas, metodo_parede = eg.extrair_paredes(linhas, fator)
    portas = eg.detectar_portas(doc, arcos, paredes_amplas, fator, paredes_envelope=paredes)

    paredes_m = [{"start": [p * fator for p in l["start"]], "end": [p * fator for p in l["end"]]} for l in paredes]
    portas_m = [{**p, "posicao": [c * fator for c in p["posicao"]]} for p in portas]

    segmentos, qtd_cortadas = g3d.dividir_paredes_pelas_portas(paredes_m, portas_m)
    mesh_paredes = g3d.gerar_paredes_3d(segmentos, espessura_m=ESPESSURA_PAREDE_M, altura_m=ALTURA_PAREDE_M)
    piso = g3d.gerar_piso(paredes_m)
    mesh_paredes_piso = trimesh.util.concatenate([mesh_paredes, piso])

    xs = [p for l in paredes for p in (l["start"][0], l["end"][0])]
    ys = [p for l in paredes for p in (l["start"][1], l["end"][1])]

    objetos_texto = eg.detectar_objetos_por_texto(
        doc,
        envelope=(min(xs), max(xs), min(ys), max(ys)),
        margem_m=0.3,
        fator_para_metros=fator,
    )
    NAO_FISICOS = {"CONCRETE", "GR1"}
    NOME_COMODO = None
    for o in objetos_texto:
        if o["nome"].upper() in ("ACCESSIBLE UNISEX BATHROOM",) or (
            NOME_COMODO is None and len(o["nome"]) > 15 and o["nome"].isupper()
        ):
            NOME_COMODO = o["nome"]

    TAMANHOS_OBJETO = {
        "TOILET": (0.4, 0.4, 0.4), "MIRROR": (0.5, 0.05, 0.6), "MIXER": (0.15, 0.15, 0.2),
        "BASIN": (0.5, 0.4, 0.15), "GRAB RAIL": (0.05, 0.4, 0.05), "HOOK": (0.05, 0.05, 0.05),
    }
    TAMANHO_PADRAO_OBJETO = (0.2, 0.2, 0.2)

    # ------------------------------------------------------------------
    # Monta a cena com NÓS SEPARADOS (contrato confirmado com o app):
    # - paredes + piso fundidos num único nó "paredes"
    # - cada porta é seu próprio nó
    # - cada objeto/móvel é seu próprio nó
    # Cada item do array "objetos" da resposta carrega um campo "node" com
    # o nome LITERAL do nó no .glb -- o app casa por esse nome, sem adivinhar.
    # Nomes de nó devem ser únicos (o app indexa por nome, última vitória).
    # ------------------------------------------------------------------
    scene = trimesh.Scene()
    scene.add_geometry(mesh_paredes_piso, node_name="paredes")

    objetos = []

    for i, l in enumerate(paredes):
        objetos.append({
            "id": f"parede-{i}",
            "tipo": "parede",
            "node": "paredes",  # ainda fundida -- granularidade por segmento fica para depois
            "x1": round(l["start"][0] * fator, 4),
            "y1": round(l["start"][1] * fator, 4),
            "x2": round(l["end"][0] * fator, 4),
            "y2": round(l["end"][1] * fator, 4),
        })

    for i, p in enumerate(portas):
        node_name = f"porta_{i}"
        mesh_porta = g3d.gerar_porta_3d(p, espessura_parede_m=ESPESSURA_PAREDE_M)
        scene.add_geometry(mesh_porta, node_name=node_name)
        objetos.append({
            "id": f"porta-{i}",
            "tipo": "porta",
            "node": node_name,
            "x": round(p["posicao"][0] * fator, 4),
            "y": round(p["posicao"][1] * fator, 4),
            "largura_m": p.get("largura_estimada_m"),
        })

    caixas_objetos_qtd = 0
    for i, o in enumerate(objetos_texto):
        if o["nome"] in NAO_FISICOS or o["nome"] == NOME_COMODO:
            continue
        node_name = f"objeto_{i}"
        x, y = o["posicao"][0] * fator, o["posicao"][1] * fator
        dx, dy, dz = TAMANHOS_OBJETO.get(o["nome"], TAMANHO_PADRAO_OBJETO)
        caixa = trimesh.creation.box(extents=[dx, dy, dz])
        caixa.apply_translation([x, y, dz / 2])
        scene.add_geometry(caixa, node_name=node_name)
        caixas_objetos_qtd += 1
        objetos.append({
            "id": f"objeto-{i}",
            "tipo": "objeto",
            "node": node_name,
            "nome": o["nome"],
            "x": round(x, 4),
            "y": round(y, 4),
        })

    id_modelo = str(uuid.uuid4())
    caminho_glb = os.path.join(PASTA_MODELOS, f"{id_modelo}.glb")
    scene.export(caminho_glb)

    os.unlink(caminho_dxf)

    return {
        "escala": {"fator_para_metros": fator, "confianca": confianca},
        "paredes_qtd": len(paredes),
        "portas_qtd": len(portas),
        "objetos_qtd": caixas_objetos_qtd,
        "nome_comodo": NOME_COMODO,
        "envelope_m": {
            "largura": round((max(xs) - min(xs)) * fator, 2),
            "altura": round((max(ys) - min(ys)) * fator, 2),
        },
        "objetos": objetos,
        "modelo_3d_url": f"/modelo/{id_modelo}.glb",
        "modelo_3d_nos_separados": True,  # agora true -- portas e objetos têm nó próprio
        "status": "concluido",
    }


@app.get("/modelo/{nome_arquivo}")
def baixar_modelo(nome_arquivo: str):
    caminho = os.path.join(PASTA_MODELOS, nome_arquivo)
    if not os.path.exists(caminho):
        raise HTTPException(status_code=404, detail="Modelo não encontrado")
    return FileResponse(caminho, media_type="model/gltf-binary")


# NOTA SOBRE O ESCOPO DXF-ONLY:
# Conversão DWG->DXF automatizada exigiria ODA File Converter (ferramenta
# GUI, difícil de rodar num servidor Linux sem X11) ou reimplementar via
# Autodesk Platform Services (que já validamos funcionar, mas adiciona
# outra chamada de API externa + autenticação + espera assíncrona dentro
# deste serviço). Decisão de hoje: manter DXF-only por enquanto, usuário
# converte manualmente antes de enviar. Registrar como melhoria futura.

# NOTA SOBRE GRANULARIDADE DE NÓS (26/07):
# Paredes continuam fundidas num nó único "paredes" -- coloração de parede
# individual ainda não é possível. Se/quando for necessário, cada segmento
# de parede precisaria virar seu próprio nó (mesmo padrão usado aqui para
# porta/objeto), o que aumenta a contagem de nós no .glb consideravelmente
# em plantas grandes (ex: 133 segmentos na "Two-story-house") -- avaliar
# impacto de performance antes de fazer isso.
