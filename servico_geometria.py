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
    endereçáveis) + URL do modelo .glb gerado
"""

import os
import uuid
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import ezdxf

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
    portas = eg.detectar_portas(doc, arcos, paredes_amplas, fator)

    segmentos, qtd_cortadas = g3d.dividir_paredes_pelas_portas(
        [{"start": [p * fator for p in l["start"]], "end": [p * fator for p in l["end"]]} for l in paredes],
        [{**p, "posicao": [c * fator for c in p["posicao"]]} for p in portas],
    )
    mesh_paredes = g3d.gerar_paredes_3d(segmentos, espessura_m=0.15, altura_m=2.7)
    piso = g3d.gerar_piso(
        [{"start": [p * fator for p in l["start"]], "end": [p * fator for p in l["end"]]} for l in paredes]
    )
    import trimesh

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

    caixas_objetos = []
    for o in objetos_texto:
        if o["nome"] in NAO_FISICOS or o["nome"] == NOME_COMODO:
            continue
        x, y = o["posicao"][0] * fator, o["posicao"][1] * fator
        dx, dy, dz = TAMANHOS_OBJETO.get(o["nome"], TAMANHO_PADRAO_OBJETO)
        caixa = trimesh.creation.box(extents=[dx, dy, dz])
        caixa.apply_translation([x, y, dz / 2])
        caixas_objetos.append(caixa)

    modelo = trimesh.util.concatenate([mesh_paredes, piso] + caixas_objetos)

    id_modelo = str(uuid.uuid4())
    caminho_glb = os.path.join(PASTA_MODELOS, f"{id_modelo}.glb")
    modelo.export(caminho_glb)

    objetos = []
    for i, l in enumerate(paredes):
        objetos.append({
            "id": f"parede-{i}",
            "tipo": "parede",
            "x1": round(l["start"][0] * fator, 4),
            "y1": round(l["start"][1] * fator, 4),
            "x2": round(l["end"][0] * fator, 4),
            "y2": round(l["end"][1] * fator, 4),
        })
    for i, p in enumerate(portas):
        objetos.append({
            "id": f"porta-{i}",
            "tipo": "porta",
            "x": round(p["posicao"][0] * fator, 4),
            "y": round(p["posicao"][1] * fator, 4),
            "largura_m": p.get("largura_estimada_m"),
        })
    for i, o in enumerate(objetos_texto):
        if o["nome"] in NAO_FISICOS or o["nome"] == NOME_COMODO:
            continue
        objetos.append({
            "id": f"objeto-{i}",
            "tipo": "objeto",
            "nome": o["nome"],
            "x": round(o["posicao"][0] * fator, 4),
            "y": round(o["posicao"][1] * fator, 4),
        })

    os.unlink(caminho_dxf)

    return {
        "escala": {"fator_para_metros": fator, "confianca": confianca},
        "paredes_qtd": len(paredes),
        "portas_qtd": len(portas),
        "objetos_qtd": len(caixas_objetos),
        "nome_comodo": NOME_COMODO,
        "envelope_m": {
            "largura": round((max(xs) - min(xs)) * fator, 2),
            "altura": round((max(ys) - min(ys)) * fator, 2),
        },
        "objetos": objetos,
        "modelo_3d_url": f"/modelo/{id_modelo}.glb",
        "modelo_3d_nos_separados": False,  # ver nota registrada no Lovable
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
