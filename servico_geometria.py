"""
API do serviço de geometria -- empacota extrair_geometria.py e
gerar_modelo_3d.py como um serviço web, pronto pra hospedar.

ESCOPO ATUAL (deliberado, não esconder): aceita DXF diretamente.
Conversão DWG->DXF ainda é manual (ODA File Converter), fora deste
serviço. Ver nota no final do arquivo sobre por quê.

Uso local (teste antes de hospedar):
    pip install fastapi uvicorn python-multipart pillow
    uvicorn servico_geometria:app --reload --port 8000

Endpoints:
    POST /processar-planta   -> modelo 3D (.glb) + JSON de objetos endereçáveis
    POST /gerar-2d           -> planta baixa técnica em PNG (metros reais)
    POST /converter-imagem   -> conversão de imagem (tiff/jpeg/png), usado
                                 pela captura de topo do app
    GET  /modelo/{arquivo}   -> download do .glb gerado
    GET  /saude               -> healthcheck

NOTA DE ARQUITETURA (04/08): a extração de geometria (ler DXF, achar
parede/porta/objeto, escala) é compartilhada entre /processar-planta e
/gerar-2d através de _extrair_geometria_completa() -- antes essa lógica
só existia dentro de processar_planta e seria duplicada ao adicionar o
endpoint 2D. Um único lugar de verdade, os dois endpoints consomem o
mesmo resultado.
"""

import os
import uuid
import tempfile
import shutil
import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import ezdxf
import trimesh
from PIL import Image

import extrair_geometria as eg
import gerar_modelo_3d as g3d

app = FastAPI(title="Serviço de Geometria -- Conversor CAD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO ainda em aberto: ajustar pra origem real do Lovable em produção
    allow_methods=["*"],
    allow_headers=["*"],
)

PASTA_MODELOS = os.path.join(tempfile.gettempdir(), "modelos_3d")
os.makedirs(PASTA_MODELOS, exist_ok=True)

ESPESSURA_PAREDE_M = 0.15
ALTURA_PAREDE_M = 2.7

FORMATOS_IMAGEM_VALIDOS = {"tiff": "TIFF", "jpeg": "JPEG", "png": "PNG"}
MEDIA_TYPES_IMAGEM = {"tiff": "image/tiff", "jpeg": "image/jpeg", "png": "image/png"}

NAO_FISICOS = {"CONCRETE", "GR1"}


@app.get("/saude")
def saude():
    """Endpoint simples pra confirmar que o serviço está de pé."""
    return {"status": "ok"}


import re

PADROES_ANOTACAO_TECNICA = [
    r'^ESCALA\b',
    r'^APROVA[CÇ][AÕ]ES?\b',
    r'^CORTE\b',
    r'^A\s*=\s*[\d.,]+\s*M2$',      # rótulo de área, ex: "A=25M2"
    r'^I\s*=\s*\d+%?$',             # inclinação, ex: "I=25%"
    r'^\d+([.,]\d+)?%?$',           # número ou percentual solto
    r'^[^\wÀ-ÿ]{1,2}$',             # símbolo isolado, ex: "℄", "-"
    r'^[A-ZÀ-ÿ]{1,2}$',              # letra solta, ex: "C"/"L" (símbolo de eixo mal exportado)
]


def _e_anotacao_tecnica(nome):
    """Filtro por padrão (lista negra), não por nome conhecido (lista
    branca) -- lista branca faria objeto de vocabulário novo (ex: 'BED'
    numa planta de quarto que nunca vimos) sumir silenciosamente do
    desenho. Isso aqui só remove lixo de anotação técnica óbvio (cota,
    percentual, cabeçalho de escala/aprovação). NÃO pega tudo -- termo
    técnico que parece nome de objeto (ex: 'EXCLUSION LINE', 'BACK REST')
    ainda passa, porque não tem padrão textual que diferencie isso de um
    nome de móvel real sem olhar a camada (layer) de origem no DXF, que
    esta função não consulta hoje."""
    nome_limpo = nome.strip()
    return any(re.match(p, nome_limpo, re.IGNORECASE) for p in PADROES_ANOTACAO_TECNICA)


def _nome_comodo_mais_provavel(objetos_texto):
    """Escolhe o texto com MAIOR ALTURA DE FONTE como nome do cômodo --
    convenção real de desenho técnico (nome de cômodo é desenhado maior
    que anotação de detalhe/instrução), não suposição sobre o CONTEÚDO
    do texto. Tentativa anterior usava 'string mais longa', que se
    provou errada na prática: um texto de instrução (ex: 'EXTENT OF
    BABY CHANGE STATION WHEN IN USE', 43 caracteres) pode ser mais
    comprido que o nome do cômodo em si ('ACCESSIBLE UNISEX BATHROOM',
    26 caracteres). Ainda é heurística -- funciona bem quando o
    desenhista seguiu a convenção de fonte maior pro nome do cômodo, o
    que não é garantido em todo arquivo."""
    candidatos = [o for o in objetos_texto if len(o["nome"]) > 15 and o["nome"].isupper()]
    if not candidatos:
        return None
    return max(candidatos, key=lambda o: o.get("altura_texto", 0))["nome"]


def _extrair_geometria_completa(caminho_dxf):
    """
    Pipeline de extração compartilhado por /processar-planta e /gerar-2d.
    Lê o DXF, detecta escala, extrai parede/porta/objeto, corta os
    segmentos de parede nas portas (mesma lógica usada pra gerar o
    modelo 3D). Retorna um dicionário com tudo que os dois endpoints
    precisam -- nenhum dos dois refaz esse trabalho por conta própria.
    """
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

    xs = [p for l in paredes for p in (l["start"][0], l["end"][0])]
    ys = [p for l in paredes for p in (l["start"][1], l["end"][1])]

    objetos_texto = eg.detectar_objetos_por_texto(
        doc,
        envelope=(min(xs), max(xs), min(ys), max(ys)) if xs else (0, 0, 0, 0),
        margem_m=0.3,
        fator_para_metros=fator,
    )

    nome_comodo = _nome_comodo_mais_provavel(objetos_texto)

    objetos_fisicos = []
    for o in objetos_texto:
        if o["nome"] in NAO_FISICOS or o["nome"] == nome_comodo:
            continue
        if _e_anotacao_tecnica(o["nome"]):
            continue
        objetos_fisicos.append({
            "nome": o["nome"],
            "x": o["posicao"][0] * fator,
            "y": o["posicao"][1] * fator,
        })

    return {
        "doc": doc,
        "fator": fator,
        "confianca": confianca,
        "paredes": paredes,
        "portas": portas,
        "paredes_m": paredes_m,
        "portas_m": portas_m,
        "segmentos_m": segmentos,
        "objetos_texto": objetos_texto,
        "objetos_fisicos": objetos_fisicos,
        "nome_comodo": nome_comodo,
        "xs": xs,
        "ys": ys,
        "envelope_m": {
            "largura": round((max(xs) - min(xs)) * fator, 2) if xs else 0,
            "altura": round((max(ys) - min(ys)) * fator, 2) if ys else 0,
        },
    }


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
        g = _extrair_geometria_completa(caminho_dxf)
    finally:
        os.unlink(caminho_dxf)

    paredes, portas = g["paredes"], g["portas"]
    fator = g["fator"]

    mesh_paredes = g3d.gerar_paredes_3d(g["segmentos_m"], espessura_m=ESPESSURA_PAREDE_M, altura_m=ALTURA_PAREDE_M)
    piso = g3d.gerar_piso(g["paredes_m"])
    mesh_paredes_piso = trimesh.util.concatenate([mesh_paredes, piso])

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
    # ------------------------------------------------------------------
    scene = trimesh.Scene()
    scene.add_geometry(mesh_paredes_piso, node_name="paredes")

    objetos = []

    for i, l in enumerate(paredes):
        objetos.append({
            "id": f"parede-{i}",
            "tipo": "parede",
            "node": "paredes",
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
    for i, o in enumerate(g["objetos_texto"]):
        if o["nome"] in NAO_FISICOS or o["nome"] == g["nome_comodo"]:
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

    return {
        "escala": {"fator_para_metros": fator, "confianca": g["confianca"]},
        "paredes_qtd": len(paredes),
        "portas_qtd": len(portas),
        "objetos_qtd": caixas_objetos_qtd,
        "nome_comodo": g["nome_comodo"],
        "envelope_m": g["envelope_m"],
        "objetos": objetos,
        "modelo_3d_url": f"/modelo/{id_modelo}.glb",
        "modelo_3d_nos_separados": True,
        "status": "concluido",
    }


@app.get("/modelo/{nome_arquivo}")
def baixar_modelo(nome_arquivo: str):
    caminho = os.path.join(PASTA_MODELOS, nome_arquivo)
    if not os.path.exists(caminho):
        raise HTTPException(status_code=404, detail="Modelo não encontrado")
    return FileResponse(caminho, media_type="model/gltf-binary")


@app.post("/gerar-2d")
async def gerar_2d(arquivo: UploadFile = File(...)):
    """
    Gera a planta baixa técnica em PNG, em metros reais: parede com
    espessura real (não linha fina), vão de porta como gap natural entre
    segmentos já cortados, objeto/móvel rotulado, nome do cômodo (se
    detectado) e barra de escala. Pensado para ser entregável ao
    cliente -- diferente do desenho de debug interno do script original.
    """
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
        g = _extrair_geometria_completa(caminho_dxf)
    finally:
        os.unlink(caminho_dxf)

    buffer = io.BytesIO()
    eg.desenhar_planta_tecnica(
        segmentos_m=g["segmentos_m"],
        objetos_fisicos=g["objetos_fisicos"],
        nome_comodo=g["nome_comodo"],
        envelope_m=g["envelope_m"],
        buffer=buffer,
        espessura_m=ESPESSURA_PAREDE_M,
    )
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=planta-2d.png"},
    )


@app.post("/converter-imagem")
async def converter_imagem(
    arquivo: UploadFile = File(...),
    formato: str = Form(...),  # "tiff", "jpeg" ou "png"
):
    """
    Converte uma imagem (tipicamente a captura top-down do modelo 3D,
    enviada pelo app como PNG) pro formato pedido. Motivo de existir:
    navegador exporta PNG/JPEG nativamente via canvas, mas não exporta
    TIFF -- por isso essa conversão fica no serviço, não no Lovable.
    """
    formato = formato.lower().strip()
    if formato not in FORMATOS_IMAGEM_VALIDOS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato inválido: '{formato}'. Use tiff, jpeg ou png.",
        )

    conteudo = await arquivo.read()
    try:
        imagem = Image.open(io.BytesIO(conteudo))
        imagem.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Não consegui ler a imagem: {e}")

    if formato == "jpeg" and imagem.mode in ("RGBA", "P", "LA"):
        fundo = Image.new("RGB", imagem.size, (255, 255, 255))
        imagem = imagem.convert("RGBA")
        fundo.paste(imagem, mask=imagem.split()[-1])
        imagem = fundo

    buffer = io.BytesIO()
    imagem.save(buffer, format=FORMATOS_IMAGEM_VALIDOS[formato])
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type=MEDIA_TYPES_IMAGEM[formato],
        headers={"Content-Disposition": f"attachment; filename=captura.{formato}"},
    )


# NOTA SOBRE O ESCOPO DXF-ONLY:
# Conversão DWG->DXF automatizada exigiria ODA File Converter (ferramenta
# GUI, difícil de rodar num servidor Linux sem X11) ou reimplementar via
# Autodesk Platform Services (que já validamos funcionar, mas adiciona
# outra chamada de API externa + autenticação + espera assíncrona dentro
# deste serviço). Decisão de hoje: manter DXF-only por enquanto, usuário
# converte manualmente antes de enviar. Registrar como melhoria futura.

# NOTA SOBRE GRANULARIDADE DE NÓS (26/07):
# Paredes continuam fundidas num nó único "paredes" no .glb 3D --
# coloração de parede individual ainda não é possível ali. NÃO se aplica
# ao /gerar-2d: a dona do projeto confirmou (04/08) que arquitetos veem
# parede como bloco único mesmo, então essa "limitação" deixou de ser
# prioridade a resolver.

# NOTA SOBRE /gerar-2d (04/08):
# Reconhecimento de cômodo hoje é UM nome por arquivo inteiro
# (nome_comodo), não um polígono fechado por cômodo com objetos dentro.
# Pedido de "abrir seção por cômodo, listando o que tem dentro" foi
# identificado como feature nova, não incluída aqui -- decisão consciente
# de adiar até confirmação da dona do projeto (ver conversa 04/08).