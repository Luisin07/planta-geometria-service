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
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ezdxf
import trimesh
from PIL import Image
from google import genai
from google.genai import types as genai_types
import replicate

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

# NOTA (05/08, revertido no mesmo dia): tentei configurar Supabase Storage
# direto neste serviço, mas o Lovable Cloud abstrai o Supabase -- sem
# painel externo, sem service_role key acessível fora do app. Solução
# errada, baseada em suposição não verificada sobre a arquitetura.
#
# A solução real já existe do lado do app: processarPlanta (Lovable) já
# baixa o .glb da URL efêmera devolvida por /processar-planta e regrava
# permanente no bucket floor-plans deles, como
# storage:<user>/<planta>/servico.glb -- isso já resolve o problema de
# armazenamento temporário, sem o Render precisar de nenhuma credencial
# de banco. Mais seguro também: a chave de admin nunca sai do backend
# deles.
#
# Este serviço volta a devolver uma URL efêmera local (/modelo/{arquivo}),
# que o app consome imediatamente e persiste do lado dele -- mesmo padrão
# que já funciona pra /processar-planta. Quando /aplicar-textura for
# conectado na interface, precisa do mesmo tratamento do lado do app
# (função "aplicarTextura" análoga a processarPlanta, salvando o
# resultado em floor-plans também).

ESPESSURA_PAREDE_M = 0.15
ALTURA_PAREDE_M = 2.7

FORMATOS_IMAGEM_VALIDOS = {"tiff": "TIFF", "jpeg": "JPEG", "png": "PNG"}
MEDIA_TYPES_IMAGEM = {"tiff": "image/tiff", "jpeg": "image/jpeg", "png": "image/png"}

NAO_FISICOS = {"CONCRETE", "GR1"}

MODELOS_CANDIDATOS_3B = [
    "gemini-3.5-flash",       # confirmado funcionando em 05/08
    "gemini-2.5-flash-lite",  # confirmado indisponível pra conta nova em 05/08, mantido como fallback
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

SCHEMA_OPERACAO_3B = {
    "type": "OBJECT",
    "properties": {
        "suportado": {
            "type": "BOOLEAN",
            "description": "false se o comando pedir algo que NÃO seja mudar a cor de um "
                            "objeto (ex: mover, redimensionar, trocar objeto) -- o sistema "
                            "hoje só sabe fazer mudança de cor. Nunca force um comando de "
                            "outra natureza a caber em 'cor'.",
        },
        "motivo_se_nao_suportado": {
            "type": "STRING",
            "description": "Se suportado=false, explica em uma frase o que o comando pedia "
                            "e por que está fora do escopo atual. Se suportado=true, usar "
                            "string vazia \"\".",
        },
        "objeto_id": {
            "type": "STRING",
            "description": "O campo 'id' EXATO do objeto na lista fornecida -- nunca inventar "
                            "um id. Se suportado=false, usar string vazia \"\".",
        },
        "propriedade": {
            "type": "STRING",
            "description": "Sempre 'cor' se suportado=true. Se suportado=false, usar string vazia \"\".",
        },
        "valor_hex": {
            "type": "STRING",
            "description": "Cor convertida para hex, ex: '#0000FF' para azul, se suportado=true. "
                            "Se suportado=false, usar string vazia \"\".",
        },
        "confianca": {
            "type": "STRING",
            "enum": ["alta", "media", "baixa"],
            "description": "Alta: o objeto mencionado bate claramente com um item da lista. "
                            "Baixa: ambíguo (ex: mais de um objeto poderia ser 'o espelho').",
        },
    },
    # TODOS os campos sempre obrigatórios -- ver nota no prototipo isolado
    # (descoberta-3b/interpretar_comando_gemini.py) sobre por que
    # "obrigatório condicional" só em texto não é confiável.
    "required": ["suportado", "motivo_se_nao_suportado", "objeto_id", "propriedade", "valor_hex", "confianca"],
}


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


LIMITE_CARACTERES_NOME_OBJETO = 20  # dado real (05/08): maior nome de móvel/acessório visto até
                                     # agora tem 19 chars ("TOILET ROLL HOLDER", "BABY CHANGE STATION").
                                     # Nome de cômodo e frase de instrução passam bem disso
                                     # ("ACCESSIBLE UNISEX BATHROOM"=26, "EXTENT OF BABY CHANGE
                                     # STATION WHEN IN USE"=43). Não é lei universal -- é corte
                                     # calibrado no dado real que já vimos, pode precisar ajuste
                                     # se aparecer nome de móvel genuinamente longo em outro arquivo.


def _e_anotacao_tecnica(nome):
    """Filtro por padrão (lista negra), não por nome conhecido (lista
    branca) -- lista branca faria objeto de vocabulário novo (ex: 'BED'
    numa planta de quarto que nunca vimos) sumir silenciosamente do
    desenho. Isso aqui só remove lixo de anotação técnica óbvio (cota,
    percentual, cabeçalho de escala/aprovação) e texto longo demais pra
    ser nome de objeto único (nome de cômodo, frase de instrução) --
    termo técnico curto que parece nome de objeto (ex: 'EXCLUSION LINE',
    'BACK REST') ainda passa, porque não tem padrão textual que
    diferencie isso de um nome de móvel real sem olhar a camada (layer)
    de origem no DXF, que esta função não consulta hoje."""
    nome_limpo = nome.strip()
    if len(nome_limpo) > LIMITE_CARACTERES_NOME_OBJETO:
        return True
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
    portas += eg.detectar_portas_correr(doc, paredes, fator)

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

    # NOTA (10/08): fusão de parede em linha dupla aplicada SÓ aqui, no
    # pipeline 3D -- não no /gerar-2d, que já funciona bem com a lista
    # original e não foi testado com essa mudança ainda. Achado real:
    # sem isso, cada face de parede desenhada em linha dupla vira sua
    # própria caixa 3D, fragmentando a malha em 100+ pedaços
    # desconectados (confirmado: Two-story-house-410202.dxf, 132
    # componentes soltos -> 46 depois da fusão).
    paredes_fundidas = eg.mesclar_paredes_duplas(paredes, fator)
    paredes_fundidas_m = [
        {"start": [p * fator for p in l["start"]], "end": [p * fator for p in l["end"]]}
        for l in paredes_fundidas
    ]
    segmentos_3d, _ = g3d.dividir_paredes_pelas_portas(paredes_fundidas_m, g["portas_m"])

    mesh_paredes = g3d.gerar_paredes_3d(segmentos_3d, espessura_m=ESPESSURA_PAREDE_M, altura_m=ALTURA_PAREDE_M)
    piso = g3d.gerar_piso(g["paredes_m"])
    # NOTA (10/08): parede e piso eram fundidos num nó único ("paredes")
    # antes desta mudança -- impedia dar cor/textura no piso sem afetar a
    # parede junto (e vice-versa). Separados em dois nós agora, cada um
    # endereçável independente. Decisão consciente de não separar POR
    # SEGMENTO de parede ainda (arquitetos veem parede como bloco único,
    # confirmado com a dona do projeto em 04/08 -- ver documento de
    # retomada, seção 6) -- só parede-vs-piso, não parede-vs-parede.

    TAMANHOS_OBJETO = {
        # TOILET atualizado (11/08) com proporção real, calibrada contra o
        # primeiro asset real da biblioteca (Meshy) -- era um cubo grosseiro
        # (0.4,0.4,0.4) antes de ter referência real pra calibrar contra.
        "TOILET": (0.41, 0.72, 0.75), "MIRROR": (0.5, 0.05, 0.6), "MIXER": (0.15, 0.15, 0.2),
        "BASIN": (0.5, 0.4, 0.15), "GRAB RAIL": (0.05, 0.4, 0.05), "HOOK": (0.05, 0.05, 0.05),
    }
    TAMANHO_PADRAO_OBJETO = (0.2, 0.2, 0.2)

    # ------------------------------------------------------------------
    # BIBLIOTECA_OBJETOS -- primeira entrada REAL hoje (11/08): TOILET,
    # gerado no Meshy, remesh pra ~9.6k faces (original tinha ~2 milhões,
    # inviável), curado e testado.
    #
    # Decisão de onde os arquivos ficam: dentro do próprio repositório
    # deste serviço (pasta biblioteca_objetos/), não no Supabase Storage
    # do Lovable -- porque o Render não tem (e não deveria ter, decisão
    # de 05/08) credencial pra acessar o Storage do Lovable Cloud
    # diretamente. Biblioteca compartilhada entre TODOS os clientes (não
    # é dado por usuário), então faz sentido morar junto do código, versionada.
    #
    # ACHADO REAL (11/08): o asset do Meshy vem em convenção Y-up (Y é
    # "altura" -- confirmado visualmente, projetando a nuvem de vértices
    # e reconhecendo a silhueta do vaso de lado no plano XZ). Este
    # serviço trabalha inteiro em Z-up (convenção CAD). Sem converter,
    # o objeto entraria deitado na cena -- mesma categoria de bug do
    # Z-up/Y-up resolvido ontem pro modelo inteiro, agora no nível do
    # objeto individual. Conversão: (x,y,z) -> (x,-z,y), rotação de
    # verdade (determinante +1, não espelha), confirmada visualmente
    # antes de aplicar.
    #
    # LIMITAÇÃO CONHECIDA, NÃO RESOLVIDA: rotação de FRENTE do objeto.
    # A conversão de eixo acima resolve "objeto em pé", não "objeto
    # virado pro lado certo do ambiente" -- isso continua exigindo
    # detecção por bloco nomeado do DXF (não construído), objeto entra
    # sempre virado numa direção fixa por enquanto.
    # ------------------------------------------------------------------
    # DESATIVADO TEMPORARIAMENTE (11/08, à noite): serviço entrou em crash
    # loop repetido em produção logo após esse arquivo ser adicionado.
    # Suspeita forte: textura de 4096x4096 do toilet.glb (28MB no total)
    # estourando os 512MB de RAM do plano Starter ao processar planta com
    # objeto TOILET. Revertido pra restaurar estabilidade primeiro,
    # investigar/reduzir textura depois, sem pressão de serviço fora do ar.
    BIBLIOTECA_OBJETOS = {
        # "TOILET": {"arquivo": "biblioteca_objetos/toilet.glb"},
    }

    def _converter_y_up_para_z_up(malha):
        """(x,y,z) -> (x,-z,y) -- confirmado visualmente (11/08) que o
        asset do Meshy é Y-up; este serviço é Z-up inteiro."""
        v = malha.vertices.copy()
        malha.vertices = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])
        return malha

    def _criar_geometria_objeto(nome, dx, dy, dz):
        """
        Tenta biblioteca de modelo real primeiro; cai pra caixa genérica
        se o tipo não estiver na biblioteca OU o arquivo não existir no
        disco (fallback duplo, de propósito -- nunca quebra o pipeline
        por causa de asset faltando ou mal configurado).
        """
        entrada = BIBLIOTECA_OBJETOS.get(nome)
        if entrada:
            caminho_asset = entrada["arquivo"]
            if os.path.exists(caminho_asset):
                malha = trimesh.load(caminho_asset, force="mesh")
                malha = _converter_y_up_para_z_up(malha)
                # Escala uniforme (preserva proporção real do asset) pela
                # maior dimensão -- decisão não validada com asset real
                # ainda, pode precisar ajuste quando houver modelo de
                # verdade pra comparar visualmente.
                escala = max(dx, dy, dz) / max(malha.extents)
                malha.apply_scale(escala)
                return malha
        return trimesh.creation.box(extents=[dx, dy, dz])

    # ------------------------------------------------------------------
    # Monta a cena com NÓS SEPARADOS (contrato confirmado com o app):
    # - paredes: nó próprio, um objeto por segmento na lista "objetos"
    #   (todos apontando pro mesmo node "paredes" -- parede continua
    #   bloco único visualmente, só não funde mais com o piso)
    # - piso: nó próprio, endereçável, um item só na lista "objetos"
    # - cada porta é seu próprio nó
    # - cada objeto/móvel é seu próprio nó
    # Cada item do array "objetos" da resposta carrega um campo "node" com
    # o nome LITERAL do nó no .glb -- o app casa por esse nome, sem adivinhar.
    # ------------------------------------------------------------------
    scene = trimesh.Scene()
    scene.add_geometry(mesh_paredes, node_name="paredes")
    scene.add_geometry(piso, node_name="piso")

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

    objetos.append({
        "id": "piso",
        "tipo": "piso",
        "node": "piso",
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
        if _e_anotacao_tecnica(o["nome"]):
            continue
        node_name = f"objeto_{i}"
        x, y = o["posicao"][0] * fator, o["posicao"][1] * fator
        dx, dy, dz = TAMANHOS_OBJETO.get(o["nome"], TAMANHO_PADRAO_OBJETO)
        caixa = _criar_geometria_objeto(o["nome"], dx, dy, dz)
        # ATENÇÃO (10/08, não resolvido): assume que a geometria está
        # centralizada na origem (verdade pra trimesh.creation.box(), não
        # necessariamente verdade pra um asset carregado de arquivo -- um
        # modelo real pode ter a origem na base, não no centro). Só
        # importa quando BIBLIOTECA_OBJETOS tiver entrada de verdade;
        # validar visualmente assim que houver o primeiro asset real.
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

    deteccao_estruturas = eg.detectar_estruturas_desconectadas(paredes, fator)

    return {
        "escala": {"fator_para_metros": fator, "confianca": g["confianca"]},
        "paredes_qtd": len(paredes),
        "portas_qtd": len(portas),
        "objetos_qtd": caixas_objetos_qtd,
        "nome_comodo": g["nome_comodo"],
        "envelope_m": g["envelope_m"],
        "objetos": objetos,
        "modelo_id": f"{id_modelo}.glb",  # mesmo valor usado no path de /modelo/{arquivo} e em /aplicar-textura
        "modelo_3d_url": f"/modelo/{id_modelo}.glb",
        "modelo_3d_nos_separados": True,
        "aviso_multiplas_estruturas": {
            "nivel": deteccao_estruturas["alerta"],  # "nenhum" | "baixo" | "moderado" | "forte"
            "grupos_desconectados_qtd": deteccao_estruturas["grupos_qtd"],
            "menor_gap_entre_maiores_m": deteccao_estruturas["menor_gap_entre_maiores_m"],
            "mensagem": {
                "nenhum": None,
                "baixo": "Detectados pequenos grupos de parede desconectados -- provavelmente cômodos/alas normais, não é alerta forte.",
                "moderado": "Muitos grupos de parede desconectados detectados. Pode ser desenho com múltiplas vistas (andares, cortes) ou múltiplas unidades na mesma prancha -- considere isolar a região desejada com isolar_regiao.py antes de processar, se o resultado parecer misturado.",
                "forte": "Grupos de parede fisicamente distantes (vários metros) detectados -- muito provável que este arquivo tenha múltiplas unidades/estruturas na mesma prancha. Recomendado isolar a região desejada com isolar_regiao.py antes de processar.",
            }[deteccao_estruturas["alerta"]],
        },
        "status": "concluido",
    }


@app.get("/modelo/{nome_arquivo}")
def baixar_modelo(nome_arquivo: str):
    """
    URL efêmera -- o app precisa baixar isso e persistir do lado dele
    (padrão já implementado em processarPlanta, que salva em
    storage:<user>/<planta>/servico.glb no bucket floor-plans) logo
    após receber a resposta de /processar-planta. Este arquivo local
    some no próximo reinício/deploy do serviço.
    """
    caminho = os.path.join(PASTA_MODELOS, nome_arquivo)
    if not os.path.exists(caminho):
        raise HTTPException(status_code=404, detail="Modelo não encontrado (link efêmero pode ter expirado)")
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


class ComandoRequest(BaseModel):
    comando: str
    objetos: list[dict]


@app.post("/interpretar-comando")
async def interpretar_comando(req: ComandoRequest):
    """
    Camada 3b (fase de descoberta validada em 05/08, ver documento de
    retomada seção 11) -- traduz um comando em linguagem natural (esperado
    em português) numa operação estruturada que a Camada 3a já sabe
    executar. Hoje só suporta mudança de cor -- qualquer outro pedido
    (mover, redimensionar, trocar objeto) volta com suportado=false e um
    motivo explícito, em vez de forçar uma resposta errada.

    Input esperado: {"comando": "...", "objetos": [...]} -- a lista de
    objetos é a mesma que já vem no JSON de /processar-planta (precisa
    ser guardada pelo app depois de processar a planta, pra não precisar
    reprocessar o DXF só pra interpretar um comando).
    """
    if "GEMINI_API_KEY" not in os.environ:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY não configurada no serviço. Configure nas "
                   "variáveis de ambiente do Render antes de usar este endpoint.",
        )

    lista_objetos = "\n".join(
        f'- id="{o["id"]}", nome="{o.get("nome", o.get("tipo", ""))}", tipo="{o.get("tipo", "")}"'
        for o in req.objetos
    )
    prompt = f"""Você traduz um comando em português para uma operação estruturada
sobre um modelo 3D de planta baixa.

Objetos disponíveis nesta planta (só pode escolher um destes, pelo id exato):
{lista_objetos}

Comando do usuário: "{req.comando}"

O sistema hoje SÓ sabe executar uma operação: mudar a cor de um objeto.
Se o comando pedir qualquer outra coisa (mover, girar, redimensionar,
trocar por outro objeto, etc.), marque suportado=false e explique o
motivo -- não tente forçar o comando a caber em 'cor'.

Responda com o JSON da operação, no formato definido. Se o comando for
ambíguo mas ainda assim for sobre cor (mais de um objeto poderia
corresponder), escolha o mais provável e marque confianca="baixa"."""

    cliente = genai.Client()
    ultimo_erro = None
    for nome_modelo in MODELOS_CANDIDATOS_3B:
        try:
            resposta = cliente.models.generate_content(
                model=nome_modelo,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SCHEMA_OPERACAO_3B,
                ),
            )
            operacao = json.loads(resposta.text)
            break
        except Exception as e:
            ultimo_erro = e
            continue
    else:
        raise HTTPException(
            status_code=422,  # NAO usar 502 -- Lovable trata 502 como sinal de retry (cold start); isso aqui e falha real do provedor, retry so gasta credito de novo
            detail=f"Nenhum modelo Gemini candidato respondeu. Último erro: {ultimo_erro}",
        )

    # Mesma validação defensiva do protótipo isolado -- não confia cegamente
    # no que o modelo devolveu, mesmo com schema forçando os campos.
    if operacao.get("suportado"):
        ids_validos = {o["id"] for o in req.objetos}
        if not operacao.get("objeto_id") or operacao["objeto_id"] not in ids_validos:
            operacao["suportado"] = False
            operacao["motivo_se_nao_suportado"] = (
                "Modelo retornou objeto_id inválido ou ausente -- tratado como não suportado "
                "por segurança, não aplicar."
            )

    return operacao


@app.post("/aplicar-textura")
async def aplicar_textura(
    arquivo: UploadFile = File(...),
    node_objeto: str = Form(...),
    prompt_textura: str = Form(...),
):
    """
    Camada 3c (fase de descoberta validada em 05/08, integração em cena
    real validada localmente antes deste endpoint -- ver documento de
    retomada seções 11 e 14). Gera uma textura via Replicate e aplica
    num objeto específico dentro de um modelo 3D já processado, via UV
    mapping -- sem afetar o resto da cena.

    CORRIGIDO EM 07/08: antes recebia um "modelo_id" (referência a
    arquivo na pasta temporária local do serviço) -- quebrava sempre que
    o Render reiniciava entre o processamento original e a aplicação de
    textura (praticamente sempre, no plano gratuito). Agora é stateless
    como os outros endpoints: recebe o .glb de verdade no corpo da
    requisição (o app já tem esse arquivo persistido em
    storage:.../servico.glb -- é só reenviar os bytes, sem depender de
    o serviço "lembrar" de nada entre chamadas).

    Limitação conhecida (mesma da descoberta isolada): sem continuidade
    perfeita do padrão da textura nas quinas do objeto -- cada face
    recebe uma cópia independente e completa da textura.
    """
    if "REPLICATE_API_TOKEN" not in os.environ:
        raise HTTPException(
            status_code=503,
            detail="REPLICATE_API_TOKEN não configurada no serviço.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tmp:
        shutil.copyfileobj(arquivo.file, tmp)
        caminho_glb = tmp.name

    try:
        cena = trimesh.load(caminho_glb)
    finally:
        os.unlink(caminho_glb)

    if not isinstance(cena, trimesh.Scene):
        raise HTTPException(status_code=400, detail="Modelo não é uma cena com nós nomeados")
    if node_objeto not in cena.graph.nodes:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{node_objeto}' não existe nesta cena. "
                   f"Nós disponíveis: {list(cena.graph.nodes)}",
        )

    _transform, nome_geometria = cena.graph[node_objeto]
    geom_original = cena.geometry[nome_geometria]
    centro = tuple(geom_original.bounds.mean(axis=0))
    tamanho = tuple(geom_original.extents)

    try:
        saida_replicate = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": prompt_textura, "aspect_ratio": "1:1", "output_format": "png"},
        )
        item = saida_replicate[0] if isinstance(saida_replicate, list) else saida_replicate
        buffer_textura = io.BytesIO(item.read() if hasattr(item, "read") else
                                     __import__("urllib.request", fromlist=["urlopen"]).urlopen(str(item)).read())
        imagem_textura = Image.open(buffer_textura)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Falha ao gerar textura no Replicate: {e}")

    malha_nova, uv = g3d.criar_caixa_com_uv(tamanho=tamanho, centro=centro)
    malha_nova.visual = trimesh.visual.TextureVisuals(uv=uv, image=imagem_textura)

    # Substitui SÓ a geometria desse objeto -- resto da cena intocado
    # (validado localmente: outros nós continuam com vértices/UV originais).
    cena.geometry[nome_geometria] = malha_nova

    id_novo_modelo = str(uuid.uuid4())
    caminho_saida = os.path.join(PASTA_MODELOS, f"{id_novo_modelo}.glb")
    cena.export(caminho_saida)

    return {
        "modelo_id": f"{id_novo_modelo}.glb",
        "modelo_3d_url": f"/modelo/{id_novo_modelo}.glb",
        "node_texturizado": node_objeto,
        "status": "concluido",
    }


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