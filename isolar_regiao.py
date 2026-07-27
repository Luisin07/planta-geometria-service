"""
Isola uma região de um DXF multi-vista (várias plantas/cortes na mesma
prancha), recortando por bounding box, e salva um novo arquivo .dxf menor.

Não altera nada do pipeline principal -- é uma ferramenta de preparação,
rodada manualmente antes de subir o arquivo pro serviço, exatamente como
já foi feito uma vez antes com este mesmo tipo de arquivo (ver documento
de contexto do projeto, seção 3).

Uso:
    python isolar_regiao.py caminho/arquivo.dxf MIN_X MAX_X MIN_Y MAX_Y

Exemplo (valores estimados a partir do resultado já processado -- ajustar
depois de ver o resultado):
    python isolar_regiao.py "1__Projeto-arquitetura.dxf" 38 72 8 38

Gera: <nome_original>_recortado.dxf
"""

import sys
import os
import ezdxf


def dentro_da_caixa(ponto, min_x, max_x, min_y, max_y, margem=0):
    x, y = ponto[0], ponto[1]
    return (min_x - margem) <= x <= (max_x + margem) and (min_y - margem) <= y <= (max_y + margem)


def entidade_dentro(e, min_x, max_x, min_y, max_y):
    """Critério simples: pelo menos um ponto de referência da entidade cai
    dentro da caixa. Para LINE usa start/end; para ARC/TEXT/INSERT usa o
    ponto de referência (center/insert)."""
    tipo = e.dxftype()
    try:
        if tipo == "LINE":
            return dentro_da_caixa(e.dxf.start, min_x, max_x, min_y, max_y) or \
                   dentro_da_caixa(e.dxf.end, min_x, max_x, min_y, max_y)
        if tipo == "ARC":
            return dentro_da_caixa(e.dxf.center, min_x, max_x, min_y, max_y)
        if tipo in ("TEXT", "MTEXT"):
            return dentro_da_caixa(e.dxf.insert, min_x, max_x, min_y, max_y)
        if tipo == "INSERT":
            return dentro_da_caixa(e.dxf.insert, min_x, max_x, min_y, max_y)
        if tipo in ("LWPOLYLINE", "POLYLINE"):
            try:
                pontos = [(p[0], p[1]) for p in e.get_points()]
            except AttributeError:
                pontos = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            return any(dentro_da_caixa(p, min_x, max_x, min_y, max_y) for p in pontos)
    except Exception:
        return False
    return False


def main():
    if len(sys.argv) < 6:
        print("Uso: python isolar_regiao.py caminho/arquivo.dxf MIN_X MAX_X MIN_Y MAX_Y")
        sys.exit(1)

    caminho = sys.argv[1]
    min_x, max_x, min_y, max_y = (float(v) for v in sys.argv[2:6])

    doc_original = ezdxf.readfile(caminho)
    msp_original = doc_original.modelspace()

    doc_novo = ezdxf.new(dxfversion=doc_original.dxfversion)
    msp_novo = doc_novo.modelspace()

    # copia blocos referenciados (necessário para INSERT de porta funcionar)
    for nome_bloco in doc_original.blocks.block_names():
        if nome_bloco not in doc_novo.blocks:
            try:
                doc_novo.blocks.new(name=nome_bloco)
            except Exception:
                pass

    contagem_por_tipo = {}
    total_copiadas = 0

    for e in msp_original:
        if entidade_dentro(e, min_x, max_x, min_y, max_y):
            try:
                msp_novo.add_foreign_entity(e)
                total_copiadas += 1
                contagem_por_tipo[e.dxftype()] = contagem_por_tipo.get(e.dxftype(), 0) + 1
            except Exception:
                pass  # alguma entidade pode não ser copiável diretamente; ignora e segue

    base = os.path.splitext(os.path.basename(caminho))[0]
    saida = f"{base}_recortado.dxf"
    doc_novo.saveas(saida)

    print(f"Região: X [{min_x}, {max_x}]  Y [{min_y}, {max_y}]")
    print(f"Entidades copiadas: {total_copiadas}")
    for tipo, qtd in sorted(contagem_por_tipo.items()):
        print(f"  {tipo}: {qtd}")
    print(f"\nArquivo gerado: {saida}")
    print("Agora rode: python extrair_geometria.py \"" + saida + "\"")


if __name__ == "__main__":
    main()
