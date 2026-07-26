import ezdxf
import extrair_geometria as eg

doc = ezdxf.readfile('Accessible_Unisex_Bathroom_Plan (1).dxf')
linhas, arcos, poly = eg.carregar_entidades(doc)
fator, conf, exp = eg.detectar_escala(doc, arcos)
paredes, amplas, metodo = eg.extrair_paredes(linhas, fator)

print('metodo:', metodo)
print('Paredes finais:', len(paredes))
print('Paredes amplas:', len(amplas))
print('Total de linhas lidas do DXF (antes de qualquer filtro):', len(linhas))

with open('debug_amplas.txt', 'w') as f:
    f.write(f"metodo: {metodo}\n")
    f.write(f"paredes finais: {len(paredes)}\n")
    f.write(f"paredes amplas: {len(amplas)}\n\n")
    f.write("=== PAREDES AMPLAS ===\n")
    for l in amplas:
        f.write(f"{l['start']} {l['end']}\n")
    f.write("\n=== TODAS AS LINHAS LIDAS DO DXF (por layer) ===\n")
    for l in linhas:
        f.write(f"[{l['layer']}] {l['start']} {l['end']}\n")

print("Arquivo debug_amplas.txt gerado.")
