import os
from ler_gabarito import carregar_imagem, corrigir_imagem, ler_gabarito_csv

UPLOADS = os.path.join(os.path.dirname(__file__), 'respostas')
# prefer primeiro arquivo de gabarito padrao
candidates = [

    '11409.png',
    'gabarito_simulado.png'
]
img_path = None
for c in candidates:
    p = os.path.join(UPLOADS, c)
    if os.path.exists(p):
        img_path = p
        break

if img_path is None:
    files = [f for f in os.listdir(UPLOADS) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
    if not files:
        raise SystemExit('Nenhuma imagem de teste encontrada em uploads/')
    img_path = os.path.join(UPLOADS, files[0])

print('Usando imagem de teste:', img_path)
img = carregar_imagem(img_path)
# cria um gabarito dummy com 14 questoes
gabarito = {i: 'A' for i in range(1, 15)}
resultado = corrigir_imagem(img, gabarito, usar_marcadores_cartao=True)
print('Marcadores OK:', resultado['marcadoresOk'])
print('Certas:', resultado['certas'], 'Erradas:', resultado['erradas'], 'Total:', resultado['total'])
print('Respostas lidas:')
for k, v in sorted(resultado['respostasLidas'].items()):
    print(k, v)
