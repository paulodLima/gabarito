import cv2
import numpy as np

# Parâmetros
largura = 600
questoes = 10  # Altere aqui a quantidade de questões
alternativas = ['A', 'B', 'C', 'D']

# Configurações de layout
margem_superior = 150  # aumentei para deixar espaço para o nome
espaco_entre_questoes = 100
espaco_entre_alternativas = 100
raio_circulo = 20
margem_inferior = 100

# Calcular altura dinamicamente
altura = margem_superior + (questoes * espaco_entre_questoes) + margem_inferior

# Criar imagem branca
imagem = np.ones((altura, largura, 3), dtype=np.uint8) * 255

# Fonte do texto
fonte = cv2.FONT_HERSHEY_SIMPLEX

# Adicionar campo nome no topo
texto_nome = "nome: teste"
cv2.putText(imagem, texto_nome, (30, 80), fonte, 1, (0, 0, 0), 2)

# Desenhar questões e alternativas
for i in range(questoes):
    y = margem_superior + i * espaco_entre_questoes

    # Número da questão
    cv2.putText(imagem, str(i + 1), (30, y + 10), fonte, 1, (0, 0, 0), 2)

    for j, letra in enumerate(alternativas):
        x = 100 + j * espaco_entre_alternativas

        # Letra da alternativa
        cv2.putText(imagem, letra, (x - 10, y - 30), fonte, 0.8, (0, 0, 0), 2)

        # Círculo da alternativa
        cv2.circle(imagem, (x, y), raio_circulo, (0, 0, 0), 2)

# -------------------------------
# Adicionar marcadores pretos nos 4 cantos (30x30 px)
tam_marcador = 30

# Topo esquerdo
cv2.rectangle(imagem, (0, 0), (tam_marcador, tam_marcador), (0, 0, 0), -1)

# Topo direito
cv2.rectangle(imagem, (largura - tam_marcador, 0), (largura, tam_marcador), (0, 0, 0), -1)

# Base esquerda
cv2.rectangle(imagem, (0, altura - tam_marcador), (tam_marcador, altura), (0, 0, 0), -1)

# Base direita
cv2.rectangle(imagem, (largura - tam_marcador, altura - tam_marcador), (largura, altura), (0, 0, 0), -1)

# -------------------------------

marcadas = {
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'A',
    6: 'B',
    7: 'B',
    8: 'D',
    9: 'C',
    10: 'A'
}

for q, alt in marcadas.items():
    y = margem_superior + (q - 1) * espaco_entre_questoes
    x = 100 + alternativas.index(alt) * espaco_entre_alternativas
    cv2.circle(imagem, (x, y), raio_circulo - 5, (0, 0, 0), -1)

# Salvar imagem
cv2.imwrite('gabarito_padrao_com_nome.png', imagem)
print("Imagem gabarito_padrao_com_nome.png gerada com sucesso!")
