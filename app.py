import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

alternativas = ['A', 'B', 'C', 'D']
CORS(app, origins=["http://localhost:4200"])

def detectar_respostas(imagem, total_questoes):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=50, param2=30, minRadius=15, maxRadius=30)

    if circles is None:
        return {}

    circles = np.round(circles[0, :]).astype("int")
    circles = sorted(circles, key=lambda c: c[1])

    linhas = []
    linha_atual = []
    tolerancia_y = 30

    for circle in circles:
        if not linha_atual:
            linha_atual.append(circle)
        elif abs(circle[1] - linha_atual[-1][1]) < tolerancia_y:
            linha_atual.append(circle)
        else:
            linhas.append(linha_atual)
            linha_atual = [circle]
    if linha_atual:
        linhas.append(linha_atual)

    respostas = {}
    for i, linha in enumerate(linhas):
        if i >= total_questoes:
            break

        linha = sorted(linha, key=lambda c: c[0])
        preenchidos = []

        for j, (x, y, r) in enumerate(linha):
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.circle(mask, (x, y), r - 5, 255, -1)
            media = cv2.mean(gray, mask=mask)[0]
            preenchidos.append((j, media))

        preenchidos.sort(key=lambda t: t[1])
        index_resposta = preenchidos[0][0]
        respostas[i + 1] = alternativas[index_resposta]

    return respostas

@app.route('/corrigir', methods=['POST'])
def corrigir():
    if 'imagem' not in request.files or 'gabarito' not in request.form:
        return jsonify({'erro': 'Imagem e gabarito são obrigatórios'}), 400

    try:
        gabarito_correto = json.loads(request.form['gabarito'])
        aluno = request.form['aluno']
        gabarito_correto = {int(k): v.upper() for k, v in gabarito_correto.items()}
    except Exception:
        return jsonify({'erro': 'Gabarito inválido. Envie como JSON com chave numérica e valor tipo A/B/C/D'}), 400

    file = request.files['imagem']
    if file.filename == '':
        return jsonify({'erro': 'Nome de arquivo vazio'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    imagem = cv2.imread(filepath)
    respostas_lidas = detectar_respostas(imagem, total_questoes=len(gabarito_correto))

    certas = 0
    erradas = 0
    detalhes = {}

    for q, resp_correta in gabarito_correto.items():
        resp_aluno = respostas_lidas.get(q)
        if resp_aluno == resp_correta:
            certas += 1
            detalhes[q] = {"resposta": resp_aluno, "status": "correta"}
        else:
            erradas += 1
            detalhes[q] = {"resposta": resp_aluno if resp_aluno else "sem resposta", "status": "errada"}

    return jsonify({
        'certas': certas,
        'erradas': erradas,
        'total': len(gabarito_correto),
        'detalhes': detalhes,
        'aluno' : aluno,
    })

if __name__ == '__main__':
    app.run(debug=True)
