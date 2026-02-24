from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import os
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
import zipfile

# ✅ IMPORTA DO SEU ARQUIVO DE PROCESSAMENTO (ajuste se o nome do arquivo for outro)
from gerar_gabarito import (
    gerar_pdf_cartao,
    PROVA_DATA,
    PROVA_SEQ_START
)

from ler_gabarito import (
    ler_gabarito_csv,
    carregar_imagem_bytes,
    corrigir_imagem,
)
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUT_BASE = BASE_DIR / "saida_api"

# ✅ AJUSTE este caminho para onde está seu CSV do gabarito
GABARITO_CSV_PATH = BASE_DIR / "resposta" / "gabarito.csv"

# ✅ Cache do gabarito pra não ler o CSV a cada request
_GABARITO_CACHE = None


def get_gabarito():
    global _GABARITO_CACHE
    if _GABARITO_CACHE is None:
        if not GABARITO_CSV_PATH.exists():
            raise FileNotFoundError(f"Gabarito CSV não encontrado: {GABARITO_CSV_PATH}")
        _GABARITO_CACHE = ler_gabarito_csv(str(GABARITO_CSV_PATH))
    return _GABARITO_CACHE
app = FastAPI()

# CORS: enquanto testa, deixe aberto. Depois eu te mostro como travar no seu domínio.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/gerar")
def gerar():
    """
    Recebe JSON no formato:
    [
      {"serie":"3º Ano","turma":"A","quantidade":4},
      ...
    ]
    Retorna um zip com os PDFs.
    """
    turmas = request.get_json(silent=True)

    if not isinstance(turmas, list) or len(turmas) == 0:
        return jsonify({"error": "Envie um JSON array com turmas"}), 400

    # valida mínimo
    for t in turmas:
        if not isinstance(t, dict):
            return jsonify({"error": "Cada item deve ser objeto"}), 400
        if not str(t.get("serie", "")).strip():
            return jsonify({"error": "Campo 'serie' obrigatório"}), 400
        if not str(t.get("turma", "")).strip():
            return jsonify({"error": "Campo 'turma' obrigatório"}), 400
        try:
            qtd = int(t.get("quantidade", 0))
        except Exception:
            return jsonify({"error": "Campo 'quantidade' deve ser número"}), 400
        if qtd <= 0:
            return jsonify({"error": "Campo 'quantidade' deve ser > 0"}), 400

    # cria pasta por job
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = OUT_BASE / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    contador_global = PROVA_SEQ_START

    # gera PDFs
    for turma in turmas:
        serie = str(turma["serie"]).strip()
        nome_turma = str(turma["turma"]).strip()
        qtd = int(turma["quantidade"])

        for i in range(1, qtd + 1):
            numero = str(i)
            prova_id = f"{PROVA_DATA}-{contador_global:03d}"

            nome_arquivo = f"cartao_{serie.replace(' ', '')}_T{nome_turma}_{numero}_{prova_id}.pdf"
            out_file = out_dir / nome_arquivo

            gerar_pdf_cartao(str(out_file), serie, nome_turma, numero)
            contador_global += 1

    # cria ZIP
    zip_path = OUT_BASE / f"cartoes_{job_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pdf in out_dir.glob("*.pdf"):
            z.write(pdf, arcname=pdf.name)

    return send_file(zip_path, as_attachment=True, download_name=zip_path.name)


@app.post("/corrigir")
def corrigir():
    # 1) arquivo
    if "file" not in request.files:
        return jsonify({"error": "Envie o arquivo no campo 'file'"}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "Arquivo inválido"}), 400

    # 2) gabarito (vem como string JSON no multipart)
    gabarito_raw = request.form.get("gabarito", "")
    if not gabarito_raw:
        return jsonify({"error": "Envie o gabarito no campo 'gabarito'"}), 400

    try:
        gabarito_dict = json.loads(gabarito_raw)
    except Exception:
        return jsonify({"error": "Campo 'gabarito' deve ser um JSON válido"}), 400

    if not isinstance(gabarito_dict, dict) or len(gabarito_dict) == 0:
        return jsonify({"error": "Gabarito inválido (envie um objeto com {\"1\":\"A\", ...})"}), 400

    # 3) normaliza gabarito -> {int: "A"}
    alternativas_validas = {"A", "B", "C", "D", "E"}
    gabarito: dict[int, str] = {}

    for k, v in gabarito_dict.items():
        try:
            q = int(k)
        except Exception:
            return jsonify({"error": f"Chave de questão inválida no gabarito: {k!r}"}), 400

        resp = str(v).strip().upper()
        if resp not in alternativas_validas:
            return jsonify({"error": f"Resposta inválida na questão {q}: {resp!r} (use A..E)"}), 400

        gabarito[q] = resp

    # 4) decodifica imagem e corrige
    try:
        img = carregar_imagem_bytes(f.read())

        # opcional: metadados enviados pelo Next (se você mandar)
        serie = request.form.get("serie", "")
        turma = request.form.get("turma", "")
        prova_id = request.form.get("provaId", "")

        resultado = corrigir_imagem(
            imagem_bgr=img,
            gabarito_correto=gabarito,
            usar_marcadores_cartao=True,
            serie=serie,
            turma=turma,
            prova_id=prova_id,
        )

        return jsonify(resultado)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
