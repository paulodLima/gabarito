from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import json
import zipfile

# ✅ IMPORTA DO SEU ARQUIVO DE PROCESSAMENTO (ajuste se o nome do arquivo for outro)
print(">>> Importando gerar_gabarito...")
from gerar_gabarito import gerar_pdf_cartao, PROVA_DATA, PROVA_SEQ_START
print(">>> Importando ler_gabarito...")
from ler_gabarito import ler_gabarito_csv, carregar_imagem_bytes, corrigir_imagem
print(">>> Imports do projeto OK")

app = FastAPI()

# CORS (depois a gente restringe pro seu domínio do Firebase)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
OUT_BASE = BASE_DIR / "saida_api"
OUT_BASE.mkdir(parents=True, exist_ok=True)

GABARITO_CSV_PATH = BASE_DIR / "resposta" / "gabarito.csv"
_GABARITO_CACHE = None


def get_gabarito():
    global _GABARITO_CACHE
    if _GABARITO_CACHE is None:
        if not GABARITO_CSV_PATH.exists():
            raise FileNotFoundError(f"Gabarito CSV não encontrado: {GABARITO_CSV_PATH}")
        _GABARITO_CACHE = ler_gabarito_csv(str(GABARITO_CSV_PATH))
    return _GABARITO_CACHE


@app.get("/health")
def health():
    return {"ok": True}


# =========================
# /gerar
# =========================
class TurmaReq(BaseModel):
    serie: str
    turma: str
    quantidade: int


@app.post("/gerar")
def gerar(turmas: List[TurmaReq]):
    """
    Recebe JSON no formato:
    [
      {"serie":"3º Ano","turma":"A","quantidade":4},
      ...
    ]
    Retorna um zip com os PDFs.
    """

    if not turmas:
        return JSONResponse({"error": "Envie um JSON array com turmas"}, status_code=400)

    # valida mínimo
    for t in turmas:
        if int(t.quantidade) <= 0:
            return JSONResponse({"error": "Campo 'quantidade' deve ser > 0"}, status_code=400)

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = OUT_BASE / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    contador_global = PROVA_SEQ_START

    for turma in turmas:
        serie = turma.serie.strip()
        nome_turma = turma.turma.strip()
        qtd = int(turma.quantidade)

        for i in range(1, qtd + 1):
            numero = str(i)
            prova_id = f"{PROVA_DATA}-{contador_global:03d}"

            nome_arquivo = f"cartao_{serie.replace(' ', '')}_T{nome_turma}_{numero}_{prova_id}.pdf"
            out_file = out_dir / nome_arquivo

            gerar_pdf_cartao(str(out_file), serie, nome_turma, numero)
            contador_global += 1

    zip_path = OUT_BASE / f"cartoes_{job_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pdf in out_dir.glob("*.pdf"):
            z.write(pdf, arcname=pdf.name)

    return FileResponse(
        path=str(zip_path),
        filename=zip_path.name,
        media_type="application/zip",
    )


# =========================
# /corrigir
# =========================
@app.post("/corrigir")
async def corrigir(
        file: UploadFile = File(...),
        gabarito: str = Form(...),
        serie: Optional[str] = Form(""),
        turma: Optional[str] = Form(""),
        provaId: Optional[str] = Form(""),
):
    """
    multipart/form-data:
      - file: arquivo (imagem/pdf)
      - gabarito: string JSON (ex: {"1":"A","2":"B"...})
      - serie/turma/provaId (opcional)
    """

    # 1) gabarito
    try:
        gabarito_dict = json.loads(gabarito)
    except Exception:
        return JSONResponse({"error": "Campo 'gabarito' deve ser um JSON válido"}, status_code=400)

    if not isinstance(gabarito_dict, dict) or len(gabarito_dict) == 0:
        return JSONResponse({"error": "Gabarito inválido (envie um objeto com {\"1\":\"A\", ...})"}, status_code=400)

    alternativas_validas = {"A", "B", "C", "D", "E"}
    gabarito_norm: Dict[int, str] = {}

    for k, v in gabarito_dict.items():
        try:
            q = int(k)
        except Exception:
            return JSONResponse({"error": f"Chave de questão inválida no gabarito: {k!r}"}, status_code=400)

        resp = str(v).strip().upper()
        if resp not in alternativas_validas:
            return JSONResponse({"error": f"Resposta inválida na questão {q}: {resp!r} (use A..E)"}, status_code=400)

        gabarito_norm[q] = resp

    # 2) lê arquivo
    try:
        content = await file.read()
        img = carregar_imagem_bytes(content)

        resultado = corrigir_imagem(
            imagem_bgr=img,
            gabarito_correto=gabarito_norm,
            usar_marcadores_cartao=True,
            serie=serie or "",
            turma=turma or "",
            prova_id=provaId or "",
        )

        return JSONResponse(resultado)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)