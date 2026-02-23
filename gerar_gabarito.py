import os
import csv
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageFont


# =========================
# PERGUNTA QUANTAS QUESTÕES
# =========================
def perguntar_int(msg: str, minimo: int = 1) -> int:
    while True:
        try:
            n = int(input(msg).strip())
            if n < minimo:
                raise ValueError
            return n
        except ValueError:
            print(f"Digite um número válido (>= {minimo}).")


QUESTOES = perguntar_int("Quantas questões terá o gabarito? ", minimo=1)

# =========================
# CONFIGURAÇÕES PRINCIPAIS
# =========================
LARGURA = 700
ALTERNATIVAS = ["A", "B", "C", "D"]

MARGEM_SUPERIOR = 180  # espaço para nome + série/turma + QR
ESPACO_ENTRE_QUESTOES = 100
ESPACO_ENTRE_ALTERNATIVAS = 100
RAIO_CIRCULO = 20
MARGEM_INFERIOR = 100

TAM_MARCADOR = 30
RESERVA_QR_DIREITA = 180  # área fixa do QR na direita
X_INICIO_ALTERNATIVAS = 100

OUT_DIR = "saida_gabaritos"
CSV_PATH = "alunos/alunos.csv"

PROVA_DATA = "2026-02-23"
PROVA_SEQ_START = 1


# =========================
# FONTES (UNICODE / ACENTOS)
# =========================
def descobrir_fonte_ttf() -> str:
    """
    Tenta achar uma fonte TTF que suporte acentos/Unicode no sistema.
    Ajuste manual se quiser.
    """
    sistema = platform.system().lower()

    candidatos = []
    if "windows" in sistema:
        candidatos = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]
    elif "darwin" in sistema or "mac" in sistema:
        candidatos = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        ]
    else:
        # Linux
        candidatos = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]

    for p in candidatos:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Não achei uma fonte TTF padrão no seu sistema.\n"
        "Defina FONT_PATH manualmente (ex: Arial/DejaVuSans)."
    )


FONT_PATH = descobrir_fonte_ttf()


# =========================
# FUNÇÕES
# =========================
def ler_alunos_csv(csv_path: str):
    """
    Espera um CSV com colunas: aluno, serie, turma

    Exemplo:
    aluno,serie,turma
    Carlos Eduardo Silva,2º Ano,A
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Arquivo CSV não encontrado: {csv_file.resolve()}\n"
            "Crie um 'alunos.csv' com colunas: aluno, serie, turma"
        )

    alunos = []
    # utf-8-sig ajuda quando o CSV vem do Excel (BOM)
    with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        obrigatorias = {"aluno", "serie", "turma"}
        if not reader.fieldnames or not obrigatorias.issubset(set(reader.fieldnames)):
            raise ValueError("O CSV precisa ter as colunas: aluno, serie, turma")

        for row in reader:
            nome = (row.get("aluno") or "").strip()
            serie = (row.get("serie") or "").strip()
            turma = (row.get("turma") or "").strip()
            if nome:
                alunos.append({"aluno": nome, "serie": serie, "turma": turma})

    return alunos


def sanitizar_nome_arquivo(nome: str) -> str:
    permitido = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-"
    limpo = "".join(c for c in nome if c in permitido).strip()
    limpo = limpo.replace(" ", "_")
    return limpo or "aluno"


def gerar_qr_imagem(payload: dict, tamanho_px: int = 120) -> np.ndarray:
    """Gera QR (JSON) e devolve como imagem OpenCV BGR."""
    data_str = json.dumps(payload, ensure_ascii=False)

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=2,
    )
    qr.add_data(data_str)
    qr.make(fit=True)

    pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    pil_img = pil_img.resize((tamanho_px, tamanho_px))

    qr_np = np.array(pil_img)
    return cv2.cvtColor(qr_np, cv2.COLOR_RGB2BGR)


def put_text_utf8(img_bgr: np.ndarray, texto: str, x: int, y: int, font_size: int = 28, color_bgr=(0, 0, 0)):
    """
    Escreve texto com acentos/Unicode (PIL) em cima de uma imagem OpenCV (BGR).
    (x, y) é o topo-esquerdo do texto.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = ImageFont.truetype(FONT_PATH, font_size)

    # PIL usa RGB
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text((x, y), texto, font=font, fill=color_rgb)

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out_bgr


def medir_texto_pil(texto: str, font_size: int) -> tuple[int, int]:
    """Mede largura/altura do texto usando PIL (preciso para reduzir fonte)."""
    font = ImageFont.truetype(FONT_PATH, font_size)
    # bbox: (left, top, right, bottom)
    bbox = font.getbbox(texto)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h


def put_text_utf8_ajustado(img_bgr: np.ndarray, texto: str, x: int, y: int, largura_max: int, font_size_inicial: int, font_size_min: int = 14):
    """
    Escreve texto (UTF-8) reduzindo o tamanho da fonte até caber em largura_max.
    """
    fs = font_size_inicial
    while fs >= font_size_min:
        w, _ = medir_texto_pil(texto, fs)
        if w <= largura_max:
            return put_text_utf8(img_bgr, texto, x, y, font_size=fs)
        fs -= 1

    # Se não couber nem com mínimo, corta o texto
    txt = texto
    while len(txt) > 3:
        txt = txt[:-1]
        w, _ = medir_texto_pil(txt + "...", font_size_min)
        if w <= largura_max:
            return put_text_utf8(img_bgr, txt + "...", x, y, font_size=font_size_min)

    return put_text_utf8(img_bgr, "...", x, y, font_size=font_size_min)


def gerar_gabarito_para_aluno(nome_aluno: str, serie: str, turma: str, prova_id: str, out_path: str):
    altura = MARGEM_SUPERIOR + (QUESTOES * ESPACO_ENTRE_QUESTOES) + MARGEM_INFERIOR
    imagem = np.ones((altura, LARGURA, 3), dtype=np.uint8) * 255

    # QR com aluno/serie/turma/provaId
    payload = {"aluno": nome_aluno, "serie": serie, "turma": turma, "provaId": prova_id}
    qr_img = gerar_qr_imagem(payload, tamanho_px=120)
    qr_h, qr_w = qr_img.shape[:2]

    # QR centralizado na reserva direita
    x_qr = LARGURA - RESERVA_QR_DIREITA + (RESERVA_QR_DIREITA - qr_w) // 2
    y_qr = 20
    imagem[y_qr:y_qr + qr_h, x_qr:x_qr + qr_w] = qr_img

    # Texto com acentos (PIL)
    largura_texto_max = (LARGURA - RESERVA_QR_DIREITA) - 40

    imagem = put_text_utf8_ajustado(
        imagem,
        f"nome: {nome_aluno}",
        x=30,
        y=45,
        largura_max=largura_texto_max,
        font_size_inicial=30,
        font_size_min=16
    )

    imagem = put_text_utf8_ajustado(
        imagem,
        f"série: {serie}    turma: {turma}",
        x=30,
        y=85,
        largura_max=largura_texto_max,
        font_size_inicial=24,
        font_size_min=14
    )

    # provaId abaixo do QR (também via PIL pra garantir)
    imagem = put_text_utf8(imagem, prova_id, x=x_qr, y=y_qr + qr_h + 10, font_size=18)

    # Questões e alternativas (OpenCV está ok aqui)
    fonte_cv = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(QUESTOES):
        y = MARGEM_SUPERIOR + i * ESPACO_ENTRE_QUESTOES

        # número da questão
        cv2.putText(imagem, str(i + 1), (30, y + 10), fonte_cv, 1, (0, 0, 0), 2)

        for j, letra in enumerate(ALTERNATIVAS):
            x = X_INICIO_ALTERNATIVAS + j * ESPACO_ENTRE_ALTERNATIVAS

            cv2.putText(imagem, letra, (x - 10, y - 30), fonte_cv, 0.8, (0, 0, 0), 2)
            cv2.circle(imagem, (x, y), RAIO_CIRCULO, (0, 0, 0), 2)

    # Marcadores pretos nos cantos
    cv2.rectangle(imagem, (0, 0), (TAM_MARCADOR, TAM_MARCADOR), (0, 0, 0), -1)
    cv2.rectangle(imagem, (LARGURA - TAM_MARCADOR, 0), (LARGURA, TAM_MARCADOR), (0, 0, 0), -1)
    cv2.rectangle(imagem, (0, altura - TAM_MARCADOR), (TAM_MARCADOR, altura), (0, 0, 0), -1)
    cv2.rectangle(imagem, (LARGURA - TAM_MARCADOR, altura - TAM_MARCADOR), (LARGURA, altura), (0, 0, 0), -1)

    cv2.imwrite(out_path, imagem)
    print(f"Gabarito gerado: {out_path} | provaId={prova_id}")


# =========================
# EXECUÇÃO
# =========================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    alunos = ler_alunos_csv(CSV_PATH)
    if not alunos:
        raise RuntimeError("Nenhum aluno encontrado no CSV.")

    print(f"👥 {len(alunos)} alunos carregados do CSV")
    print(f"📝 Questões: {QUESTOES}")
    print(f"🔤 Fonte TTF: {FONT_PATH}")

    for idx, row in enumerate(alunos, start=PROVA_SEQ_START):
        nome = row["aluno"]
        serie = row.get("serie", "")
        turma = row.get("turma", "")

        prova_id = f"{PROVA_DATA}-{idx:03d}"
        nome_arq = sanitizar_nome_arquivo(nome)
        out_file = os.path.join(OUT_DIR, f"gabarito_{nome_arq}_{prova_id}.png")

        gerar_gabarito_para_aluno(nome, serie, turma, prova_id, out_file)

    print("\n✅ Finalizado!")
    print("📁 Pasta de saída:", os.path.abspath(OUT_DIR))
    print("📄 CSV usado:", os.path.abspath(CSV_PATH))