import os
import csv
import json
from pathlib import Path
from io import BytesIO

import qrcode
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def perguntar_quantidade_questoes():
    while True:
        try:
            q = int(input("Quantas questões terá o cartão-resposta? ").strip())
            if q < 1:
                raise ValueError
            return q
        except ValueError:
            print("❌ Digite um número válido (maior que 0).")

# CONFIGURAÇÕES
# =========================
OUT_DIR = "saida_pdfs"
CSV_PATH = "alunos/alunos.csv"

# =========================
# MARCADORES AO REDOR DO CARTÃO
# =========================
CARD_MARKER_SIZE = 6 * mm          # tamanho do quadradinho preto
CARD_MARKER_OFFSET = 2 * mm        # distância do quadrado para fora da borda do cartão

# Logos (coloque seus arquivos aqui)
LOGO_ESCOLA_PATH = "logos/logo_escola.png"
LOGO_GOVERNO_PATH = "logos/logo_governo.png"

PROVA_DATA = "2026-02-23"
PROVA_SEQ_START = 1

QUESTOES = perguntar_quantidade_questoes()
ALTERNATIVAS = ["A", "B", "C", "D"]

PAGE_W, PAGE_H = A4
MARGIN = 15 * mm

# QR
QR_SIZE = 28 * mm
QR_X = PAGE_W - MARGIN - QR_SIZE
QR_Y = PAGE_H - MARGIN - 95 * mm


# =========================
# CSV alunos
# =========================
def ler_alunos_csv(csv_path: str):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Arquivo CSV não encontrado: {csv_file.resolve()}\n"
            "Crie um 'alunos.csv' com colunas: aluno, serie, turma"
        )

    alunos = []
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


# =========================
# QR code (PIL -> bytes)
# =========================
def gerar_qr_bytes(payload: dict) -> BytesIO:
    data_str = json.dumps(payload, ensure_ascii=False)

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(data_str)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    bio = BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio


# =========================
# DESENHO DO PDF
# =========================
def draw_header(c: canvas.Canvas, nome_escola: str, ano: str, aluno: str, turma: str, numero: str):
    header_h = 32 * mm
    x0 = MARGIN
    y0 = PAGE_H - MARGIN - header_h
    w = PAGE_W - 2 * MARGIN

    # Caixa externa
    c.setLineWidth(1)
    c.rect(x0, y0, w, header_h, stroke=1, fill=0)

    # LOGO ESQUERDA
    if os.path.exists(LOGO_ESCOLA_PATH):
        c.drawImage(
            LOGO_ESCOLA_PATH,
            x0 + 4 * mm,
            y0 + 6 * mm,
            width=22 * mm,
            height=20 * mm,
            preserveAspectRatio=True,
            mask="auto",
            )
    else:
        # placeholder se não achar a logo
        c.setLineWidth(1)
        c.rect(x0 + 4 * mm, y0 + 6 * mm, 22 * mm, 20 * mm, stroke=1, fill=0)
        c.setFont("Helvetica", 7)
        c.drawCentredString(x0 + 15 * mm, y0 + 16 * mm, "LOGO")

    # LOGO DIREITA
    if os.path.exists(LOGO_GOVERNO_PATH):
        c.drawImage(
            LOGO_GOVERNO_PATH,
            x0 + w - 26 * mm,
            y0 + 6 * mm,
            width=22 * mm,
            height=20 * mm,
            preserveAspectRatio=True,
            mask="auto",
            )
    else:
        c.setLineWidth(1)
        c.rect(x0 + w - 26 * mm, y0 + 6 * mm, 22 * mm, 20 * mm, stroke=1, fill=0)
        c.setFont("Helvetica", 7)
        c.drawCentredString(x0 + w - 15 * mm, y0 + 16 * mm, "LOGO")

    # Textos centrais
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(PAGE_W / 2, y0 + header_h - 8 * mm, nome_escola)

    c.setFont("Helvetica", 9)
    c.drawCentredString(PAGE_W / 2, y0 + header_h - 14 * mm, "ENSINO MÉDIO")

    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(PAGE_W / 2, y0 + header_h - 20 * mm, f"LINGUA PORTUGUESA – {ano}")

    # Campos Aluno / Turma / Nº
    fields_y = y0 - 10 * mm
    c.setFont("Helvetica", 9)

    c.drawString(MARGIN, fields_y, "Aluno(a):")
    c.line(MARGIN + 18 * mm, fields_y - 1.5 * mm, PAGE_W - MARGIN - 70 * mm, fields_y - 1.5 * mm)

    c.drawString(PAGE_W - MARGIN - 66 * mm, fields_y, "Turma:")
    c.line(PAGE_W - MARGIN - 50 * mm, fields_y - 1.5 * mm, PAGE_W - MARGIN - 28 * mm, fields_y - 1.5 * mm)

    c.drawString(PAGE_W - MARGIN - 25 * mm, fields_y, "Nº:")
    c.line(PAGE_W - MARGIN - 17 * mm, fields_y - 1.5 * mm, PAGE_W - MARGIN, fields_y - 1.5 * mm)

    # Valores
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN + 20 * mm, fields_y, (aluno or "")[:45])
    c.drawString(PAGE_W - MARGIN - 45 * mm, fields_y, (turma or "")[:10])
    c.drawString(PAGE_W - MARGIN - 14 * mm, fields_y, (numero or "")[:6])


def draw_instructions(c: canvas.Canvas):
    y = PAGE_H - MARGIN - 50 * mm
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN, y, "CAROS ESTUDANTES, ATENÇÃO PARA AS INSTRUÇÕES:")
    y -= 5 * mm
    c.setFont("Helvetica", 8)

    bullets = [
        f"Este BLOCO contém {QUESTOES} questões.",
        "Cada componente curricular possui o valor de 6,0 pontos.",
        "Questões objetivas: assinale uma alternativa.",
        "Preencha o círculo completamente e com nitidez, com CANETA TINTA AZUL OU PRETA.",
        "Marque apenas uma opção. Qualquer rasura pode ser considerada nula.",
    ]
    for b in bullets:
        c.drawString(MARGIN + 2 * mm, y, f"- {b}")
        y -= 4.5 * mm


def draw_qr(c: canvas.Canvas, payload: dict):
    bio = gerar_qr_bytes(payload)
    img = ImageReader(bio)
    c.drawImage(img, QR_X, QR_Y, width=QR_SIZE, height=QR_SIZE, mask="auto")
    c.setFont("Helvetica", 7)
    c.drawString(QR_X, QR_Y - 8, (payload.get("provaId", "") or "")[:30])
def draw_answer_card(c: canvas.Canvas, questoes: int):
    # =========================
    # MAIS PRA CIMA (SEM CORTAR)
    # =========================
    y_title = 185 * mm  # sobe o bloco (antes 130mm)

    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(PAGE_W / 2, y_title, "LINGUA PORTUGUESA")

    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(PAGE_W / 2, y_title - 6 * mm, "CARTÃO - RESPOSTA")

    # =========================
    # CARTÃO (COMPACTADO)
    # =========================
    row_h = 6 * mm          # ✅ antes 7mm (isso era o que fazia cortar)
    col_w = 9.5 * mm        # um pouco menor para ficar proporcional
    circle_r = 2.0 * mm     # levemente menor

    card_w = 78 * mm        # um pouco menor que 80 pra ficar mais “oficial”
    card_h = (questoes + 2) * row_h

    card_x = (PAGE_W - card_w) / 2
    card_y = (y_title - 12 * mm) - card_h  # mantém proporcional ao título

    # Borda
    c.setLineWidth(1)
    c.rect(card_x, card_y, card_w, card_h, stroke=1, fill=0)
    draw_card_corner_markers(c, card_x, card_y, card_w, card_h)

    # Cabeçalho A B C D
    start_x = card_x + 22 * mm
    header_y = card_y + card_h - 1.5 * row_h

    c.setFont("Helvetica-Bold", 8)
    for j, letra in enumerate(ALTERNATIVAS):
        cx = start_x + j * col_w
        c.drawCentredString(cx, header_y + 2.6 * mm, letra)

    # Linhas
    c.setFont("Helvetica", 8)
    y = header_y - row_h

    for q in range(1, questoes + 1):
        c.drawRightString(card_x + 16 * mm, y + 2.2 * mm, str(q))

        for j in range(len(ALTERNATIVAS)):
            cx = start_x + j * col_w
            cy = y + 2.7 * mm
            c.circle(cx, cy, circle_r, stroke=1, fill=0)

        # Linha separadora leve
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(0.5)
        c.line(card_x + 2 * mm, y, card_x + card_w - 2 * mm, y)

        c.setStrokeColor(colors.black)
        c.setLineWidth(1)

        y -= row_h
def draw_card_corner_markers(c: canvas.Canvas, card_x: float, card_y: float, card_w: float, card_h: float):
    """
    Desenha 4 quadrados pretos nos cantos do cartão-resposta (ao redor do retângulo do cartão).
    card_x, card_y = canto inferior esquerdo do cartão
    """
    tam = CARD_MARKER_SIZE
    off = CARD_MARKER_OFFSET

    # Posições (quadrado fora do retângulo, como na sua imagem)
    tl = (card_x - off - tam, card_y + card_h + off)          # top-left
    tr = (card_x + card_w + off, card_y + card_h + off)       # top-right
    bl = (card_x - off - tam, card_y - off - tam)             # bottom-left
    br = (card_x + card_w + off, card_y - off - tam)          # bottom-right

    c.saveState()
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)

    for (x, y) in (tl, tr, bl, br):
        c.rect(x, y, tam, tam, stroke=0, fill=1)

    c.restoreState()

def gerar_pdf_aluno(out_path: str, aluno: dict, prova_id: str, numero: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    draw_header(
        c,
        nome_escola="COLEGIO ESTADUAL MARIA ABADIA MEIRELES SHINOHARA",
        ano=aluno.get("serie", ""),
        aluno=aluno.get("aluno", ""),
        turma=aluno.get("turma", ""),
        numero=numero,
    )

    draw_instructions(c)

    payload = {
        "aluno": aluno.get("aluno", ""),
        "serie": aluno.get("serie", ""),
        "turma": aluno.get("turma", ""),
        "provaId": prova_id,
    }
    draw_qr(c, payload)
    draw_answer_card(c, questoes=QUESTOES)

    c.showPage()
    c.save()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    alunos = ler_alunos_csv(CSV_PATH)
    if not alunos:
        raise RuntimeError("Nenhum aluno encontrado no CSV.")

    print(f"👥 {len(alunos)} alunos carregados do CSV")
    print(f"📝 Questões: {QUESTOES}")
    print("🖼️ Logo escola:", os.path.abspath(LOGO_ESCOLA_PATH))
    print("🖼️ Logo governo:", os.path.abspath(LOGO_GOVERNO_PATH))

    for idx, aluno in enumerate(alunos, start=PROVA_SEQ_START):
        prova_id = f"{PROVA_DATA}-{idx:03d}"
        numero = str(idx)

        nome_arq = sanitizar_nome_arquivo(aluno["aluno"])
        out_file = os.path.join(OUT_DIR, f"cartao_resposta_{nome_arq}_{prova_id}.pdf")

        gerar_pdf_aluno(out_file, aluno, prova_id, numero)
        print("✅ PDF gerado:", out_file)

    print("\n✅ Finalizado!")
    print("📁 Pasta de saída:", os.path.abspath(OUT_DIR))
    print("📄 CSV usado:", os.path.abspath(CSV_PATH))