import os
import csv
from pathlib import Path
import json
import argparse
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas


# =========================
# CONFIGURAÇÕES
# =========================
OUT_DIR = "saida_pdfs"
JSON_PATH = "alunos/alunos.json"

# Logos (coloque seus arquivos aqui)
LOGO_ESCOLA_PATH = "logos/logo_escola.png"
LOGO_GOVERNO_PATH = "logos/logo_governo.png"

PROVA_DATA = "2026-02-23"
PROVA_SEQ_START = 1

# ✅ FIXO conforme solicitado
QUESTOES_OBJETIVAS = 14          # 1 a 14 (bolinhas)
DISSERTATIVAS_INICIO = 15        # 15 a 20
DISSERTATIVAS_FIM = 20
LINHAS_POR_DISSERTATIVA = 3      # ✅ 3 linhas por questão dissertativa

ALTERNATIVAS = ["A", "B", "C", "D", "E"]

PAGE_W, PAGE_H = A4
MARGIN = 15 * mm

# =========================
# MARCADORES AO REDOR DO CARTÃO (mantidos)
# =========================
CARD_MARKER_SIZE = 4 * mm      # menor
CARD_MARKER_OFFSET = 2.5 * mm


# =========================
# CSV alunos (aluno, serie, turma)
# =========================
def ler_alunos_csv(csv_path: str):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Arquivo CSV não encontrado: {csv_file.resolve()}\n"
            "Crie um 'alunos.json' com colunas: aluno, serie, turma"
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

def ler_turmas_json(json_path: str):
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(
            f"Arquivo JSON não encontrado: {json_file.resolve()}"
        )

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    turmas = []

    for item in data:
        serie = str(item.get("serie", "")).strip()
        turma = str(item.get("turma", "")).strip()
        quantidade = int(item.get("quantidade", 0))

        if not serie or not turma or quantidade <= 0:
            continue

        turmas.append({
            "serie": serie,
            "turma": turma,
            "quantidade": quantidade
        })

    return turmas

# =========================
# DESENHO DO PDF
# =========================
def draw_header(c: canvas.Canvas, nome_escola: str, ano: str, aluno: str, turma: str, numero: str):
    header_h = 32 * mm
    x0 = MARGIN
    y0 = PAGE_H - (MARGIN - 5 * mm) - header_h
    w = PAGE_W - 2 * MARGIN

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
    c.drawCentredString(PAGE_W / 2, y0 + header_h - 20 * mm, f"LINGUA PORTUGUESA - PROVA")

    # Campos Aluno / Turma / Nº
    fields_y = y0 - 10 * mm
    c.setFont("Helvetica", 9)

    c.drawString(MARGIN, fields_y, "Aluno(a):")
    c.line(MARGIN + 18 * mm, fields_y - 1.5 * mm, PAGE_W - MARGIN - 70 * mm, fields_y - 1.5 * mm)

    c.drawString(PAGE_W - MARGIN - 25 * mm, fields_y, "Sérieº:")
    c.line(PAGE_W - MARGIN - 17 * mm, fields_y - 1.5 * mm, PAGE_W - MARGIN, fields_y - 1.5 * mm)

    c.drawString(PAGE_W - MARGIN - 66 * mm, fields_y, "Turma:")
    c.line(PAGE_W - MARGIN - 50 * mm, fields_y - 1.5 * mm, PAGE_W - MARGIN - 28 * mm, fields_y - 1.5 * mm)



    # Valores
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN + 20 * mm, fields_y, "")
    c.drawString(PAGE_W - MARGIN - 14 * mm, fields_y, (ano or "")[:6])
    c.drawString(PAGE_W - MARGIN - 45 * mm, fields_y, (turma or "")[:10])


def draw_instructions(c: canvas.Canvas):
    y = PAGE_H - MARGIN - 50 * mm
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN, y, "CAROS ESTUDANTES, ATENÇÃO PARA AS INSTRUÇÕES:")
    y -= 5 * mm
    c.setFont("Helvetica", 8)

    bullets = [
        f"Este BLOCO contém {QUESTOES_OBJETIVAS} questões objetivas (1 a {QUESTOES_OBJETIVAS}).",
        f"As questões {DISSERTATIVAS_INICIO} a {DISSERTATIVAS_FIM} são dissertativas (responda nas linhas).",
        "Questões objetivas: assinale uma alternativa.",
        "Preencha o círculo completamente e com nitidez, com CANETA TINTA AZUL OU PRETA.",
        "Marque apenas uma opção. Qualquer rasura pode ser considerada nula.",
        "Cada questão dissertativa possui 3 linhas para resposta.",
    ]
    for b in bullets:
        c.drawString(MARGIN + 2 * mm, y, f"- {b}")
        y -= 4.5 * mm


def draw_card_corner_markers(c: canvas.Canvas, card_x: float, card_y: float, card_w: float, card_h: float):
    """
    Desenha 4 quadrados pretos DENTRO do cartão-resposta (um em cada canto interno).
    card_x, card_y = canto inferior esquerdo do cartão
    """
    tam = CARD_MARKER_SIZE
    pad = CARD_MARKER_OFFSET  # agora é "padding" interno

    # posições internas
    tl = (card_x + pad,               card_y + card_h - pad - tam)   # top-left
    tr = (card_x + card_w - pad - tam, card_y + card_h - pad - tam) # top-right
    bl = (card_x + pad,               card_y + pad)                 # bottom-left
    br = (card_x + card_w - pad - tam, card_y + pad)                # bottom-right

    c.saveState()
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)

    for (x, y) in (tl, tr, bl, br):
        c.rect(x, y, tam, tam, stroke=0, fill=1)

    c.restoreState()


def draw_answer_card_objetivas(c: canvas.Canvas, questoes: int):
    # Mantive sua posição alta pra caber as dissertativas abaixo
    y_title = 210 * mm

    c.setFont("Helvetica-Bold", 9)
    row_h = 6 * mm
    col_w = 9.5 * mm
    circle_r = 2.0 * mm

    card_w = 78 * mm
    card_h = (questoes + 2) * row_h

    card_x = (PAGE_W - card_w) / 2
    card_y = (y_title - 12 * mm) - card_h

    c.setLineWidth(1)
    c.rect(card_x, card_y, card_w, card_h, stroke=1, fill=0)
    draw_card_corner_markers(c, card_x, card_y, card_w, card_h)

    start_x = card_x + 22 * mm
    header_y = card_y + card_h - 1.5 * row_h

    c.setFont("Helvetica-Bold", 8)
    for j, letra in enumerate(ALTERNATIVAS):
        cx = start_x + j * col_w
        c.drawCentredString(cx, header_y + 2.6 * mm, letra)

    c.setFont("Helvetica", 8)
    y = header_y - row_h

    for q in range(1, questoes + 1):
        c.drawRightString(card_x + 16 * mm, y + 2.2 * mm, str(q))

        for j in range(len(ALTERNATIVAS)):
            cx = start_x + j * col_w
            cy = y + 2.7 * mm
            c.circle(cx, cy, circle_r, stroke=1, fill=0)

        y -= row_h

    # Retorna o Y do fim do cartão pra posicionar as dissertativas
    return card_y - 4 * mm


def draw_dissertativas(c: canvas.Canvas, start_q: int, end_q: int, y_start: float, linhas_por: int = 3):
    """
    Desenha o bloco de dissertativas (15 a 20) com 3 linhas para cada.
    """
    y = y_start

    line_w = PAGE_W - 2 * MARGIN
    line_gap = 5.0 * mm   # linhas mais próximas
    block_gap = 2.0 * mm    # espaço entre uma questão e outra

    c.setFont("Helvetica", 9)
    for q in range(start_q, end_q + 1):
        # número da questão
        c.setFont("Helvetica-Bold", 9)
        c.drawString(MARGIN, y, f"{q})")
        c.setFont("Helvetica", 9)

        # linhas (3)
        x1 = MARGIN + 10 * mm
        x2 = MARGIN + line_w
        for _ in range(linhas_por):
            c.line(x1, y - 1.2 * mm, x2, y - 1.2 * mm)
            y -= line_gap

        y -= block_gap

    return y


def gerar_pdf_cartao(out_path: str, serie: str, turma: str, numero: str):
    c = canvas.Canvas(out_path, pagesize=A4)

    draw_header(
        c,
        nome_escola="COLEGIO ESTADUAL MARIA ABADIA MEIRELES SHINOHARA",
        ano=serie,
        aluno="",            # vazio
        turma=turma,
        numero=numero,
    )

    draw_instructions(c)

    # ✅ Objetivas 1..14
    y_after_card = draw_answer_card_objetivas(c, questoes=QUESTOES_OBJETIVAS)

    # ✅ Dissertativas 15..20 com 3 linhas
    draw_dissertativas(
        c,
        start_q=DISSERTATIVAS_INICIO,
        end_q=DISSERTATIVAS_FIM,
        y_start=y_after_card,
        linhas_por=LINHAS_POR_DISSERTATIVA
    )

    c.showPage()
    c.save()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Caminho do arquivo turmas.json")
    parser.add_argument("--out", required=True, help="Pasta de saída dos PDFs")
    args = parser.parse_args()

    JSON_PATH = args.json
    OUT_DIR = args.out

    os.makedirs(OUT_DIR, exist_ok=True)

    turmas = ler_turmas_json(JSON_PATH)
    if not turmas:
        raise RuntimeError("Nenhuma turma encontrada no JSON.")

    contador_global = PROVA_SEQ_START

    for turma in turmas:
        serie = turma["serie"]
        nome_turma = turma["turma"]
        qtd = int(turma["quantidade"])

        for i in range(1, qtd + 1):
            numero = str(i)
            prova_id = f"{PROVA_DATA}-{contador_global:03d}"

            out_file = os.path.join(
                OUT_DIR,
                f"cartao_resposta_{serie.replace(' ', '')}_T{nome_turma}_{numero}_{prova_id}.pdf"
            )

            gerar_pdf_cartao(out_file, serie, nome_turma, numero)
            contador_global += 1

    print(os.path.abspath(OUT_DIR))