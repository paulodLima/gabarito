import os
import csv
from PIL import Image, ImageOps
import io
import numpy as np
import cv2

try:
    import fitz  # PyMuPDF (opcional, pra ler PDF em teste local)
except ImportError:
    fitz = None

alternativas = ["A", "B", "C", "D", "E"]


# =========================
# 1) LER GABARITO DO CSV
# =========================
def ler_gabarito_csv(caminho_csv: str) -> dict[int, str]:
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"CSV não encontrado: {caminho_csv}")

    with open(caminho_csv, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            class _D:
                delimiter = ","
            dialect = _D()

        reader = csv.DictReader(f, dialect=dialect)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        headers_lower = {h.lower(): h for h in headers}

        col_q = headers_lower.get("questao") or headers_lower.get("questão") or headers_lower.get("q")
        col_r = headers_lower.get("resposta") or headers_lower.get("alternativa") or headers_lower.get("resp")

        if not col_q or not col_r:
            raise ValueError(f"CSV precisa ter colunas 'questao' e 'resposta'. Headers: {headers}")

        gabarito: dict[int, str] = {}
        for i, row in enumerate(reader, start=2):
            q_raw = (row.get(col_q) or "").strip()
            r_raw = (row.get(col_r) or "").strip().upper()

            if not q_raw and not r_raw:
                continue

            try:
                q = int(q_raw)
            except ValueError:
                raise ValueError(f"Linha {i}: 'questao' inválida: {q_raw!r}")

            if r_raw not in alternativas:
                raise ValueError(f"Linha {i}: resposta inválida {r_raw!r}. Use apenas A, B, C ou D.")

            gabarito[q] = r_raw

    if not gabarito:
        raise ValueError("CSV de gabarito está vazio.")

    return gabarito


# =========================
# 2) CARREGAR IMAGEM (arquivo) OU PDF (opcional)
# =========================
def carregar_imagem(path: str, dpi: int = 220) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg"):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cv2.imread retornou None: {path}")
        return img

    if ext == ".pdf":
        if fitz is None:
            raise RuntimeError("Para ler PDF, instale PyMuPDF: pip install pymupdf")

        doc = fitz.open(path)
        if doc.page_count < 1:
            raise ValueError("PDF sem páginas.")
        page = doc.load_page(0)

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        doc.close()
        return img

    raise ValueError(f"Extensão não suportada: {ext}")


# ✅ Para o FRONT: carregar imagem direto de bytes (upload)
def carregar_imagem_bytes(file_bytes: bytes) -> np.ndarray:
    try:
        # abre com PIL
        img = Image.open(io.BytesIO(file_bytes))

        # corrige orientação EXIF (CRÍTICO para iPhone)
        img = ImageOps.exif_transpose(img)

        # converte para RGB
        img = img.convert("RGB")

        # PIL -> OpenCV (BGR)
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        return img_bgr

    except Exception as e:
        raise ValueError(f"Erro ao decodificar imagem (EXIF): {e}")


# =========================
# 3) RETIFICAR PELOS 4 QUADRADOS (AGORA DENTRO DO CARTÃO)
# =========================
def _ordenar_pontos_quadrilatero(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect


def _quad_valido(rect: np.ndarray, area_min: float = 15000.0) -> bool:
    if len(np.unique(rect.round(1), axis=0)) < 4:
        return False

    area = abs(cv2.contourArea(rect.astype(np.float32)))
    if area < area_min:
        return False

    tl, tr, br, bl = rect
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    if min(w1, w2, h1, h2) < 200:
        return False
    return True


def encontrar_marcadores_cartao(imagem_bgr: np.ndarray):
    """
    Detecta os 4 quadrados pretos PREENCHIDOS (marcadores) dentro do cartão.
    Estratégia:
    - recorta ROI do meio/baixo
    - threshold invertido (preto vira branco)
    - procura contornos quadrados com boa "preenchimento"
    - pega os 4 melhores por área
    """
    h, w = imagem_bgr.shape[:2]

    # ROI: região provável do cartão (ajustável)
    x1 = int(w * 0.08)
    x2 = int(w * 0.92)
    y1 = int(h * 0.25)
    y2 = int(h * 0.92)

    roi = imagem_bgr[y1:y2, x1:x2].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # limpa ruído
    th = cv2.medianBlur(th, 5)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    roi_h, roi_w = gray.shape[:2]
    img_area = roi_w * roi_h

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.00010:   # menor (marcadores agora menores)
            continue
        if area > img_area * 0.020:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4:
            continue

        x, y, ww, hh = cv2.boundingRect(approx)
        aspect = ww / float(hh) if hh else 0
        if not (0.80 <= aspect <= 1.25):
            continue

        # "preenchimento" (solidity): quadrado preenchido tende a ser alto
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) if hull is not None else 0
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < 0.85:
            continue

        cx = x + ww / 2.0
        cy = y + hh / 2.0

        # salva com área + posição (volta pro sistema da imagem original)
        candidates.append((area, (cx + x1, cy + y1)))

    if len(candidates) < 4:
        return None

    candidates.sort(key=lambda t: -t[0])
    pts = np.array([p for _, p in candidates[:4]], dtype=np.float32)
    rect = _ordenar_pontos_quadrilatero(pts)

    if not _quad_valido(rect):
        return None

    return rect


def retificar_cartao_por_marcadores(imagem_bgr: np.ndarray, escala: float = 1.0):
    rect = encontrar_marcadores_cartao(imagem_bgr)
    if rect is None:
        return None

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB) * escala)

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB) * escala)

    maxW = max(maxW, 650)
    maxH = max(maxH, 650)

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(imagem_bgr, M, (maxW, maxH))


# =========================
# 4) DETECTAR RESPOSTAS (CÍRCULOS)
# =========================
def _kmeans_1d(values: np.ndarray, k: int, iters: int = 30):
    values = values.astype(np.float32).reshape(-1)
    if len(values) < k:
        k = max(1, len(values))

    sorted_v = np.sort(values)
    centers = np.array([sorted_v[int((i + 0.5) * len(sorted_v) / k)] for i in range(k)], dtype=np.float32)

    for _ in range(iters):
        d = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(d, axis=1)

        new_centers = centers.copy()
        for i in range(k):
            pts = values[labels == i]
            if len(pts) > 0:
                new_centers[i] = float(np.mean(pts))

        if np.allclose(new_centers, centers, atol=0.5):
            centers = new_centers
            break
        centers = new_centers

    d = np.abs(values[:, None] - centers[None, :])
    labels = np.argmin(d, axis=1)
    return centers, labels

import cv2
import numpy as np

def detectar_respostas(imagem_bgr: np.ndarray, total_questoes: int):
    alternativas = ["A", "B", "C", "D", "E"]
    num_alts = len(alternativas)

    if imagem_bgr is None:
        return {}

    gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    S = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)  # saturação
    V = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)  # brilho

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=18,
        param1=60, param2=26,
        minRadius=9, maxRadius=26
    )
    if circles is None:
        return {}

    circles = np.round(circles[0, :]).astype(int)

    # mede "escuro" de cada círculo
    bolhas = []
    for (x, y, r) in circles:
        mask = np.zeros(gray.shape, dtype=np.uint8)

        cv2.circle(mask, (x, y), max(r - 4, 1), 255, -1)
        mean_gray = cv2.mean(gray, mask=mask)[0]  # preto ajuda (fica baixo)
        mean_s    = cv2.mean(S, mask=mask)[0]     # azul ajuda (fica alto)
        mean_v    = cv2.mean(V, mask=mask)[0]     # tinta tende a reduzir V

        # score maior = mais marcado (azul ou preto)
        score = (0.95 * mean_s) + (0.55 * (255.0 - mean_v)) + (0.35 * (255.0 - mean_gray))

        bolhas.append((x, y, r, score))

    # --------
    # 1) AGRUPAR POR LINHAS (Y) SEM KMEANS
    # --------
    bolhas.sort(key=lambda t: t[1])  # ordena por Y

    ys = np.array([b[1] for b in bolhas], dtype=np.float32)
    if len(ys) < total_questoes * num_alts:
        # ainda dá pra tentar, mas aviso mental: faltou detecção
        pass

    # estima espaçamento vertical típico
    diffs = np.diff(np.sort(ys))
    diffs = diffs[diffs > 1]  # remove zeros/ruído
    esp_y = np.median(diffs) if len(diffs) else 25.0
    tol_y = max(10.0, 0.55 * esp_y)  # tolerância para ser "mesma linha"

    linhas = []
    linha_atual = []
    y_ref = None

    for b in bolhas:
        x, y, r, mean = b
        if y_ref is None:
            linha_atual = [b]
            y_ref = y
            continue

        if abs(y - y_ref) <= tol_y:
            linha_atual.append(b)
            # atualiza referência suavemente
            y_ref = 0.7 * y_ref + 0.3 * y
        else:
            linhas.append(linha_atual)
            linha_atual = [b]
            y_ref = y

    if linha_atual:
        linhas.append(linha_atual)

    # ordena linhas por y médio e pega as primeiras total_questoes
    linhas.sort(key=lambda row: np.mean([t[1] for t in row]))
    linhas = linhas[:total_questoes]

    respostas = {}

    # --------
    # 2) PARA CADA LINHA: ordenar por X -> A..E
    # --------
    for idx_q, row in enumerate(linhas, start=1):
        # remove possíveis "sobras" se o Hough pegou círculos extra:
        # mantém os 5 mais próximos do centro da linha (por Y)
        y_m = np.mean([t[1] for t in row])
        row.sort(key=lambda t: abs(t[1] - y_m))
        row = row[:num_alts]

        # agora ordena por X (A..E)
        row.sort(key=lambda t: t[0])

        if len(row) < num_alts:
            continue

        scores = [t[3] for t in row]
        order = np.argsort(scores)[::-1]  # maior = mais marcado
        best = order[0]
        second = order[1]

        # diferença mínima para evitar marcar em branco
        if (scores[best] - scores[second]) < 7:
            continue

        respostas[idx_q] = alternativas[best]

    return respostas

def _centros_por_gaps(valores: np.ndarray, k_esperado: int):
    v = np.sort(valores.astype(np.float32))
    if len(v) < k_esperado:
        return None

    diffs = np.diff(v)
    if len(diffs) == 0:
        return None

    med = np.median(diffs)
    if med <= 0:
        return None

    limiar_gap = 2.2 * med
    cortes = np.where(diffs > limiar_gap)[0]

    grupos = []
    ini = 0
    for c in cortes:
        grupos.append(v[ini:c+1])
        ini = c+1
    grupos.append(v[ini:])

    if len(grupos) != k_esperado:
        return None

    centros = np.array([g.mean() for g in grupos], dtype=np.float32)
    return np.sort(centros)
# =========================
# 5) CORRIGIR (IMAGEM)
# =========================
def corrigir_imagem(
        imagem_bgr: np.ndarray,
        gabarito_correto: dict[int, str],
        usar_marcadores_cartao: bool = True,
        aluno: str = "",
        serie: str = "",
        turma: str = "",
        prova_id: str = ""
):
    gabarito_correto = {int(k): str(v).upper() for k, v in gabarito_correto.items()}

    imagem_cartao = imagem_bgr
    marcadores_ok = False

    if usar_marcadores_cartao:
        warped = retificar_cartao_por_marcadores(imagem_bgr)
        if warped is not None:
            imagem_cartao = warped
            marcadores_ok = True

    total_questoes = max(gabarito_correto.keys())
    respostas_lidas = detectar_respostas(imagem_cartao, total_questoes=total_questoes)

    certas = 0
    erradas = 0

    for q, resp_correta in gabarito_correto.items():
        resp_aluno = respostas_lidas.get(q)
        if resp_aluno == resp_correta:
            certas += 1
        else:
            erradas += 1

    return {
        "marcadoresOk": marcadores_ok,
        "certas": certas,
        "erradas": erradas,
        "total": len(gabarito_correto),
        "respostasLidas": respostas_lidas,
    }


# =========================
# 6) TESTE LOCAL (opcional)
# =========================
if __name__ == "__main__":
    PASTA = os.path.abspath("./respostas")  # pasta com as fotos
    CAMINHO_CSV_GABARITO = os.path.abspath("./resposta/gabarito.csv")

    gabarito = ler_gabarito_csv(CAMINHO_CSV_GABARITO)

    extensoes_validas = (".png", ".jpg", ".jpeg", ".pdf")
    arquivos = [f for f in os.listdir(PASTA) if f.lower().endswith(extensoes_validas)]

    if not arquivos:
        print("⚠️ Nenhum arquivo encontrado em:", PASTA)
        raise SystemExit(0)

    print(f"🔍 Encontrados {len(arquivos)} arquivos para correção")
    print(f"✅ Gabarito carregado ({len(gabarito)} questões)\n")

    for nome in sorted(arquivos):
        caminho = os.path.join(PASTA, nome)
        try:
            img = carregar_imagem(caminho, dpi=220)
            resultado = corrigir_imagem(img, gabarito, usar_marcadores_cartao=True)

            print("📄 Arquivo:", nome)
            print("Marcadores OK:", resultado["marcadoresOk"])
            print("Certas:", resultado["certas"], "| Erradas:", resultado["erradas"], "| Total:", resultado["total"])
            print("Respostas lidas:", resultado["respostasLidas"])
            print("-" * 60)

        except Exception as e:
            print("❌ Erro ao processar:", nome)
            print("Motivo:", e)
            print("-" * 60)