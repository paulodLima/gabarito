import os
import cv2
import numpy as np
import csv
import json
import itertools

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

alternativas = ["A", "B", "C", "D"]


# =========================
# 1) LER GABARITO DO CSV
# =========================
def ler_gabarito_csv(caminho_csv: str) -> dict:
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

        gabarito = {}
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
# 2) CARREGAR IMAGEM (PNG/JPG) OU PDF
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


# =========================
# 3) LER QR CODE (JSON)
# =========================
def ler_payload_qr_opencv(imagem_bgr: np.ndarray) -> dict | None:
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(imagem_bgr)
    if data:
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"raw": data}

    gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data, _, _ = detector.detectAndDecode(thr)
    if data:
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"raw": data}
    return None


def recortar_area_qr(imagem_bgr: np.ndarray, reserva_direita_px: int = 360, y0_px: int = 140, altura_px: int = 520) -> np.ndarray:
    h, w = imagem_bgr.shape[:2]
    x0 = max(w - reserva_direita_px, 0)
    y0 = min(max(y0_px, 0), h)
    y1 = min(y0 + altura_px, h)
    return imagem_bgr[y0:y1, x0:w]


def ler_payload_qr(imagem_bgr: np.ndarray) -> dict | None:
    roi = recortar_area_qr(imagem_bgr)
    payload = ler_payload_qr_opencv(roi)
    if payload:
        return payload
    return ler_payload_qr_opencv(imagem_bgr)


# =========================
# 4) RETIFICAR PELOS QUADRADOS DO CARTÃO (CORRIGIDO)
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


def _quad_valido(rect: np.ndarray, area_min: float = 20000.0) -> bool:
    # sem pontos repetidos
    if len(np.unique(rect.round(1), axis=0)) < 4:
        return False
    # área mínima
    area = abs(cv2.contourArea(rect.astype(np.float32)))
    if area < area_min:
        return False
    # lados mínimos (evita colapsar)
    tl, tr, br, bl = rect
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    if min(w1, w2, h1, h2) < 80:
        return False
    return True


def encontrar_marcadores_cartao(imagem_bgr: np.ndarray):
    """
    CORRIGIDO:
    - detecta quadrados pretos na ROI do cartão
    - pega os 4 MAIORES (os marcadores reais)
    - ordena e valida
    """
    h, w = imagem_bgr.shape[:2]

    # ROI: região provável do cartão (meio/parte de baixo)
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)
    y1 = int(h * 0.35)
    y2 = int(h * 0.92)
    roi = imagem_bgr[y1:y2, x1:x2].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 5)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    roi_h, roi_w = gray.shape[:2]
    img_area = roi_w * roi_h

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.00015:
            continue
        if area > img_area * 0.03:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4:
            continue

        x, y, ww, hh = cv2.boundingRect(approx)
        aspect = ww / float(hh) if hh else 0
        if not (0.80 <= aspect <= 1.25):
            continue

        cx = x + ww / 2.0
        cy = y + hh / 2.0
        squares.append((area, (cx + x1, cy + y1)))

    if len(squares) < 4:
        return None

    # ✅ pegue os 4 maiores => marcadores reais
    squares.sort(key=lambda t: -t[0])
    pts = np.array([p for _, p in squares[:4]], dtype=np.float32)
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

    maxW = max(maxW, 600)
    maxH = max(maxH, 600)

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(imagem_bgr, M, (maxW, maxH))


# =========================
# 5) DETECTAR RESPOSTAS (CÍRCULOS) - igual ao seu
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


def detectar_respostas(imagem_bgr: np.ndarray, total_questoes: int):
    if imagem_bgr is None:
        return {}

    gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=28,
        param1=60, param2=26,
        minRadius=9, maxRadius=26
    )
    if circles is None:
        return {}

    circles = np.round(circles[0, :]).astype(int)

    data = []
    for (x, y, r) in circles:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), max(r - 4, 1), 255, -1)
        mean = cv2.mean(gray, mask=mask)[0]
        data.append((x, y, r, mean))

    xs = np.array([d[0] for d in data], dtype=np.float32)
    ys = np.array([d[1] for d in data], dtype=np.float32)

    cx, lx = _kmeans_1d(xs, k=len(alternativas))
    order_x = np.argsort(cx)
    map_x = {int(order_x[i]): i for i in range(len(order_x))}

    cy, ly = _kmeans_1d(ys, k=total_questoes)
    order_y = np.argsort(cy)
    map_y = {int(order_y[i]): i for i in range(len(order_y))}

    grid = {}
    for (x, y, r, mean), clx, cly in zip(data, lx, ly):
        alt_idx = map_x.get(int(clx))
        q_idx = map_y.get(int(cly))
        if alt_idx is None or q_idx is None:
            continue
        key = (q_idx, alt_idx)
        if key not in grid or mean < grid[key]:
            grid[key] = mean

    respostas = {}
    for q_idx in range(min(total_questoes, len(order_y))):
        candidatos = []
        for alt_idx in range(len(alternativas)):
            mean = grid.get((q_idx, alt_idx))
            if mean is not None:
                candidatos.append((alt_idx, mean))

        if not candidatos:
            continue

        candidatos.sort(key=lambda t: t[1])
        respostas[q_idx + 1] = alternativas[candidatos[0][0]]

    return respostas


# =========================
# 6) CORRIGIR (PDF/IMG)
# =========================
def corrigir_arquivo(caminho: str, gabarito_correto: dict, usar_marcadores_cartao: bool = True, aluno_fallback: str = ""):
    gabarito_correto = {int(k): str(v).upper() for k, v in gabarito_correto.items()}
    imagem = carregar_imagem(caminho, dpi=220)

    # QR na imagem ORIGINAL
    payload = ler_payload_qr(imagem)
    aluno = aluno_fallback
    serie = ""
    turma = ""
    prova_id = ""

    if payload:
        aluno = (payload.get("aluno") or aluno_fallback or "").strip()
        serie = (payload.get("serie") or "").strip()
        turma = (payload.get("turma") or "").strip()
        prova_id = (payload.get("provaId") or "").strip()

    imagem_cartao = imagem
    marcadores_ok = False

    if usar_marcadores_cartao:
        warped = retificar_cartao_por_marcadores(imagem)
        if warped is not None:
            imagem_cartao = warped
            marcadores_ok = True

    respostas_lidas = detectar_respostas(imagem_cartao, total_questoes=len(gabarito_correto))

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
            detalhes[q] = {"resposta": resp_aluno or "sem resposta", "status": "errada"}

    return {
        "arquivo": os.path.basename(caminho),
        "aluno": aluno,
        "serie": serie,
        "turma": turma,
        "provaId": prova_id,
        "certas": certas,
        "erradas": erradas,
        "total": len(gabarito_correto),
        "detalhes": detalhes,
        "respostas_lidas": respostas_lidas,
        "qr_payload": payload,
        "marcadores_cartao_ok": marcadores_ok,
    }


# =========================
# 7) EXECUÇÃO EM LOTE
# =========================
if __name__ == "__main__":
    PASTA = os.path.abspath("./saida_pdfs")
    CAMINHO_CSV_GABARITO = os.path.abspath("./resposta/gabarito.csv")

    gabarito = ler_gabarito_csv(CAMINHO_CSV_GABARITO)

    extensoes_validas = (".png", ".jpg", ".jpeg", ".pdf")
    arquivos = [f for f in os.listdir(PASTA) if f.lower().endswith(extensoes_validas)]

    if not arquivos:
        print("⚠️ Nenhum arquivo encontrado em:", PASTA)
        raise SystemExit(0)

    print(f"🔍 Encontrados {len(arquivos)} arquivos para correção")
    print(f"✅ Gabarito carregado ({len(gabarito)} questões)\n")

    for nome in arquivos:
        caminho = os.path.join(PASTA, nome)
        try:
            resultado = corrigir_arquivo(
                caminho=caminho,
                gabarito_correto=gabarito,
                usar_marcadores_cartao=True,
                aluno_fallback=""
            )

            print("📄 Arquivo:", resultado["arquivo"])
            print("Aluno:", resultado["aluno"] or "(não lido do QR)")
            print("Série:", resultado["serie"], "| Turma:", resultado["turma"])
            print("ProvaId:", resultado["provaId"])
            print("Marcadores cartão OK:", resultado["marcadores_cartao_ok"])
            print("Certas:", resultado["certas"], "| Erradas:", resultado["erradas"], "| Total:", resultado["total"])
            print("Respostas lidas:", resultado["respostas_lidas"])
            if not resultado["qr_payload"]:
                print("⚠️ QR não foi lido")
            print("-" * 60)

        except Exception as e:
            print("❌ Erro ao processar:", nome)
            print("Motivo:", e)
            print("-" * 60)