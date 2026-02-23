import os
import cv2
import numpy as np
import csv
import json

UPLOAD_FOLDER = os.path.abspath("./uploads")
alternativas = ["A", "B", "C", "D"]


# =========================
# 1) LER GABARITO DO CSV
# =========================
def ler_gabarito_csv(caminho_csv: str) -> dict:
    """
    Lê um CSV no formato:
      questao,resposta
      1,B
      2,C

    Aceita separador , ou ; e valida respostas A-D.
    Retorna: {1:"B", 2:"C", ...}
    """
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
            raise ValueError(
                f"CSV precisa ter colunas tipo 'questao' e 'resposta'. Headers encontrados: {headers}"
            )

        gabarito = {}
        for i, row in enumerate(reader, start=2):
            if not row:
                continue

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
        raise ValueError("CSV de gabarito está vazio ou sem linhas válidas.")

    return gabarito


# =========================
# 2) LER QR CODE (JSON) DA IMAGEM
# =========================
def ler_payload_qr_opencv(imagem_bgr: np.ndarray) -> dict | None:
    """
    Tenta ler um QRCode na imagem com OpenCV.
    Espera JSON (ex.: {"aluno":"...", "serie":"...", "turma":"...", "provaId":"..."}).
    Retorna dict ou None.
    """
    detector = cv2.QRCodeDetector()

    # tentativa 1: direto
    data, points, _ = detector.detectAndDecode(imagem_bgr)
    if data:
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"raw": data}

    # tentativa 2: binarização (ajuda quando a foto está ruim)
    gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    data, points, _ = detector.detectAndDecode(thr)
    if data:
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"raw": data}

    return None


def recortar_area_qr(imagem_bgr: np.ndarray, reserva_direita_px: int = 220, altura_topo_px: int = 220) -> np.ndarray:
    """
    Recorta o canto superior direito (onde seu gerador coloca o QR).
    Isso costuma aumentar MUITO a taxa de leitura do QR.
    """
    h, w = imagem_bgr.shape[:2]
    x0 = max(w - reserva_direita_px, 0)
    y0 = 0
    x1 = w
    y1 = min(altura_topo_px, h)
    return imagem_bgr[y0:y1, x0:x1]


def ler_payload_qr(imagem_bgr: np.ndarray) -> dict | None:
    """
    Tenta ler QR primeiro na área do QR (canto superior direito) e depois na imagem toda.
    """
    roi = recortar_area_qr(imagem_bgr, reserva_direita_px=260, altura_topo_px=260)
    payload = ler_payload_qr_opencv(roi)
    if payload:
        return payload
    return ler_payload_qr_opencv(imagem_bgr)


# =========================
# 3) DETECTAR RESPOSTAS (CÍRCULOS)
# =========================
def detectar_respostas(imagem, total_questoes):
    if imagem is None:
        return {}

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=40,
        param1=50, param2=30,
        minRadius=15, maxRadius=30
    )

    if circles is None:
        return {}

    circles = np.round(circles[0, :]).astype("int")
    circles = sorted(circles, key=lambda c: c[1])  # ordena por Y

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

        linha = sorted(linha, key=lambda c: c[0])  # ordena por X
        preenchidos = []

        for j, (x, y, r) in enumerate(linha[:len(alternativas)]):  # garante no máximo A-D
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.circle(mask, (x, y), max(r - 5, 1), 255, -1)
            media = cv2.mean(gray, mask=mask)[0]
            preenchidos.append((j, media))

        preenchidos.sort(key=lambda t: t[1])  # menor média = mais escuro = marcado
        index_resposta = preenchidos[0][0]
        respostas[i + 1] = alternativas[index_resposta]

    return respostas


# =========================
# 4) CORRIGIR UMA IMAGEM (LÊ ALUNO DO QR)
# =========================
def corrigir_local(caminho_imagem, gabarito_correto, aluno_fallback=""):
    gabarito_correto = {int(k): str(v).upper() for k, v in gabarito_correto.items()}

    if not os.path.exists(caminho_imagem):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_imagem}")

    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise ValueError(
            f"cv2.imread retornou None. Caminho: {caminho_imagem}\n"
            "Causas comuns: arquivo não é imagem, extensão HEIC, caminho errado, permissões."
        )

    # ✅ QR
    payload = ler_payload_qr(imagem)  # tenta na ROI e depois na imagem toda
    aluno = aluno_fallback
    serie = ""
    turma = ""
    prova_id = ""

    if payload:
        aluno = (payload.get("aluno") or aluno_fallback or "").strip()
        serie = (payload.get("serie") or "").strip()
        turma = (payload.get("turma") or "").strip()
        prova_id = (payload.get("provaId") or "").strip()

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
            detalhes[q] = {"resposta": resp_aluno or "sem resposta", "status": "errada"}

    return {
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
    }


# =========================
# 5) EXECUÇÃO EM LOTE
# =========================
if __name__ == "__main__":
    # Pasta onde estão os gabaritos gerados
    PASTA_GABARITOS = os.path.abspath("./saida_gabaritos")

    # CSV com respostas corretas
    CAMINHO_CSV_GABARITO = os.path.abspath("./resposta/gabarito.csv")

    if not os.path.exists(PASTA_GABARITOS):
        raise FileNotFoundError(f"Pasta não encontrada: {PASTA_GABARITOS}")

    # ✅ Gabarito do CSV
    gabarito = ler_gabarito_csv(CAMINHO_CSV_GABARITO)

    extensoes_validas = (".png", ".jpg", ".jpeg")
    arquivos = [
        f for f in os.listdir(PASTA_GABARITOS)
        if f.lower().endswith(extensoes_validas)
    ]

    if not arquivos:
        print("⚠️ Nenhuma imagem encontrada em:", PASTA_GABARITOS)
        exit(0)

    print(f"🔍 Encontrados {len(arquivos)} gabaritos para correção\n")
    print(f"✅ Gabarito carregado do CSV ({len(gabarito)} questões)\n")

    for nome_arquivo in arquivos:
        caminho = os.path.join(PASTA_GABARITOS, nome_arquivo)

        try:
            resultado = corrigir_local(
                caminho_imagem=caminho,
                gabarito_correto=gabarito,
                aluno_fallback=""  # não usa mais nome_arquivo como aluno
            )

            print("📄 Arquivo:", nome_arquivo)
            print("Aluno:", resultado["aluno"] or "(não lido do QR)")
            print("Série:", resultado["serie"])
            print("Turma:", resultado["turma"])
            print("ProvaId:", resultado["provaId"])
            print("Certas:", resultado["certas"])
            print("Erradas:", resultado["erradas"])
            print("Total:", resultado["total"])
            print("Respostas lidas:", resultado["respostas_lidas"])
            if not resultado["qr_payload"]:
                print("⚠️ QR não foi lido (tente aumentar tamanho do QR ou melhorar a foto).")
            print("-" * 60)

        except Exception as e:
            print("❌ Erro ao processar:", nome_arquivo)
            print("Motivo:", e)
            print("-" * 60)