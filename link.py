from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Entradas
url = input("Digite a URL do produto Mercado Livre: ").strip()
desconto_percentual = input("Digite o valor do desconto (ex: 15 para 15%): ").strip()
cupom = input("Digite o código do cupom (ou deixe vazio se não quiser): ").strip()

# Conversão do desconto
try:
    desconto_percentual = float(desconto_percentual)
except ValueError:
    print("Valor de desconto inválido, usando 0%")
    desconto_percentual = 0.0
desconto = desconto_percentual / 100

# Configuração do navegador
options = Options()
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/114.0.0.0 Safari/537.36")
# options.add_argument("--headless")  # Se quiser rodar em segundo plano

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get(url)
wait = WebDriverWait(driver, 10)

try:
    # Nome do produto
    nome = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1.ui-pdp-title"))).text

    # Descrição do produto (opcional — pode adaptar)
    descricao = nome  # ou busque de outro lugar se quiser um resumo diferente

    # Imagem principal
    imagem_url = wait.until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "figure.ui-pdp-gallery__figure img.ui-pdp-image")
        )
    ).get_attribute("data-zoom")

    # Tenta pegar o valor com desconto (melhor opção)
    try:
        preco_desconto_meta = driver.find_element(
            By.CSS_SELECTOR, 'span[itemprop="offers"] meta[itemprop="price"]'
        )
        preco_com_desconto = float(preco_desconto_meta.get_attribute("content"))
    except:
        # Se não houver valor com desconto, pega o preço normal
        parte_inteira = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.andes-money-amount__fraction"))).text
        parte_centavos = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.andes-money-amount__cents"))).text
        preco_com_desconto = float(f"{parte_inteira}.{parte_centavos}")

    # Tenta pegar o preço original (se houver)
    try:
        original_inteira = driver.find_element(
            By.CSS_SELECTOR, "s.andes-money-amount--previous span.andes-money-amount__fraction"
        ).text
        original_centavos = driver.find_element(
            By.CSS_SELECTOR, "s.andes-money-amount--previous span.andes-money-amount__cents"
        ).text
        preco_original = float(f"{original_inteira}.{original_centavos}")
    except:
        preco_original = None

    # Aplica desconto adicional se informado
    if desconto > 0:
        preco_final = preco_com_desconto * (1 - desconto)
    else:
        preco_final = preco_com_desconto

    # Formatação BR
    def formatar(valor):
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    preco_formatado = formatar(preco_final)
    preco_desconto_ml = formatar(preco_com_desconto)
    preco_original_formatado = formatar(preco_original) if preco_original else None

    # Texto cupom
    texto_cupom = f"🏷 Cupom: {cupom}" if cupom else ""

    # Monta a mensagem
    mensagem = f"""{nome.upper()}

{descricao}

{"De " + preco_original_formatado if preco_original_formatado else ""} 
Por {preco_formatado}🔥parcelado

{texto_cupom}

👉 Pegar promoção:
{url}

🛒 Imagem: {imagem_url}
"""

    print("\n" + mensagem.strip())

except Exception as e:
    print("Erro ao buscar dados:", e)

driver.quit()
