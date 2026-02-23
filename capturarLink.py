from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument("user-data-dir=C:/Users/paulos/AppData/Local/Google/Chrome/User Data")
options.add_argument("--profile-directory=Default")
driver = webdriver.Chrome(options=options)
# options.add_argument("--headless")  # Descomente para rodar sem abrir a janela do navegador

try:
    # Abre a página do produto
    driver.get("https://produto.mercadolivre.com.br/MLB-3827168491-tnis-fila-racer-speedzone-masculino-_JM")

    # Espera o botão "Compartilhar" estar clicável e clica nele
    compartilhar_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Compartilhar')]"))
    )
    compartilhar_button.click()

    # Espera o textarea que contém o link aparecer e pega o texto
    textarea_link = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "textarea[data-testid='text-field__label_link']"))
    )
    link_produto = textarea_link.get_attribute('value')

    print("Link do produto capturado:", link_produto)

finally:
    driver.quit()
