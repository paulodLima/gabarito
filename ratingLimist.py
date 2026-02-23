#!/usr/bin/env python3
"""
test_rate_limit.py
-----------------------------------
Script para testar rate limiting em um endpoint JHipster Gateway.

⚠️ Use apenas em ambiente que você controla (ex: localhost ou staging).
"""

import requests
import time
from datetime import datetime, timezone
import urllib3
import json

# ---------------- CONFIGURAÇÃO ---------------- #
# URL do endpoint passando PELO GATEWAY:
URL = "https://localhost:8443/api/authenticate"

# Payload do login (simulando autenticação)
PAYLOAD = {
    "username": "invalid",
    "password": "wrong"
}

# Se o certificado do gateway for self-signed:
VERIFY_SSL = False  # True em produção
if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Quantidade máxima de requisições para teste
MAX_REQUESTS = 200
# Atraso entre requisições (em segundos)
DELAY = 0.1
# ------------------------------------------------ #


def now_iso():
    """Retorna timestamp UTC ISO format."""
    return datetime.now(timezone.utc).isoformat()


def main():
    print(f"=== Iniciando teste de rate limit ===")
    print(f"URL: {URL}")
    print(f"Total requisições planejadas: {MAX_REQUESTS}")
    print(f"Delay entre requisições: {DELAY}s")
    print("-------------------------------------\n")

    session = requests.Session()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Origin": "https://localhost:8080"
    }

    for i in range(1, MAX_REQUESTS + 1):
        t0 = datetime.now(timezone.utc)
        try:
            r = session.post(URL, headers=headers, json=PAYLOAD, verify=VERIFY_SSL, timeout=10)
        except requests.exceptions.SSLError as e:
            print(f"{i:04d} {now_iso()} SSL ERROR: {e}")
            break
        except Exception as e:
            print(f"{i:04d} {now_iso()} ERRO: {e}")
            continue

        t1 = datetime.now(timezone.utc)
        elapsed = (t1 - t0).total_seconds()
        print(f"{i:04d} {now_iso()} | {r.status_code} | {elapsed:.3f}s")

        # Mostra cabeçalhos relacionados a rate-limit (se existirem)
        for hk, hv in r.headers.items():
            if hk.lower().startswith("x-ratelimit") or hk.lower() == "retry-after":
                print(f"    -> {hk}: {hv}")

        # Quando o gateway aplicar o limite, ele deve retornar 429
        if r.status_code == 429:
            print("\n🚫 Rate limit detectado!")
            if "Retry-After" in r.headers:
                print("   Retry-After:", r.headers["Retry-After"])
            print("   Corpo da resposta (limite):")
            print(r.text[:500])
            break

        time.sleep(DELAY)

    print("\n=== Teste finalizado ===")


if __name__ == "__main__":
    main()
