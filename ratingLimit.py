#!/usr/bin/env python3
"""
auth_rate_test.py
Teste de rate limit para POST /api/authenticate.
Atenção: definir VERIFY_SSL=False só para ambiente de desenvolvimento.
"""

import requests
import time
from datetime import datetime, timezone
import urllib3
import json
import sys
from typing import List

# -------- CONFIGURAÇÃO ----------
URL = "https://localhost:8443/api/authenticate"
VERIFY_SSL = False        # False = ignora verificação de certificado (dev only)
max_requests = 1000
delay = 0.05              # segundos entre requisições
use_xff = False           # True varia X-Forwarded-For
vary_credentials = False  # True usa credential_list em round-robin
credential_list = [
    {"username": "user1", "password": "pass1"},
    {"username": "user2", "password": "pass2"},
    # adicionar credenciais de teste aqui (não use credenciais reais sem autorização)
]
# Se vary_credentials=False, o payload abaixo será usado repetidamente:
static_payload = {"username": "invalid", "password": "wrong"}
TIMEOUT = 10
LOG_FILE = None           # "auth_rate_log.txt" para gravar
# ---------------------------------

if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def make_xff(i: int) -> str:
    last = (i % 250) + 1
    return f"192.0.2.{last}"

def log(msg: str):
    print(msg, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def run():
    session = requests.Session()
    # montar headers comuns
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Origin": "https://localhost:8443"
    }

    for i in range(1, max_requests + 1):
        h = headers.copy()
        if use_xff:
            h["X-Forwarded-For"] = make_xff(i)

        if vary_credentials and credential_list:
            payload = credential_list[(i-1) % len(credential_list)]
        else:
            payload = static_payload

        t0 = datetime.now(timezone.utc)
        try:
            r = session.post(URL, headers=h, json=payload, timeout=TIMEOUT, verify=VERIFY_SSL)
        except requests.exceptions.SSLError as e:
            log(f"{i:04d} {now_iso()} SSL-ERROR -> {e}")
            log("Sugestão: para dev/local, ajuste VERIFY_SSL=True com certificado válido ou VERIFY_SSL=False (inseguro).")
            break
        except requests.RequestException as e:
            t1 = datetime.now(timezone.utc)
            elapsed = (t1 - t0).total_seconds()
            log(f"{i:04d} {now_iso()} REQUEST-ERROR {elapsed:.3f}s -> {e}")
            time.sleep(delay)
            continue

        t1 = datetime.now(timezone.utc)
        elapsed = (t1 - t0).total_seconds()
        # log básico
        log(f"{i:04d} {now_iso()} {r.status_code} {elapsed:.3f}s payload={json.dumps({'username': payload.get('username')})}")

        # imprimir cabeçalhos de controle de rate-limit se existirem
        for hk, hv in r.headers.items():
            kl = hk.lower()
            if kl.startswith("x-ratelimit") or kl == "retry-after":
                log(f"   HEADER {hk}: {hv}")

        # detectar rate-limit
        if r.status_code == 429:
            log(">>> Rate limit detectado (429)!")
            if 'Retry-After' in r.headers:
                log(f"Retry-After: {r.headers['Retry-After']}")
            log("Response body (primeiros 1000 chars):")
            text = r.text or ""
            log(text[:1000])
            break

        # se o endpoint retorna 401 para credenciais inválidas,
        # isso é esperado — mas se começar a retornar 429, é rate-limit.
        if r.status_code == 401:
            # opcional: detectar mudança recorrente para 401->429 etc
            pass

        time.sleep(delay)

if __name__ == "__main__":
    run()
