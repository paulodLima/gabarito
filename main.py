from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# CORS: enquanto testa, deixe aberto. Depois eu te mostro como travar no seu domínio.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/processar")
def processar(payload: dict):
    # aqui você chama sua lógica python real
    return {"recebido": payload}