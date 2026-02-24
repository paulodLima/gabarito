FROM python:3.11-slim

WORKDIR /app

# Instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

# Cloud Run usa a variável PORT
ENV PORT=8080

# Sobe o servidor ouvindo em 0.0.0.0
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]