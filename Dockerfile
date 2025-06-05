FROM python:3.9-slim

WORKDIR /app

# Ajoute les outils de build et les dépendances système nécessaires
RUN apt-get update && \
    apt-get install -y build-essential gcc g++ python3-dev libffi-dev libssl-dev && \
    apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]