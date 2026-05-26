FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_cloud.txt .
RUN pip install --no-cache-dir -r requirements_cloud.txt

COPY clinical_api.py .

EXPOSE 8080

CMD ["uvicorn", "clinical_api:app", "--host", "0.0.0.0", "--port", "8080"]
