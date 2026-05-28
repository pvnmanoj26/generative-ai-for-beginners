FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_cloud.txt .
RUN pip install --no-cache-dir -r requirements_cloud.txt

COPY clinical_app_cloud_v02.py .
COPY adk_agents/ ./adk_agents/

EXPOSE 8080

CMD ["python", "clinical_app_cloud_v02.py"]
