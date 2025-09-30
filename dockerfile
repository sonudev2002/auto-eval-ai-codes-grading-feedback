New-Item -Path "Dockerfile" -ItemType File -Force | Out-Null
Set-Content -Path "Dockerfile" -Value @"
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements-web.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements-web.txt
COPY . /app
EXPOSE 10000
CMD ["gunicorn", "app:app", "--workers", "4", "--worker-class", "gthread", "--bind", "0.0.0.0:$PORT"]
"@
