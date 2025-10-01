# --- Base Image ---
FROM python:3.10-slim

# --- Set Working Directory ---
WORKDIR /app

# --- Install System Dependencies ---
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    default-libmysqlclient-dev \
    python3-dev \
    clang-format \
    openjdk-21-jre-headless \
    curl \
 && rm -rf /var/lib/apt/lists/*

# --- Upgrade pip ---
RUN pip install --upgrade pip setuptools wheel

# --- Copy Project Files ---
COPY . .

# --- Install Python Dependencies ---
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-web.txt

# --- Set Environment Variables ---
ENV PYTHONUNBUFFERED=1 \
    PORT=10000

# --- Expose Port ---
EXPOSE 10000

# --- Start App ---
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers=4 --timeout=120
