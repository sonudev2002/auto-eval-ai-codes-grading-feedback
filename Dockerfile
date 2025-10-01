# --- Base Image ---
FROM python:3.10-slim

# --- Set Working Directory ---
WORKDIR /app

# --- Install System Dependencies ---
RUN apt-get update && apt-get install -y \
    clang-format \
    openjdk-21-jre-headless \
    curl \
 && rm -rf /var/lib/apt/lists/*

# --- Copy Project Files ---
COPY . .

# --- Install Python Dependencies ---
RUN pip install --no-cache-dir -r requirements-web.txt

# --- Set Environment Variables ---
ENV PYTHONUNBUFFERED=1 \
    PORT=10000

# --- Expose Port ---
EXPOSE 10000

# --- Start App ---
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers=4 --timeout=120
