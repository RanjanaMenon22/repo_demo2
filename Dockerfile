# Simple Dockerfile for the Iris Predictor Flask app
FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for some binary wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . /app

# Expose default port
EXPOSE 5000

# Environment defaults
ENV APP_HOST=0.0.0.0
ENV PORT=5000
ENV AUTO_OPEN=0

CMD ["python", "app.py"]
