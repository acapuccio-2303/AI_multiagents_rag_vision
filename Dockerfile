FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Working directory
WORKDIR /app

# System dependencies
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY code/ .

# Expose ports (from .env or default)
ENV BACKEND_PORT=8000 
ENV FRONTEND_PORT=8501 
EXPOSE ${BACKEND_PORT} ${FRONTEND_PORT}
