FROM python:3.12-slim

# ffmpeg for audio format conversion (PCM / A-law / mu-law)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render provides $PORT; default to 5001 locally
ENV PORT=5001
EXPOSE 5001

# 2 workers, long timeout for TTS calls
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 2 --timeout 300
