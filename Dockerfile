FROM python:3.10-slim
WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]