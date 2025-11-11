FROM python:3.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libfftw3-dev \
    libyaml-dev \
    libtag1-dev \
    libsamplerate0 \
    libsamplerate0-dev \
    libsndfile1 \
    libsndfile1-dev \
    libasound2 \
    libasound2-dev \
    cmake \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . ./

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
