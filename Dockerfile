FROM --platform=linux/amd64 python:3.11-slim

# 환경변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사
COPY requirements.txt .

# 1. 요구사항 전체 설치
# 2. 설치 직후 torchvision이 혹시라도 깔렸다면 강제로 뜯어냄
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y torchvision

# Non-root 사용자 생성 및 소스 복사
RUN useradd -m appuser
COPY --chown=appuser:appuser . .

# 사용자 전환
USER appuser

# 컨테이너가 사용할 포트 명시
EXPOSE 8000

# FastAPI 서버 실행 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]