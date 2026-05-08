# Deployment Notes (AWS / Docker)

AWS GPU 인스턴스에 Docker로 배포할 때 필요한 환경/주의사항 정리.

---

## 1. 하드웨어 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU VRAM | 10GB | 12GB+ |
| RAM | 16GB | 24GB |
| 디스크 | 50GB | 80GB (모델 캐시 포함) |
| CUDA | 12.4 | 12.4 |

**AWS 인스턴스 추천:**
- `g5.xlarge` (A10G 24GB, 4 vCPU, 16GB RAM) — 적정
- `g4dn.xlarge` (T4 16GB, 4 vCPU, 16GB RAM) — 가능
- `g6.xlarge` (L4 24GB) — 신형

---

## 2. 시스템 패키지 (Docker `apt-get install`)

PyPI 패키지가 의존하는 OS 라이브러리들:

```bash
apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*
```

| 패키지 | 필요한 이유 |
|--------|------------|
| `libsndfile1` | `soundfile` (Python WAV/FLAC 디코더) 의존성 |
| `ffmpeg` | `librosa`, `faster-whisper` 오디오 디코딩 |
| `libgl1-mesa-glx`, `libglib2.0-0` | `opencv-python-headless` / `deepface` 런타임 |
| `git` | HuggingFace 모델 일부가 git-lfs 필요 |

---

## 3. Docker 베이스 이미지

CUDA 12.4 + Python 3.10+ 조합 사용.

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    libsndfile1 ffmpeg libgl1-mesa-glx libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace 캐시 위치 (재시작 시 재다운로드 방지 — 볼륨 마운트 권장)
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**실행 시 `--gpus all` 필수:**
```bash
docker run --gpus all -p 8000:8000 --env-file .env counseling-server
```

---

## 4. `requirements.txt` 변경 사항

현재 파일 (이미 반영됨):
```
--extra-index-url https://download.pytorch.org/whl/cu124
torch>=2.6.0
```

**주의 사항:**
- `bitsandbytes`는 **Linux + CUDA** 환경에서만 8-bit 양자화 정상 동작. Windows/CPU 환경에서는 동작이 제한적.
- `deepface` 첫 실행 시 가중치(~수백MB)를 자동 다운로드하여 `~/.deepface/` 에 캐시. 컨테이너 재시작 시 재다운로드 방지하려면 볼륨 마운트 필요.
- `faster-whisper`는 cuDNN 8/9 라이브러리를 동적 로드. CUDA 12.4 base 이미지에 포함되어 있음.

추가 설치 필요한 것 없음 — `pip install -r requirements.txt` 한 번이면 됩니다.

---

## 5. 환경 변수 (`.env`)

`.env.example` 참고. 핵심:

| 키 | 값 | 비고 |
|----|----|------|
| `OPENAI_API_KEY` | `sk-proj-...` | **필수**. GPT-4o-mini 호출용. 코드 푸시 시 절대 포함 X |
| `CBT_LLM_MODEL` | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` | HuggingFace에서 자동 다운로드 |
| `CBT_LLM_DEVICE` | `cuda` | |
| `WHISPER_DEVICE` | `cuda` | |
| `TEXT_EMOTION_DEVICE` | `cuda` | |
| `AUDIO_EMOTION_DEVICE` | `cpu` | Wav2Vec2는 CPU 권장 (병목 아님) |

**AWS Secrets Manager / SSM Parameter Store** 등으로 `OPENAI_API_KEY`를 주입하는 게 안전합니다.

---

## 6. 모델 파일 (Git 미포함)

`.gitignore` 에 의해 `models/` 폴더는 Git에 포함되지 않습니다. 별도로 컨테이너에 마운트하거나 빌드 시 복사해야 합니다.

```
models/
├── text-emotion-final/     # klue/bert 텍스트 감정 분류 (~440MB)
└── voice-emotion-final/    # Wav2Vec2 음성 감정 분류 (~360MB)
```

**전달 방식 (택1):**
1. **S3 → 컨테이너 시작 시 다운로드** (권장)
   ```bash
   aws s3 sync s3://my-bucket/models/ /app/models/
   ```
2. 빌드 시 COPY (이미지 비대화 — 비권장)
3. EFS/EBS 볼륨 마운트

**EXAONE 본체**는 HuggingFace에서 자동 다운로드 (~16GB → `HF_HOME`). 첫 실행 시 5~10분 소요. 볼륨 마운트로 재사용 강력 권장.

---

## 7. 포트 / 네트워크

- 서버: `8000/tcp`
- WebSocket이라 ALB 사용 시 **WebSocket 지원 활성화** 필요 (`stickiness`, `idle_timeout` ≥ 600s)
- 보안그룹: 클라이언트(프론트) 도메인 허용

---

## 8. 헬스체크

```bash
GET /
→ {"status": "ok", "message": "상담 서버 정상 작동 중"}
```

ALB/Target Group 헬스체크 경로로 사용 가능. 단, 초기 모델 로딩(2~3분)이 끝나야 200 응답이 정상 반환되니 health check grace period를 충분히 (≥300초) 설정.

---

## 9. 첫 부팅 체크리스트

```
[ ] GPU 인식: nvidia-smi 컨테이너 내부에서 동작 확인
[ ] CUDA 12.4 호환: torch.cuda.is_available() == True
[ ] models/ 폴더 마운트 완료
[ ] .env 의 OPENAI_API_KEY 주입 확인
[ ] 8000 포트 외부 노출
[ ] 첫 EXAONE 다운로드(~16GB) 완료 후 VRAM 7~8GB 점유 확인
[ ] 헬스체크 GET / → 200 OK
[ ] WebSocket 연결 테스트: wscat -c ws://<host>:8000/ws/counseling/test-001
```

---

## 10. 로그 / 모니터링

- 로그 포맷: `2026-05-08 14:00:00 [logger.name] LEVEL: 메시지`
- 로그 레벨: `LOG_LEVEL` 환경변수로 조절 (`INFO` 기본, 디버깅 시 `DEBUG`)
- stdout으로 출력되므로 CloudWatch Logs 연동 가능 (Docker `awslogs` 드라이버)

주요 모니터링 포인트:
- `[EXAONE] 로딩 완료` — 부팅 완료
- `[VAD] ... 발화 종료` — 사용자 발화 감지
- `[LLM] ... XX초` — 응답 생성 시간 (정상 ~5~15초)
- `ERROR` 레벨 메시지 — 서비스 이상 신호
