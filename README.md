# Attune Counseling Server

AI 기반 실시간 CBT 심리상담 WebSocket 서버입니다.
클라이언트로부터 음성 청크와 영상 프레임을 WebSocket으로 수신하여
**VAD → STT → 텍스트/음성/얼굴 감정 분석 → 엔트로피 기반 감정 융합 → 5-Step CBT LLM 응답 생성** 파이프라인을 처리합니다.

---

## 기술 스택

| 항목 | 내용 |
|------|------|
| 언어 | Python 3.10+ |
| 웹 프레임워크 | FastAPI + Uvicorn |
| 음성 감지 (VAD) | Silero VAD (PyTorch) |
| 음성 인식 (STT) | faster-whisper (CUDA) |
| 텍스트 감정 | klue/bert FP16 (로컬 `models/text-emotion-final`, CUDA) |
| 음성 감정 | Wav2Vec2 (로컬 `models/voice-emotion-final`, CPU) |
| 얼굴 감정 | DeepFace |
| 감정 융합 | 엔트로피 기반 동적 가중치 (Shannon Entropy → 역수 → 정규화) |
| 상담 플랜 생성 | GPT-4o-mini (OpenAI API) |
| LLM | EXAONE-3.5-7.8B-Instruct 8-bit 양자화 (HuggingFace 자동 다운로드) |
| 인증 / 캐시 | Redis (ticket_id → userId 조회) |
| 외부 백엔드 | Spring (상담 종료 시 HTTP POST 리포트 송신) |

---

## 프로젝트 구조

```
counseling_server/
├── app/
│   ├── main.py                    # FastAPI 앱, WebSocket 엔드포인트, 로깅 설정
│   ├── schemas.py                 # WebSocket 메시지 스키마
│   ├── core/
│   │   ├── config.py              # 환경변수 설정 (pydantic-settings)
│   │   └── container.py           # AI 모델 싱글턴 컨테이너
│   └── services/
│       ├── audio_processor.py     # VAD 침묵 감지 + 배치 STT 처리
│       ├── pipeline.py            # 상담 데이터 흐름 오케스트레이터 (턴 상태 머신)
│       ├── session_manager.py     # WebSocket 연결 관리 + Redis 인증 + 데이터 라우팅
│       ├── counseling_session.py  # 세션 오케스트레이터 (플랜 생성 → 첫 발화 병렬)
│       ├── plan_generator.py      # GPT-4o-mini 5-Step CBT 플랜 생성
│       ├── step_manager.py        # 스텝별 질문 진행 및 전환 관리
│       ├── history_manager.py     # 대화 히스토리 + GPT-4o-mini 단계 요약 + 턴 감정 기록
│       ├── emotion_monitor.py     # 모달리티별 부정 감정 감지 및 하이라이트 저장
│       ├── redis_client.py        # ticket_id → userId 인증 조회 (Redis)
│       ├── report_builder.py      # Spring DTO 페이로드 빌드 (enum 변환 + emotionFlow 압축)
│       ├── report_insights.py     # GPT-4o-mini 단일 호출로 리포트 4필드 생성
│       └── spring_client.py       # Spring 백엔드 HTTP POST (X-Internal-API-Key)
├── ai_modules/
│   ├── interfaces.py              # AI 모델 베이스 클래스
│   ├── models.py                  # 실제 AI 모델 구현체 (TextEmotionModel, Wav2VecEmotionModel, EmotionFusionModel, ExaoneLLMModel)
│   └── schemas.py                 # AI 모델 입출력 스키마
├── models/                        # ⚠️ Git 제외 (팀 내부 공유)
│   ├── text-emotion-final/        # 텍스트 감정 분류 (klue/bert)
│   └── voice-emotion-final/       # 음성 감정 분류 (Wav2Vec2)
├── test_e2e.py                    # E2E 통합 테스트
├── .env.example                   # 환경변수 템플릿
├── requirements.txt
└── README.md
```

---

## 상담 흐름

```
[연결]
  WebSocket 접속 → Redis로 ticket_id → userId 조회
    ├─ 없음: "auth_failed" 응답 후 즉시 close (1008)
    └─ 있음: 세션 초기화 → "connected" 응답

[초기 상담 설정]
  클라이언트: {"type": "setup", "data": {"topic", "mood", "content"}}
  서버 (병렬 실행):
    ├─ GPT-4o-mini → 내담자 맞춤 5-Step CBT 플랜 + 인지왜곡 분석 생성
    └─ EXAONE → 첫 인사 발화 즉시 생성 (플랜 완료 기다리지 않음)
    → StepManager + HistoryManager 초기화
    → "initial_questions" 응답 (첫 발화 + step_status)

[멀티턴 상담]
  오디오 청크 (0x01 헤더) → VAD 필터 → STT 버퍼 누적
                          → Wav2Vec2 백그라운드 (Lock 기반 throttle) → 음성 감정 버퍼
  영상 프레임 (0x02 헤더) → DeepFace (run_in_executor) → 얼굴 감정 버퍼

  END_OF_SPEECH → 턴 상태에 따라 분기:
    ┌─ normal           → generate_response() — 감정 분석 + 동적 system_prompt + EXAONE 응답
    ├─ awaiting_empathy → generate_wrap_up_response() — 단계 정리 + 전환 여부 질문
    ├─ awaiting_transition:
    │   긍정 키워드("네","좋아요" 등) → execute_step_transition() — 다음 단계로 전환
    │   그 외               → generate_free_response() — 자유 탐색 응답
    └─ awaiting_completion:
        긍정 키워드 → execute_counseling_complete() — 상담 종료
        그 외       → generate_free_response()

[5단계 상담 종료]
  사용자 긍정 응답 → execute_counseling_complete:
    └─ fire-and-forget 백그라운드:
        ├─ GPT-4o-mini로 리포트 4필드(summary/strengths/actionItems/keywords) 생성
        ├─ 페이로드 빌드 (enum 변환, emotionFlow 연속 중복 제거)
        └─ Spring `/internal/counseling/report` 로 HTTP POST (X-Internal-API-Key 헤더)

[세션 종료]
  {"type": "control", "data": "END_OF_SESSION"} → 세션 정리
```

### 턴 상태 머신

```
normal
  └─ 마지막 질문(step < 5) → [응답] → awaiting_empathy
  └─ 마지막 질문(step == 5) → [응답] → awaiting_completion
  └─ 일반 질문 → [응답] → normal

awaiting_empathy
  └─ [정리 응답] → awaiting_transition

awaiting_transition
  └─ 긍정 → execute_step_transition → normal (다음 스텝)
  └─ 부정 → generate_free_response → awaiting_transition 유지

awaiting_completion
  └─ 긍정 → execute_counseling_complete → complete
  └─ 부정 → generate_free_response → awaiting_completion 유지
```

### 5-Step CBT 플랜 구조

GPT-4o-mini가 내담자 정보(topic, mood, content)를 바탕으로 맞춤 생성합니다.

| 단계 | 이름 | 질문 수 |
|------|------|---------|
| Step 1 | 공감 형성 | 2~5개 |
| Step 2 | 문제 탐색 | 3~7개 |
| Step 3 | 사고 전환 | 3~8개 |
| Step 4 | 행동 계획 | 2~5개 |
| Step 5 | 마무리 | 1~3개 |

- 각 단계는 **깔때기 구조** 질문 (앞 답변이 다음 질문으로 자연스럽게 이어짐)
- 단계 전환 시 GPT-4o-mini로 이전 단계 대화 **요약 생성** → 다음 단계 system_prompt에 누적 주입
- 인지왜곡 분석 (`core_problem`, `cognitive_pattern`)이 매 턴 system_prompt에 포함됨

### 감정 융합 (엔트로피 기반 동적 가중치)

```
1. 각 모달리티 softmax 확률 분포 → Shannon 엔트로피 H = -Σ p·ln(p)
2. 역수 → confidence (c = 1 / (H + ε))
3. 정규화 → 가중치 (w = c / Σc, 합산 = 1)
4. 가중합 → 최종 감정

→ 확신이 높은(엔트로피가 낮은) 모달리티에 자동으로 큰 가중치 부여
```

---

## 시작하기

### 1. 파이썬 가상환경 생성 및 활성화

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

> PyTorch는 CUDA 12.4 빌드로 별도 설치가 필요할 수 있습니다.
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
> ```

### 3. AI 모델 파일 배치

`models/` 폴더는 Git에 포함되지 않습니다. 팀 내부 공유 스토리지에서 다운로드 후 아래 구조로 배치하세요.

```
counseling_server/
└── models/
    ├── text-emotion-final/   # klue/bert 텍스트 감정 분류
    └── voice-emotion-final/  # Wav2Vec2 음성 감정 분류
```

> EXAONE-3.5-7.8B-Instruct는 서버 최초 실행 시 HuggingFace에서 자동 다운로드됩니다 (~16GB).
> `trust_remote_code=True` 필수 (EXAONE 커스텀 모델링 코드 포함).

### 4. 환경변수 설정

```bash
cp .env.example .env
```

주요 설정값:

| 키 | 기본값 | 설명 |
|----|--------|------|
| `OPENAI_API_KEY` | _(필수)_ | GPT-4o-mini 플랜 생성 + 단계 요약 + 리포트 인사이트용 |
| `WHISPER_MODEL_SIZE` | `small` | Whisper 모델 크기 |
| `WHISPER_DEVICE` | `cuda` | STT 디바이스 |
| `CBT_LLM_DEVICE` | `cuda` | EXAONE LLM 디바이스 |
| `TEXT_EMOTION_DEVICE` | `cuda` | 텍스트 감정 BERT 디바이스 |
| `AUDIO_EMOTION_DEVICE` | `cpu` | Wav2Vec2 음성 감정 디바이스 |
| `NEGATIVE_EMOTION_THRESHOLD` | `0.65` | 부정 감정 감지 임계값 |
| `APP_PORT` | `8000` | 서버 포트 |
| `REDIS_URL` | `redis://localhost:6379/0` | ticket_id → userId 조회용 |
| `DEV_SKIP_REDIS_AUTH` | `false` | 테스트 전용. true면 Redis 없이 ticket_id 그대로 userId로 사용 |
| `SPRING_BACKEND_URL` | `http://localhost:8080` | 상담 종료 리포트 송신 대상 |
| `SPRING_INTERNAL_API_KEY` | _(필수, 운영)_ | `X-Internal-API-Key` 헤더 값 |
| `USE_DUMMY_MODELS` | `false` | 테스트 전용. true면 AI 모델 전부 가짜 응답기로 대체 (GPU 불필요) |

### 5. 서버 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

서버 상태 확인: `http://localhost:8000/`

---

## WebSocket API

```
ws://localhost:8000/ws/counseling/{ticket_id}
```

### 클라이언트 → 서버

```json
// 초기 상담 설정 (연결 직후 1회)
{
  "type": "setup",
  "data": {
    "topic": "가정사",
    "mood": "우울",
    "content": "엄마와 사이 안좋음"
  }
}

// 음성 바이너리: bytes([0x01]) + float32 PCM 16kHz mono
// 영상 바이너리: bytes([0x02]) + JPEG bytes

// 발화 종료
{ "type": "control", "data": "END_OF_SPEECH" }

// 세션 종료
{ "type": "control", "data": "END_OF_SESSION" }
```

### 서버 → 클라이언트

```json
// 연결 수락
{ "status": "connected", "message": "상담실에 입장하였습니다." }

// 초기 상담 설정 완료 (첫 발화 + 현재 단계 정보)
{
  "status": "initial_questions",
  "message": "첫 상담사 발화 텍스트",
  "step_status": {
    "step": 1, "title": "공감 형성", "goal": "...",
    "question_idx": 0, "total_questions": 3,
    "current_question": "...", "total_steps": 5, "complete": false
  }
}

// 답변 생성 중 (로딩 스피너용)
{ "status": "processing", "message": "답변 생성 중..." }

// STT 변환 결과
{ "status": "stt_done", "text": "STT 변환 결과 텍스트" }

// AI 상담사 응답
{ "status": "response", "message": "AI 상담사 응답 텍스트", "step_status": {...} }

// 단계 정리 완료, 전환 여부 대기
{ "status": "awaiting_empathy", "step_status": {...} }

// 전환 질문 완료, 내담자 확인 대기
{ "status": "awaiting_transition", "step_status": {...} }

// 5단계 마무리, 종료 확인 대기
{ "status": "awaiting_completion", "step_status": {...} }

// 단계 전환 완료 (또는 상담 전체 종료)
{
  "status": "step_changed",
  "transition": "step_changed" | "counseling_complete",
  "step_status": { ...다음 단계 정보... }
}
```

---

## 테스트

| 스크립트 | 용도 | 필요 환경 |
|---|---|---|
| `python test_report_build.py` | Spring 페이로드 JSON 모양 검증 | 네트워크 X (즉시) |
| `python test_report_build.py --with-gpt` | + GPT 인사이트 생성 검증 | OpenAI API 키만 |
| `python test_mock_spring.py` | Mock Spring 서버 (포트 8080) — POST 받으면 콘솔 출력 | uvicorn만 |
| `python test_pipeline.py [wav]` | WebSocket 클라이언트 시뮬레이터 | AI 서버 가동 필요 |

### GPU 없이 전체 흐름 테스트

`.env`에 다음 두 줄 추가하고 서버 실행:
```env
USE_DUMMY_MODELS=true
DEV_SKIP_REDIS_AUTH=true
```
- AI 모델 부팅 즉시 완료 (GPU 불필요)
- Redis 없어도 `ticket_id` 가 자동으로 user_id로 사용됨
- `python test_pipeline.py` 로 전체 흐름 검증 가능

### 자세한 검증 시나리오

→ [DEPLOYMENT.md §12 테스트/스테이징 모드](DEPLOYMENT.md) 참고

---

## VRAM 사용량 참고

| 모델 | 디바이스 | 사용량 |
|------|----------|--------|
| Whisper small | CUDA | ~0.5GB |
| EXAONE-3.5-7.8B (8-bit 양자화) | CUDA | ~8GB |
| 텍스트 감정 BERT (FP16) | CUDA | ~0.2GB |
| 음성 감정 Wav2Vec2 | CPU | - |
| **합계** | | **~8.7GB** |

> VRAM 8GB 이상 GPU 권장 (RTX 3070 / RTX 4060 Ti 이상).
