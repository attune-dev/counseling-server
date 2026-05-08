# Frontend ↔ Backend WebSocket API

상담 서버와 주고받는 모든 메시지의 사양입니다.

---

## 1. 연결

```
ws://<host>:8000/ws/counseling/{ticket_id}
```

- `ticket_id`: 세션 식별자 (URL 경로). 서버는 이를 키로 세션 상태를 관리합니다.
- 연결 직후 서버가 `connected` 메시지를 1회 전송합니다.

---

## 2. Client → Server

### 2-1. JSON 메시지 (text frame)

#### `setup` — 초기 상담 설정 (연결 직후 1회)

```json
{
  "type": "setup",
  "data": {
    "topic": "직장 스트레스",
    "mood": "불안",
    "content": "상사와의 관계가 힘들어요"
  },
  "session_id": "선택사항 — 서버는 URL ticket_id를 우선 사용",
  "timestamp": 1715156400.123
}
```

서버 처리:
- GPT-4o-mini로 5단계 CBT 플랜 생성 (~30초)
- EXAONE으로 첫 발화 생성 (병렬, ~5초)
- 완료 시 `initial_questions` 응답

#### `control` — 발화/세션 제어

```json
{ "type": "control", "data": "END_OF_SPEECH" }   // 사용자 발화 종료 (= 답변 요청)
{ "type": "control", "data": "END_OF_SESSION" }  // 상담 종료 (소켓 정리)
```

### 2-2. 바이너리 메시지 (binary frame)

| 헤더 (1 byte) | 페이로드 | 설명 |
|---|---|---|
| `0x01` | float32 PCM **16kHz mono** raw bytes | 음성 청크 |
| `0x02` | JPEG bytes | 영상 프레임 |

**제약:**
- 1메시지 ≤ 1MB (서버 측 DoS 방지 한도)
- 음성: 2초 단위 청크 권장
- 영상: 2초 간격 1프레임 권장 (얼굴 감정 분석용)

---

## 3. Server → Client

모든 메시지는 JSON. 공통 필드: `status` (메시지 종류 식별자).

### 3-1. `connected` — 연결 수락

```json
{
  "status": "connected",
  "message": "상담실에 입장하였습니다.",
  "next_action": null
}
```

### 3-2. `initial_questions` — 초기 상담 준비 완료

setup 처리 완료 후 전송. AI 상담사의 첫 발화와 1단계 정보 포함.

```json
{
  "status": "initial_questions",
  "message": "지금 마음이 어떻게 느껴?",
  "step_status": {
    "step": 1,
    "title": "공감 형성",
    "goal": "내담자가 느끼는 불안을 이해하고...",
    "question_idx": 0,
    "total_questions": 4,
    "current_question": "요즘 어떤 기분이 드는지...",
    "total_steps": 5,
    "complete": false
  }
}

### 3-3. `processing` — 답변 생성 시작 (END_OF_SPEECH 직후)

```json
{ "status": "processing", "message": "답변 생성 중..." }
```
→ 로딩 스피너 표시 권장.

### 3-4. `stt_done` — 음성 인식 결과

```json
{ "status": "stt_done", "text": "네, 요즘 너무 힘들어요" }
```
→ 사용자 발화를 채팅창에 표시.

### 3-5. `response` — AI 상담사 응답

```json
{
  "status": "response",
  "message": "그 마음, 충분히 이해돼요. ...",
  "step_status": { ... }
}
```

### 3-6. `awaiting_empathy` — 단계 정리 대기

해당 단계 마지막 질문이 끝남 → 다음 발화로 정리 발화가 나올 예정. 별도 UI 변화는 선택.

```json
{
  "status": "awaiting_empathy",
  "step_status": { ... 현재 단계 ... }
}
```

### 3-7. `awaiting_transition` — 단계 전환 확인 대기

서버가 정리 발화 + "다음 단계로 넘어가도 괜찮을까요?" 같은 질문을 함. 사용자가 **"네/좋아요/괜찮"** 같은 긍정 키워드를 말하면 전환됨.

```json
{
  "status": "awaiting_transition",
  "step_status": { ... 현재 단계 ... }
}
```

> 사용자에게 "예/아니오"로 답하라는 UX 힌트를 보여주면 좋음.

### 3-8. `awaiting_completion` — 5단계 종료 확인 대기

5단계 마지막 → 사용자가 긍정 응답하면 상담 종료.

```json
{
  "status": "awaiting_completion",
  "step_status": { ... 5단계 ... }
}
```

### 3-9. `step_changed` — 단계 전환 또는 상담 종료

```json
// 다음 단계로 전환
{
  "status": "step_changed",
  "transition": "step_changed",
  "step_status": { "step": 2, "title": "문제 탐색", ... }
}

// 전체 상담 종료
{
  "status": "step_changed",
  "transition": "counseling_complete",
  "step_status": { "step": 5, "title": "상담 완료", "complete": true, ... }
}
```

---

## 4. `step_status` 필드 레퍼런스

| 필드 | 타입 | 설명 |
|------|------|------|
| `step` | int | 현재 단계 번호 (1~5) |
| `title` | string | 단계명 ("공감 형성", "문제 탐색", "사고 전환", "행동 계획", "마무리") |
| `goal` | string | GPT가 내담자 맞춤으로 생성한 단계 목표 |
| `question_idx` | int | 현재 단계 내 질문 인덱스 (0-based) |
| `total_questions` | int | 현재 단계 총 질문 수 |
| `current_question` | string | 현재 질문 텍스트 |
| `total_steps` | int | 항상 5 |
| `complete` | bool | 전체 상담 완료 여부 (`true`면 `step` 등 일부 필드 없을 수 있음) |

---

## 5. 메시지 흐름 (state machine)

```
[연결]
  → connected

[setup 전송]
  → initial_questions

[정상 턴]
  END_OF_SPEECH → processing → stt_done → response

[단계 마지막 질문 처리 후]
  END_OF_SPEECH → processing → stt_done → response → awaiting_empathy
  END_OF_SPEECH → processing → stt_done → response (정리 발화) → awaiting_transition
  사용자 "네"  → END_OF_SPEECH → step_changed (다음 단계 정보)
  사용자 거부  → END_OF_SPEECH → response (자유 탐색, awaiting_transition 유지)

[5단계 마지막]
  END_OF_SPEECH → processing → stt_done → response → awaiting_completion
  사용자 "네"  → END_OF_SPEECH → step_changed (transition: counseling_complete)
```

---

## 6. 오디오 포맷 상세

- 샘플레이트 **16,000 Hz**
- 모노
- float32 PCM raw (WAV 헤더 없음)
- 청크 송신 시 1바이트 헤더 `0x01` 선행
- 브라우저 마이크 입력은 보통 44.1k/48k → **16k로 다운샘플링 필수**

---

## 7. 에러/연결 끊김

- 잘못된 페이로드(예: 1MB 초과 바이너리, JSON 파싱 실패)는 **서버에서 무시**되며 클라이언트로 별도 알림이 가지 않습니다.
- 서버 측 예외는 서버 로그에만 기록됩니다.
- 클라이언트는 `WebSocket.onclose` / `onerror` 핸들링으로 연결 상태를 직접 추적해야 합니다.
