import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional

from ai_modules.schemas import (
    CounselingSetup, EmotionResult, FaceInput, LLMContext, LLMResponse, STTInput, STTOutput
)
from app.core.container import AIContainer
from app.services.audio_processor import AudioProcessor
from app.services.counseling_session import CounselingSession

logger = logging.getLogger(__name__)

# ── 감정 톤 매핑 (system prompt에 주입) ──────────────────────────
EMOTION_TONE_MAP = {
    "sad": "따뜻하고 부드러운 톤으로 천천히 말하세요. 감정을 충분히 수용하고, 성급하게 전환하지 마세요.",
    "angry": "차분하고 안정적인 톤으로 말하세요. 감정의 정당성을 먼저 인정한 후, 분노 아래의 욕구를 탐색하세요.",
    "fear": "안심시키는 톤으로 안전감을 제공하세요. 두려움의 구체적 대상을 명확히 하고, 현실 검증을 부드럽게 시도하세요.",
    "disgust": "판단 없이 수용하는 톤으로 말하세요. 내담자의 가치관과 경계를 존중하세요.",
    "happy": "함께 기뻐하되 탐색을 유지하세요. 긍정 경험의 의미를 탐색하고, 과잉 낙관에 주의하세요.",
    "surprise": "호기심 있는 탐색적 톤으로 말하세요. 놀란 이유와 기대 불일치를 탐색하세요.",
    "neutral": "",
}

# ── 단계별 금지 사항 ─────────────────────────────────────────────
STEP_CONSTRAINTS = {
    1: (
        "절대 금지 표현: '~해보세요', '~어떨까요', '~하는 건', '~해볼까요', '~드릴게요'.\n"
        "조언/제안/해결책/격려 전부 금지. 감정을 있는 그대로 반영만 하세요.\n"
        "질문은 감정 탐색 질문 딱 하나만 허용."
    ),
    2: (
        "절대 금지 표현: '~해보세요', '~어떨까요', '~하는 건', '~해볼까요'.\n"
        "해결책/대안/조언/격려 전부 금지. 인지왜곡 지적 금지.\n"
        "자동적 사고를 탐색하는 질문만 하세요."
    ),
    3: (
        "직접 교정하거나 답을 알려주지 마세요.\n"
        "인지왜곡 이름을 반드시 직접 명명하세요 (흑백사고, 파국화, 과잉일반화 등).\n"
        "소크라테스식 질문으로 내담자가 스스로 발견하게 유도하세요.\n"
        "반드시 질문으로 마무리하세요. 진술로 끝내지 마세요."
    ),
    4: (
        "상담사가 구체적 활동/방법/기법을 절대 제안하지 마세요.\n"
        "내담자가 스스로 목표를 말하게 질문으로만 유도하세요.\n"
        "열린 질문만 허용."
    ),
    5: (
        "새 주제 금지. 반드시 세 가지를 자연스럽게 포함하세요:\n"
        "①오늘 상담 핵심 요약 ②다음 주까지 할 과제 ③따뜻한 격려.\n"
        "'핵심 요약:', '과제:' 같은 라벨 출력 금지. 자연스러운 대화체로."
    ),
}


class CounselingPipeline:
    """
    WebSocket에서 수신한 데이터를 버퍼링하고 AI 모델을 순서대로 호출하는 오케스트레이터.
    오디오(VAD/STT) 처리는 AudioProcessor에 위임한다.
    """

    def __init__(self, container: AIContainer):
        self.container = container
        self.audio = AudioProcessor(container)
        self.session = CounselingSession(container)
        # 세션별 비오디오 버퍼
        self._counseling_setup: Dict[str, Optional[CounselingSetup]] = {}
        self._face_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._voice_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._stt_text_buffer: Dict[str, List[str]] = {}
        # 음성 감정 분석용 마지막 PCM 스냅샷
        self._last_pcm_audio: Dict[str, bytes] = {}
        # 음성 감정 분석 throttle용 세션별 Lock (동시 실행 1개로 제한, 폭주 방지)
        self._voice_emotion_locks: Dict[str, asyncio.Lock] = {}
        # 세션별 턴 상태 (공감 턴 분리)
        self._turn_state: Dict[str, str] = {}

    # ══════════════════════════════════════════════════════════════
    # 세션 수명 주기
    # ══════════════════════════════════════════════════════════════

    def init_session(self, session_id: str) -> None:
        self.audio.init_session(session_id)
        self.session.init_session(session_id)
        self._counseling_setup[session_id] = None
        self._face_emotion_buffer[session_id] = []
        self._voice_emotion_buffer[session_id] = []
        self._stt_text_buffer[session_id] = []
        self._last_pcm_audio[session_id] = b""
        self._voice_emotion_locks[session_id] = asyncio.Lock()
        self._turn_state[session_id] = "normal"
        logger.info(f"세션 초기화: {session_id}")

    def cleanup_session(self, session_id: str) -> None:
        self.audio.cleanup_session(session_id)
        self.session.cleanup_session(session_id)
        for buf in (
            self._counseling_setup,
            self._face_emotion_buffer,
            self._voice_emotion_buffer,
            self._stt_text_buffer,
            self._last_pcm_audio,
            self._voice_emotion_locks,
            self._turn_state,
        ):
            buf.pop(session_id, None)
        logger.info(f"세션 정리: {session_id}")

    # ══════════════════════════════════════════════════════════════
    # 초기 상담 설정 / 오디오 입력
    # ══════════════════════════════════════════════════════════════

    async def start_transcription_worker(self, session_id: str) -> None:
        await self.audio.start_worker(session_id)

    def setup_counseling(self, session_id: str, topic: str, mood: str, content: str) -> None:
        self._counseling_setup[session_id] = CounselingSetup(
            topic=topic, mood=mood, content=content
        )
        logger.info(f"[Setup] {session_id}: topic={topic} / mood={mood}")

    def append_audio_chunk(self, session_id: str, chunk: bytes) -> bool:
        return self.audio.append_chunk(session_id, chunk)

    # ══════════════════════════════════════════════════════════════
    # 얼굴 감정 / 음성 감정 처리
    # ══════════════════════════════════════════════════════════════

    async def process_face_frame(self, session_id: str, image_bytes: bytes) -> None:
        try:
            face_input = FaceInput(video_frame=image_bytes)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self.container.face_emotion.analyze, face_input
            )
            if session_id in self._face_emotion_buffer:
                self._face_emotion_buffer[session_id].append(result)
                logger.info(f"[Face] {session_id}: {result.primary_emotion} {result.probabilities}")
        except Exception as e:
            logger.error(f"[Face] {session_id}: 분석 오류: {e}")

    async def _analyze_voice_emotion(self, session_id: str, voice_pcm: bytes) -> None:
        try:
            t0 = time.time()
            loop = asyncio.get_running_loop()
            voice_emotion = await loop.run_in_executor(
                None, self.container.audio_emotion.analyze, STTInput(audio_data=voice_pcm)
            )
            if session_id in self._voice_emotion_buffer:
                self._voice_emotion_buffer[session_id].append(voice_emotion)
                logger.info(
                    f"[VoiceEmo] {session_id}: {voice_emotion.primary_emotion} "
                    f"{voice_emotion.probabilities} ({time.time() - t0:.2f}초)"
                )
        except Exception as e:
            logger.error(f"[VoiceEmo] {session_id}: 오류: {e}")

    async def analyze_voice_emotion_throttled(self, session_id: str, voice_pcm: bytes) -> None:
        lock = self._voice_emotion_locks.get(session_id)
        if lock is None or lock.locked():
            return
        async with lock:
            await self._analyze_voice_emotion(session_id, voice_pcm)

    # ══════════════════════════════════════════════════════════════
    # 발화 종료 → STT
    # ══════════════════════════════════════════════════════════════

    async def on_speech_end(self, session_id: str) -> Optional[STTOutput]:
        accumulated = await self.audio.wait_and_get_text(session_id)
        if not accumulated:
            logger.warning(f"[SpeechEnd] {session_id}: 텍스트 없음, 건너뜀")
            return None

        if session_id not in self._stt_text_buffer:
            return None

        self._stt_text_buffer[session_id].append(accumulated)
        logger.info(f"[SpeechEnd] {session_id}: 최종 텍스트 = '{accumulated}'")
        return STTOutput(text=accumulated, language="ko")

    # ══════════════════════════════════════════════════════════════
    # 유틸리티
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _average_emotion(results: List[EmotionResult]) -> EmotionResult:
        if not results:
            return EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})
        if len(results) == 1:
            return results[0]
        from collections import defaultdict
        avg: dict = defaultdict(float)
        for r in results:
            for emotion, prob in r.probabilities.items():
                avg[emotion] += prob
        n = len(results)
        prob_dict = {k: round(v / n, 3) for k, v in avg.items()}
        primary = max(prob_dict, key=prob_dict.get)
        return EmotionResult(primary_emotion=primary, probabilities=prob_dict)

    def _run_emotion_pipeline(self, session_id: str, accumulated_text: str, loop):
        """감정 분석 + 융합 실행 (동기). run_in_executor 내부에서 호출."""
        # 텍스트 감정
        text_emo = self.container.text_emotion.analyze(accumulated_text)
        # face/voice 평균
        face_emo = self._average_emotion(self._face_emotion_buffer.get(session_id, []))
        voice_emo = self._average_emotion(self._voice_emotion_buffer.get(session_id, []))
        # 3모달 융합
        fused_emo = self.container.fusion.fuse(text_emo, voice_emo, face_emo)
        return text_emo, voice_emo, face_emo, fused_emo

    # ══════════════════════════════════════════════════════════════
    # 동적 시스템 프롬프트 빌드
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _build_dynamic_system_prompt(step_mgr, current_q, history_mgr, detected_emotion="neutral") -> str:
        COUNSELOR_NAME = "멍박사님"
        KOREAN_RULE = "반드시 한국어로만 답변하세요. 영어 단어는 절대 사용하지 마세요."
        analysis = step_mgr.analysis or {}
        step = step_mgr.current_step
        step_num = step_mgr.step_number

        parts = [
            "[역할]",
            f"따뜻하고 공감적인 CBT(인지행동치료) 전문 상담사 '{COUNSELOR_NAME}'.",
            KOREAN_RULE,
            "",
            "[내담자 분석]",
            f"- 핵심 문제: {analysis.get('core_problem', '미분석')}",
            f"- 예상 인지 왜곡: {analysis.get('cognitive_pattern', '미분석')}",
        ]

        # 감정 톤 주입
        tone_instruction = EMOTION_TONE_MAP.get(detected_emotion, "")
        if detected_emotion != "neutral":
            parts.extend([
                f"- 현재 내담자 감정: {detected_emotion}",
                "",
                "[감정 대응 지시]",
                tone_instruction,
            ])
        else:
            parts.append("")

        parts.extend([
            f"[현재 단계: Step {step_num} - {step['name']}]",
            f"- 목표: {step.get('goal', '')}",
            f"- 집중 포인트(CBT 기법): {step.get('focus', '')}",
        ])

        # 단계별 금지 사항
        constraint = STEP_CONSTRAINTS.get(step_num, "")
        if constraint:
            parts.extend(["", f"[금지 사항]\n{constraint}"])

        # 이전 단계 요약 주입
        if history_mgr:
            summaries = history_mgr.get_step_summaries()
            if summaries:
                parts.extend(["", "[이전 단계에서 발견한 것]"])
                for sn in sorted(summaries.keys()):
                    info = summaries[sn]
                    parts.append(f"- Step {sn} ({info['step_name']}): {info['summary']}")

        # 단계별 문장 수 규칙
        step_config = {1: 3, 2: 3, 3: 3, 4: 3, 5: 5}
        max_s = step_config.get(step_num, 3)
        if max_s > 3:
            length_rule = f"길이: {max_s}문장 이내. 요약, 과제, 격려를 모두 포함."
        else:
            length_rule = "길이: 반드시 2~3문장만. 4문장 이상 절대 금지."

        parts.extend([
            "",
            "[규칙]",
            "호칭: '당신' 절대 금지. 주어 생략하고 바로 서술.",
            "말투: 존댓말(~요, ~습니다) 일관 유지. 반말 절대 금지.",
            length_rule,
            "언어: 한국어만. 영어/메모/이모지 절대 금지.",
        ])

        if current_q:
            parts.extend([
                "",
                "[이번 응답 지시 - 반드시 준수]",
                "① 사용자의 말에 1~2문장으로 진심 어린 공감을 표현하세요.",
                "② 아래 질문을 자연스럽게 이어서 마지막에 물어보세요:",
                f"   {current_q}",
                "③ 이 질문 외에 다른 질문은 절대 추가하지 마세요.",
            ])

        return "\n".join(parts)

    # ══════════════════════════════════════════════════════════════
    # user_text 힌트 단계별 분기
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _build_user_text_hint(
        accumulated_text: str, step_mgr, current_q: Optional[str], is_last_q: bool
    ) -> str:
        step_num = step_mgr.step_number

        if is_last_q and current_q:
            if step_num == 5:
                return (
                    f"{accumulated_text}\n\n"
                    f"[이 상담의 마지막입니다. "
                    f"① 오늘 상담에서 발견한 핵심 통찰을 정리하세요. "
                    f"② 다음 주까지 실천할 과제를 자연스럽게 제안하세요. "
                    f"③ 따뜻한 격려로 마무리하면서, 상담을 마무리해도 괜찮은지 물어보세요.]"
                )
            elif step_num == 3:
                cognitive_pattern = step_mgr.analysis.get("cognitive_pattern", "")
                return (
                    f"{accumulated_text}\n\n"
                    f"[지시: ① 반드시 인지왜곡 이름을 직접 언급하세요. 참고: {cognitive_pattern[:80]} "
                    f"② 사용자의 말에 충분히 공감하세요. 정리나 전환 질문은 하지 마세요.]"
                )
            elif step_num == 4:
                return (
                    f"{accumulated_text}\n\n"
                    f"[사용자의 말에 충분히 공감하세요. 정리나 전환 질문은 하지 마세요. "
                    f"절대 구체적 방법이나 활동을 제안하지 마세요.]"
                )
            else:
                return (
                    f"{accumulated_text}\n\n"
                    f"[사용자의 말에 충분히 공감하세요. 정리나 전환 질문은 하지 마세요.]"
                )
        elif current_q:
            if step_num == 3:
                cognitive_pattern = step_mgr.analysis.get("cognitive_pattern", "")
                return (
                    f"{accumulated_text}\n\n"
                    f"[지시: ① 반드시 인지왜곡 이름(과잉일반화, 흑백사고 등)을 직접 언급하세요. "
                    f"참고 분석: {cognitive_pattern[:80]} "
                    f"② 아래 질문의 의도를 살려 자연스럽게 질문으로 마무리하세요: {current_q}]"
                )
            elif step_num == 4:
                return (
                    f"{accumulated_text}\n\n"
                    f"[절대 구체적 방법/활동/기법을 제안하지 마세요. 열린 질문만 하세요. "
                    f"아래 질문의 의도를 살려서 자연스럽게 질문하세요: {current_q}]"
                )
            else:
                return (
                    f"{accumulated_text}\n\n"
                    f"[아래 질문의 의도를 살려서, 자연스러운 말투로 질문하세요: {current_q}]"
                )
        return accumulated_text

    # ══════════════════════════════════════════════════════════════
    # 버퍼 초기화 (매 턴 종료 시)
    # ══════════════════════════════════════════════════════════════

    def _clear_turn_buffers(self, session_id: str) -> None:
        self._stt_text_buffer[session_id] = []
        self._face_emotion_buffer[session_id] = []
        self._voice_emotion_buffer[session_id] = []

    # ══════════════════════════════════════════════════════════════
    # 메인 응답 생성 (일반 턴)
    # ══════════════════════════════════════════════════════════════

    async def generate_response(self, session_id: str) -> Optional[dict]:
        accumulated_text = " ".join(self._stt_text_buffer.get(session_id, []))
        if not accumulated_text:
            logger.warning(f"[Generate] {session_id}: 누적 텍스트 없음, 건너뜀")
            return None

        face_emotions = self._face_emotion_buffer.get(session_id, [])
        voice_emotions = self._voice_emotion_buffer.get(session_id, [])
        history_mgr = self.session.get_history_manager(session_id)
        history = history_mgr.get_recent_turns() if history_mgr else []
        loop = asyncio.get_running_loop()

        # ── 현재 단계 + 질문 정보 ──────────────────────────────
        step_mgr = self.session.get_step_manager(session_id)
        if not step_mgr:
            logger.warning(f"[Generate] {session_id}: StepManager 없음, 건너뜀")
            return None

        current_q = step_mgr.get_current_question()
        q_idx = step_mgr.current_question_idx
        questions = step_mgr.get_questions()
        total_q = len(questions)
        is_last_q = (q_idx + 1 >= total_q)
        step_num = step_mgr.step_number

        logger.info(
            f"[Generate] {session_id}: Step {step_num} "
            f"'{step_mgr.current_step['name']}' "
            f"Q{q_idx + 1}/{total_q}: '{current_q}'"
        )

        # ── 텍스트 감정 분석 ──────────────────────────────────
        try:
            t0 = time.time()
            text_emo = await loop.run_in_executor(
                None, self.container.text_emotion.analyze, accumulated_text
            )
            logger.info(
                f"[TextEmo] {session_id}: {text_emo.primary_emotion} "
                f"{text_emo.probabilities} ({time.time() - t0:.2f}초)"
            )
        except Exception as e:
            logger.error(f"[TextEmo] {session_id}: 오류: {e}")
            text_emo = EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})

        # ── 감정 평균화 + 3모달 융합 ──────────────────────────
        face_emo = self._average_emotion(face_emotions)
        voice_emo = self._average_emotion(voice_emotions)

        try:
            fused_emo = self.container.fusion.fuse(text_emo, voice_emo, face_emo)
            logger.info(
                f"[Fusion] {session_id}: text={text_emo.primary_emotion} "
                f"voice={voice_emo.primary_emotion} face={face_emo.primary_emotion} "
                f"→ fused={fused_emo.primary_emotion}"
            )
        except Exception as e:
            logger.error(f"[Fusion] {session_id}: 오류: {e}")
            fused_emo = text_emo

        # ── 턴별 감정 저장 ────────────────────────────────────
        if history_mgr:
            history_mgr.add_turn_emotion(
                fused=fused_emo.primary_emotion,
                text=text_emo.primary_emotion,
                voice=voice_emo.primary_emotion,
                face=face_emo.primary_emotion,
                step=step_num,
            )

        # ── 동적 system_prompt 조립 (감정 톤 포함) ──────────────
        system_prompt = self._build_dynamic_system_prompt(
            step_mgr, current_q, history_mgr,
            detected_emotion=fused_emo.primary_emotion,
        )

        # ── user_text 힌트 단계별 분기 ──────────────────────────
        user_text_for_llm = self._build_user_text_hint(
            accumulated_text, step_mgr, current_q, is_last_q
        )

        # ── LLM 파라미터 결정 ──────────────────────────────────
        if step_num == 5:
            max_tokens = 250
            max_sentences = 5
        elif is_last_q:
            max_tokens = 200
            max_sentences = 5
        else:
            max_tokens = 120
            max_sentences = 3

        llm_context = LLMContext(
            user_text=user_text_for_llm,
            system_prompt=system_prompt,
            face_emotions=face_emotions,
            voice_emotions=voice_emotions,
            text_emotion=text_emo.primary_emotion,
            fused_emotion=fused_emo.primary_emotion,
            history=history,
            max_new_tokens=max_tokens,
            max_sentences=max_sentences,
        )

        # ── LLM 추론 ──────────────────────────────────────────
        t0 = time.time()
        response = await loop.run_in_executor(
            None, self.container.llm.generate_response, llm_context
        )
        logger.info(
            f"[LLM] {session_id}: '{response.reply_text[:80]}' ({time.time() - t0:.2f}초)"
        )

        # ── 히스토리 기록 ──────────────────────────────────────
        if history_mgr:
            history_mgr.add_user_message(accumulated_text)
            history_mgr.add_assistant_message(response.reply_text)

        # ── 공감 턴 분리 판단 ──────────────────────────────────
        pre_advance_status = step_mgr.get_status()

        if is_last_q and step_num < 5:
            # 마지막 Q 공감만 → 정리 턴 대기
            self._turn_state[session_id] = "awaiting_empathy"
            self._clear_turn_buffers(session_id)
            return {
                "llm_response": response,
                "transition": "awaiting_empathy",
                "step_status": pre_advance_status,
                "next_step_status": None,
            }
        elif is_last_q and step_num == 5:
            # 5단계 마지막 → 종료 확인 대기
            self._turn_state[session_id] = "awaiting_completion"
            self._clear_turn_buffers(session_id)
            return {
                "llm_response": response,
                "transition": "awaiting_completion",
                "step_status": pre_advance_status,
                "next_step_status": None,
            }

        # ── 일반 턴: advance_question ──────────────────────────
        transition: Optional[str] = None
        transition = step_mgr.advance_question()
        if transition == "step_changed":
            if history_mgr and pre_advance_status:
                await loop.run_in_executor(
                    None,
                    history_mgr.on_step_transition,
                    pre_advance_status["step"],
                    pre_advance_status["title"],
                )
            logger.info(
                f"[StepMgr] {session_id}: → Step {step_mgr.step_number} "
                f"'{step_mgr.current_step['name']}' 시작"
            )
        elif transition == "counseling_complete":
            if history_mgr and pre_advance_status:
                await loop.run_in_executor(
                    None,
                    history_mgr.on_step_transition,
                    pre_advance_status["step"],
                    pre_advance_status["title"],
                )
            logger.info(f"[StepMgr] {session_id}: 전체 상담 완료")

        post_advance_status = step_mgr.get_status() if not step_mgr.is_complete else step_mgr.get_status()
        self._clear_turn_buffers(session_id)

        return {
            "llm_response": response,
            "transition": transition,
            "step_status": pre_advance_status,
            "next_step_status": post_advance_status,
        }

    # ══════════════════════════════════════════════════════════════
    # 정리 턴 (공감 완료 후 → 키워드 정리 + 전환 질문)
    # ══════════════════════════════════════════════════════════════

    async def generate_wrap_up_response(self, session_id: str) -> Optional[dict]:
        accumulated_text = " ".join(self._stt_text_buffer.get(session_id, []))
        if not accumulated_text:
            return None

        step_mgr = self.session.get_step_manager(session_id)
        history_mgr = self.session.get_history_manager(session_id)
        history = history_mgr.get_recent_turns() if history_mgr else []
        loop = asyncio.get_running_loop()

        # 감정 분석
        try:
            text_emo = await loop.run_in_executor(
                None, self.container.text_emotion.analyze, accumulated_text
            )
        except Exception:
            text_emo = EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})
        face_emo = self._average_emotion(self._face_emotion_buffer.get(session_id, []))
        voice_emo = self._average_emotion(self._voice_emotion_buffer.get(session_id, []))
        fused_emo = self.container.fusion.fuse(text_emo, voice_emo, face_emo)

        sys_prompt = self._build_dynamic_system_prompt(
            step_mgr, None, history_mgr, detected_emotion=fused_emo.primary_emotion
        )

        user_text_for_llm = (
            f"{accumulated_text}\n\n"
            f"[이 단계의 마지막입니다. 반드시 아래 순서대로 하세요: "
            f"① 사용자의 말에 1문장으로 공감하세요. "
            f"② 이 단계에서 내담자가 직접 말한 감정, 경험, 핵심 키워드를 "
            f"구체적으로 언급하며 정리하세요. "
            f"③ 마지막 문장은 반드시 '다음 단계로 넘어가도 괜찮으실까요?' "
            f"형태의 질문으로 끝내세요.]"
        )

        llm_context = LLMContext(
            user_text=user_text_for_llm,
            system_prompt=sys_prompt,
            fused_emotion=fused_emo.primary_emotion,
            history=history,
            max_new_tokens=200,
            max_sentences=4,
        )

        response = await loop.run_in_executor(
            None, self.container.llm.generate_response, llm_context
        )

        if history_mgr:
            history_mgr.add_user_message(accumulated_text)
            history_mgr.add_assistant_message(response.reply_text)

        self._turn_state[session_id] = "awaiting_transition"
        self._clear_turn_buffers(session_id)

        return {
            "llm_response": response,
            "transition": "awaiting_transition",
            "step_status": step_mgr.get_status(),
            "next_step_status": None,
        }

    # ══════════════════════════════════════════════════════════════
    # 자유 탐색 턴 (전환 거부 시 질문 힌트 없이 공감만)
    # ══════════════════════════════════════════════════════════════

    async def generate_free_response(self, session_id: str) -> Optional[dict]:
        accumulated_text = " ".join(self._stt_text_buffer.get(session_id, []))
        if not accumulated_text:
            return None

        step_mgr = self.session.get_step_manager(session_id)
        history_mgr = self.session.get_history_manager(session_id)
        history = history_mgr.get_recent_turns() if history_mgr else []
        loop = asyncio.get_running_loop()

        try:
            text_emo = await loop.run_in_executor(
                None, self.container.text_emotion.analyze, accumulated_text
            )
        except Exception:
            text_emo = EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})
        face_emo = self._average_emotion(self._face_emotion_buffer.get(session_id, []))
        voice_emo = self._average_emotion(self._voice_emotion_buffer.get(session_id, []))
        fused_emo = self.container.fusion.fuse(text_emo, voice_emo, face_emo)

        sys_prompt = self._build_dynamic_system_prompt(
            step_mgr, None, history_mgr, detected_emotion=fused_emo.primary_emotion
        )

        llm_context = LLMContext(
            user_text=accumulated_text,
            system_prompt=sys_prompt,
            fused_emotion=fused_emo.primary_emotion,
            history=history,
            max_new_tokens=120,
            max_sentences=3,
        )

        response = await loop.run_in_executor(
            None, self.container.llm.generate_response, llm_context
        )

        if history_mgr:
            history_mgr.add_user_message(accumulated_text)
            history_mgr.add_assistant_message(response.reply_text)

        # _turn_state 유지 (awaiting_transition)
        self._clear_turn_buffers(session_id)

        current_state = self._turn_state.get(session_id, "normal")
        return {
            "llm_response": response,
            "transition": current_state,
            "step_status": step_mgr.get_status(),
            "next_step_status": None,
        }

    # ══════════════════════════════════════════════════════════════
    # 스텝 전환 실행 (내담자 전환 승인 시)
    # ══════════════════════════════════════════════════════════════

    async def execute_step_transition(self, session_id: str) -> Optional[dict]:
        step_mgr = self.session.get_step_manager(session_id)
        history_mgr = self.session.get_history_manager(session_id)
        loop = asyncio.get_running_loop()

        pre_status = step_mgr.get_status()
        transition = step_mgr.advance_question()

        if transition == "step_changed":
            if history_mgr:
                await loop.run_in_executor(
                    None,
                    history_mgr.on_step_transition,
                    pre_status["step"],
                    pre_status["title"],
                )
            logger.info(
                f"[StepMgr] {session_id}: → Step {step_mgr.step_number} "
                f"'{step_mgr.current_step['name']}' 시작"
            )

        self._turn_state[session_id] = "normal"
        self._clear_turn_buffers(session_id)

        return {
            "llm_response": LLMResponse(reply_text=""),
            "transition": "step_changed",
            "step_status": pre_status,
            "next_step_status": step_mgr.get_status(),
        }

    # ══════════════════════════════════════════════════════════════
    # 상담 완료 실행 (5단계 종료 승인 시)
    # ══════════════════════════════════════════════════════════════

    async def execute_counseling_complete(self, session_id: str) -> Optional[dict]:
        step_mgr = self.session.get_step_manager(session_id)
        history_mgr = self.session.get_history_manager(session_id)
        loop = asyncio.get_running_loop()

        pre_status = step_mgr.get_status()
        step_mgr.advance_question()  # → counseling_complete

        if history_mgr:
            await loop.run_in_executor(
                None,
                history_mgr.on_step_transition,
                pre_status["step"],
                pre_status["title"],
            )

        self._turn_state[session_id] = "complete"
        self._clear_turn_buffers(session_id)
        logger.info(f"[StepMgr] {session_id}: 전체 상담 완료")

        return {
            "llm_response": LLMResponse(reply_text=""),
            "transition": "counseling_complete",
            "step_status": pre_status,
            "next_step_status": step_mgr.get_status(),
        }


# 전역 인스턴스 (session_manager에서 import해서 사용)
from app.core.container import ai_container
pipeline = CounselingPipeline(ai_container)
