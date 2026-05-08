"""
실제 AI 모델 구현체
- TextEmotionModel      : klue/bert 기반 텍스트 감정 분류 (로컬 models/text-emotion-final)
- Wav2VecEmotionModel   : wav2vec2 기반 음성 감정 분류 (로컬 models/voice-emotion-final)
- EmotionFusionModel    : 엔트로피 기반 동적 가중치 감정 퓨전
- ExaoneLLMModel        : EXAONE-3.5-7.8B-Instruct (8-bit) CBT 상담 LLM
"""

import re
import logging
from typing import List, Optional

import numpy as np

from ai_modules.interfaces import (
    BaseTextEmotionModel,
    BaseEmotionModel,
    BaseEmotionFusionModel,
    BaseLLMModel,
)
from ai_modules.schemas import EmotionResult, STTInput, LLMContext, LLMResponse

logger = logging.getLogger(__name__)

# 7개 감정 레이블 (알파벳 순 인덱스)
EMOTION_LABEL_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}


# ──────────────────────────────────────────────────────────────
# 1. 텍스트 감정 분석 (BertForSequenceClassification, klue/bert)
#    입력: STT 결과 텍스트 (str)
#    출력: EmotionResult
# ──────────────────────────────────────────────────────────────
class TextEmotionModel(BaseTextEmotionModel):
    def __init__(self, model_path: str = "models/text-emotion-final", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        logger.info(f"[TextEmo] 로딩 완료: {self.model_path} on {self.device}")

    def analyze(self, text: str) -> EmotionResult:
        import torch

        if not text or not text.strip():
            return EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            pred_idx = int(probs.argmax())
            primary = EMOTION_LABEL_MAP.get(pred_idx, f"label_{pred_idx}")
            prob_dict = {
                EMOTION_LABEL_MAP.get(i, f"label_{i}"): round(float(p), 3)
                for i, p in enumerate(probs)
            }
            return EmotionResult(primary_emotion=primary, probabilities=prob_dict)
        except Exception as e:
            logger.error(f"[TextEmo] 분석 오류: {e}")
            return EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})


# ──────────────────────────────────────────────────────────────
# 2. 음성 감정 분석 (Wav2Vec2ForSequenceClassification)
#    입력: STTInput.audio_data = float32 PCM bytes (16kHz, mono)
#    출력: EmotionResult
# ──────────────────────────────────────────────────────────────
class Wav2VecEmotionModel(BaseEmotionModel):
    def __init__(self, model_path: str = "models/voice-emotion-final", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.feature_extractor = None

    def load_model(self):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_path
        ).to(self.device)
        self.model.eval()
        logger.info(f"[VoiceEmo] Wav2Vec2 로딩 완료: {self.model_path} on {self.device}")

    def analyze(self, input_data: STTInput) -> EmotionResult:
        import torch

        try:
            audio_array = np.frombuffer(input_data.audio_data, dtype=np.float32)
            if len(audio_array) < 1600:  # 최소 0.1초
                return EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})

            inputs = self.feature_extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            pred_idx = int(probs.argmax())
            primary = EMOTION_LABEL_MAP.get(pred_idx, f"label_{pred_idx}")
            prob_dict = {
                EMOTION_LABEL_MAP.get(i, f"label_{i}"): round(float(p), 3)
                for i, p in enumerate(probs)
            }
            return EmotionResult(primary_emotion=primary, probabilities=prob_dict)
        except Exception as e:
            logger.error(f"[VoiceEmo] 분석 오류: {e}")
            return EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})


# ──────────────────────────────────────────────────────────────
# 3. 감정 융합 (엔트로피 기반 동적 가중치)
# ──────────────────────────────────────────────────────────────
class EmotionFusionModel(BaseEmotionFusionModel):
    """
    엔트로피 기반 동적 가중치 감정 퓨전.

    원리:
      1. 각 모달리티의 softmax 확률 분포에서 엔트로피(H)를 계산
      2. 엔트로피의 역수(1/H)를 confidence로 사용
      3. confidence를 정규화하여 가중치로 사용
      → 확신이 높은(엔트로피가 낮은) 모달리티에 자동으로 큰 가중치 부여

    참고: AGFN (Adaptive Gated Fusion Network, arXiv 2025)의
          Information Entropy Gate에서 착안.
    """

    @staticmethod
    def _entropy(probabilities: dict) -> float:
        """확률 분포의 엔트로피: H = -sum(p * ln(p))"""
        import math
        h = 0.0
        for p in probabilities.values():
            if p > 1e-10:
                h -= p * math.log(p)
        return h

    def fuse(
        self,
        text_result: EmotionResult,
        voice_result: EmotionResult,
        face_result: EmotionResult,
    ) -> EmotionResult:
        from collections import defaultdict

        # Step 1: 각 모달리티의 엔트로피 계산
        h_text = self._entropy(text_result.probabilities)
        h_voice = self._entropy(voice_result.probabilities)
        h_face = self._entropy(face_result.probabilities)

        # Step 2: 역수 → confidence (엔트로피 낮을수록 확신 높음)
        eps = 1e-8
        conf_text = 1.0 / (h_text + eps)
        conf_voice = 1.0 / (h_voice + eps)
        conf_face = 1.0 / (h_face + eps)

        # Step 3: 정규화 → 가중치 (합 = 1)
        total_conf = conf_text + conf_voice + conf_face
        w_text = conf_text / total_conf
        w_voice = conf_voice / total_conf
        w_face = conf_face / total_conf

        logger.info(
            f"[Fusion] 엔트로피: text={h_text:.3f} voice={h_voice:.3f} face={h_face:.3f} "
            f"→ 가중치: text={w_text:.1%} voice={w_voice:.1%} face={w_face:.1%}"
        )

        # Step 4: 가중합
        combined: dict = defaultdict(float)
        for emotion, prob in text_result.probabilities.items():
            combined[emotion] += prob * w_text
        for emotion, prob in voice_result.probabilities.items():
            combined[emotion] += prob * w_voice
        for emotion, prob in face_result.probabilities.items():
            combined[emotion] += prob * w_face

        total = sum(combined.values()) or 1.0
        prob_dict = {k: round(v / total, 3) for k, v in combined.items()}
        primary = max(prob_dict, key=prob_dict.get)
        return EmotionResult(primary_emotion=primary, probabilities=prob_dict)


# ──────────────────────────────────────────────────────────────
# 4. EXAONE 3.5 7.8B (8-bit) CBT 상담 LLM
#    LoRA 없음 — CBT 제어 + 감정 톤 조절은 전적으로 system prompt로.
# ──────────────────────────────────────────────────────────────
class ExaoneLLMModel(BaseLLMModel):
    """
    EXAONE 3.5 7.8B (8-bit) CBT 상담 LLM.
    LoRA 없음 — CBT 제어 + 감정 톤 조절은 전적으로 system prompt로.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("[EXAONE] CUDA 사용 불가 → CPU 모드 (매우 느림)")
            self.device = "cpu"

        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024 ** 3
            logger.info(
                f"[EXAONE] GPU: {torch.cuda.get_device_name(0)}, "
                f"Compute {props.major}.{props.minor}, VRAM {vram_gb:.1f}GB"
            )

        logger.info(f"[EXAONE] {self.model_name} 로딩 중 (8-bit, device={self.device})...")

        # trust_remote_code 필수 — EXAONE은 커스텀 모델링 코드 포함
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            import torch as _torch
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=_torch.float32,
                trust_remote_code=True,
            )

        self.model.eval()

        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
            logger.info(f"[EXAONE] 로딩 완료. VRAM: 할당 {allocated:.2f}GB / 예약 {reserved:.2f}GB")
        else:
            logger.info("[EXAONE] CPU 모드 로딩 완료")

    def generate_response(self, context: LLMContext) -> LLMResponse:
        import torch

        # system prompt는 pipeline이 동적으로 조립하여 context.system_prompt에 전달
        system_prompt = context.system_prompt or (
            "따뜻하고 공감적인 CBT 전문 상담사 '멍박사님'입니다. "
            "반드시 한국어로만 답변하세요."
        )

        # messages 조립
        messages = [{"role": "system", "content": system_prompt}]
        for h in context.history:
            messages.append(h)
        messages.append({"role": "user", "content": context.user_text})

        # messages 유효성 검사 — None content 방지
        safe_messages = []
        for m in messages:
            content = m.get("content", "") or ""
            safe_messages.append({"role": m["role"], "content": content})

        # apply_chat_template + 방어 코드
        text = self.tokenizer.apply_chat_template(
            safe_messages, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(text, str):
            logger.warning(f"[EXAONE] apply_chat_template 비정상 반환: {type(text)}")
            while isinstance(text, list) and text:
                text = text[0]
            if not isinstance(text, str):
                text = str(text) if text else ""
        if not text.strip():
            logger.warning("[EXAONE] 빈 텍스트 감지 → 폴백")
            text = "안녕하세요"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # max_new_tokens: pipeline에서 context에 담아 전달
        max_tokens = context.max_new_tokens

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.1,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 후처리
        reply = self._clean_response(raw_reply)
        reply = self._limit_sentences(reply, context.max_sentences)

        return LLMResponse(reply_text=reply)

    @staticmethod
    def _clean_response(text: str) -> str:
        """EXAONE 응답 후처리 — 영어/라벨/당신/이모지 등 제거"""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            total = max(len(line), 1)
            eng_chars = len([c for c in line if c.isascii() and c.isalpha()])
            if eng_chars / total > 0.4:
                continue
            cleaned_lines.append(line)
        text = ' '.join(cleaned_lines)
        text = re.sub(r'\b_\w+_?\b', '', text)
        text = re.sub(r'[\U0001F300-\U0001F9FF\u2600-\u26FF\u2700-\u27BF\u200d\ufe0f]', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*', '', text)
        text = re.sub(r'[\[\]{}<>~#^`]', '', text)
        text = re.sub(
            r'(마무리\s?핵심\s?통찰|핵심\s?통찰|핵심\s?요약|다음\s?주까지의?\s?과제|과제|격려|요약)\s*[:：]\s*',
            '', text
        )
        text = re.sub(r'당신만의', '본인만의', text)
        text = re.sub(r'당신[은의이가에를도]?\s?', '', text)
        text = re.sub(r'\b[A-Za-z]{3,}\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _limit_sentences(text: str, max_n: int = 3) -> str:
        """문장 수 제한"""
        endings = list(re.finditer(r'[.?!]+(?:\s|$)', text))
        if len(endings) <= max_n:
            return text
        cut_pos = endings[max_n - 1].end()
        return text[:cut_pos].strip()
