"""
ReportInsights — GPT-4o-mini로 상담 리포트의 4개 필드 생성.
  summary, strengths, actionItems, keywords

단일 API 호출로 4필드 모두 JSON 형식 응답 받음.
실패 시 4필드 모두 None 반환 — 본 서비스는 계속 동작.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


INSIGHT_PROMPT = """\
당신은 CBT 상담 리포트 작성 전문가입니다. 아래 상담 정보를 바탕으로
한국어 리포트의 4개 필드를 JSON으로 생성하세요.

[내담자 정보]
주제: {topic}
시작 시점 기분: {mood}
호소 내용: {content}

[단계별 대화 요약]
{step_summaries_text}

[감정 흐름]
{emotion_flow_text}

[작성 지시]
1. summary: 상담 전체를 2~3문장으로 요약. 어떤 문제가 다뤄졌고 어떻게 진행되었는지.
2. strengths: 내담자가 보여준 강점/긍정적 태도 1~2문장.
3. actionItems: 실천 가능한 행동 항목 1~2문장.
4. keywords: 핵심 키워드 4~6개를 쉼표로 구분 (예: "번아웃,업무 스트레스,휴식")

[출력 형식 - 반드시 아래 JSON만 출력]
{{
  "summary": "...",
  "strengths": "...",
  "actionItems": "...",
  "keywords": "..."
}}
"""


def _format_summaries(step_summaries: Dict[int, Dict[str, Any]]) -> str:
    if not step_summaries:
        return "(없음)"
    lines = []
    for step_num in sorted(step_summaries.keys()):
        info = step_summaries[step_num]
        lines.append(f"- Step {step_num} ({info.get('step_name', '')}): {info.get('summary', '')}")
    return "\n".join(lines)


def _format_emotion_flow(turn_emotions: List[Dict[str, Any]]) -> str:
    if not turn_emotions:
        return "(없음)"
    flow = " → ".join(e.get("fused_emotion", "?") for e in turn_emotions)
    return flow


def _generate_sync(topic: str, mood: str, content: str,
                   step_summaries: Dict[int, Dict[str, Any]],
                   turn_emotions: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not settings.openai_api_key:
        logger.warning("[Insights] OPENAI_API_KEY 미설정 → 4필드 모두 None")
        return None

    prompt = INSIGHT_PROMPT.format(
        topic=topic or "(미지정)",
        mood=mood or "(미지정)",
        content=content or "(미지정)",
        step_summaries_text=_format_summaries(step_summaries),
        emotion_flow_text=_format_emotion_flow(turn_emotions),
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a CBT counseling report writer. Output only valid JSON in Korean."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        # 필드 검증
        required = ("summary", "strengths", "actionItems", "keywords")
        if not all(k in data for k in required):
            logger.warning(f"[Insights] 필수 필드 누락: {list(data.keys())}")
            return None
        return {k: str(data[k]) for k in required}
    except Exception:
        logger.exception("[Insights] GPT 호출 또는 파싱 실패")
        return None


async def generate_insights(
    topic: str,
    mood: str,
    content: str,
    step_summaries: Dict[int, Dict[str, Any]],
    turn_emotions: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    """
    Returns:
        {"summary": ..., "strengths": ..., "actionItems": ..., "keywords": ...}
        실패 시 모두 None.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, _generate_sync, topic, mood, content, step_summaries, turn_emotions
    )
    if result is None:
        return {"summary": None, "strengths": None, "actionItems": None, "keywords": None}
    logger.info(f"[Insights] 4필드 생성 완료 (keywords={result['keywords'][:50]}...)")
    return result
