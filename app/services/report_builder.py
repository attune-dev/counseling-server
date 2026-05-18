"""
ReportBuilder — 상담 종료 시점에 Spring 백엔드로 보낼 리포트 페이로드를 빌드.

현재 가진 데이터로 채울 수 있는 필드만 채우고, GPT 생성이 필요한 필드
(summary, strengths, actionItems, keywords)는 None으로 비워둠.
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# AI 서버 감정(소문자) → Spring EmotionType enum (대문자)
def _to_emotion_enum(emotion: Optional[str]) -> Optional[str]:
    if not emotion:
        return None
    return emotion.upper()


# 단계 번호 (1~5) → Spring CounselingStage enum
_STAGE_ENUM_MAP = {
    1: "RAPPORT_BUILDING",
    2: "PROBLEM_EXPLORATION",
    3: "COGNITIVE_SHIFT",
    4: "ACTION_PLANNING",
    5: "CLOSING",
}


def _to_stage_enum(step: Optional[int]) -> Optional[str]:
    if step is None:
        return None
    return _STAGE_ENUM_MAP.get(step)


def _compress_consecutive(items: List[str]) -> List[str]:
    """연속된 중복 제거. 예: [SAD, FEAR, FEAR, SAD] → [SAD, FEAR, SAD]"""
    result: List[str] = []
    for item in items:
        if not result or result[-1] != item:
            result.append(item)
    return result


def _filter_emotions_by_step(turn_emotions: List[Dict[str, Any]], step: int) -> List[str]:
    return [_to_emotion_enum(e["fused_emotion"]) for e in turn_emotions if e.get("step") == step]


def _dominant(emotions: List[str]) -> Optional[str]:
    if not emotions:
        return None
    return Counter(emotions).most_common(1)[0][0]


def build_report(
    session_id: str,
    user_id: Optional[int],
    setup,                # CounselingSetup
    started_at,           # datetime
    ended_at,             # datetime
    turn_emotions: List[Dict[str, Any]],
    step_summaries: Dict[int, Dict[str, Any]],
    insights: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """
    Spring `ReportSaveRequest` 모양에 맞춘 페이로드 생성.

    insights: GPT-4o-mini가 생성한 4필드 (summary/strengths/actionItems/keywords).
              None 이거나 실패 시 None 값으로 비워서 보냄.
    """
    insights = insights or {}
    fused_list = [_to_emotion_enum(e["fused_emotion"]) for e in turn_emotions]

    # 단계별 상세 — emotionFlow는 연속 중복 제거된 형태
    stage_details: List[Dict[str, Any]] = []
    for step_num in sorted(step_summaries.keys()):
        info = step_summaries[step_num]
        raw_flow = _filter_emotions_by_step(turn_emotions, step_num)
        stage_details.append({
            "step": step_num,
            "content": info.get("summary", ""),
            "emotionFlow": _compress_consecutive(raw_flow),
        })

    # 턴별 감정 로그 — 매 턴 그대로 (압축 X)
    emotion_logs: List[Dict[str, Any]] = [
        {
            "stage": _to_stage_enum(e["step"]),
            "emotion": _to_emotion_enum(e["fused_emotion"]),
        }
        for e in turn_emotions
    ]

    payload: Dict[str, Any] = {
        # 세션 정보
        "userId": user_id,                                       # TODO: setup 데이터에 userId 추가 후 전달
        "counselingType": "AI 상담",
        "topic": setup.topic if setup else None,
        "startedAt": started_at.isoformat(timespec="seconds") if started_at else None,
        "endedAt": ended_at.isoformat(timespec="seconds") if ended_at else None,

        # 리포트 요약 정보
        "totalTurnCount": len(turn_emotions),
        "primaryEmotion": _dominant(fused_list),
        "initialEmotion": fused_list[0] if fused_list else None,
        "finalEmotion": fused_list[-1] if fused_list else None,
        "summary": insights.get("summary"),
        "strengths": insights.get("strengths"),
        "actionItems": insights.get("actionItems"),
        "keywords": insights.get("keywords"),

        # 상세
        "stageDetails": stage_details,
        "emotionLogs": emotion_logs,
    }

    logger.info(
        f"[Report] {session_id} 페이로드 빌드 완료 "
        f"(턴 {len(turn_emotions)}, 단계 {len(stage_details)}, 감정로그 {len(emotion_logs)})"
    )
    return payload
