"""
SpringClient — 상담 종료 리포트를 Spring 백엔드에 HTTP POST.

내부망 통신 가정 (같은 EC2/VPC). 실패해도 본 서비스는 계속 동작해야 하므로
예외는 잡고 로그만 남김.
"""

import logging
from typing import Any, Dict

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


async def send_report(payload: Dict[str, Any]) -> bool:
    """
    Spring `/internal/counseling/report` 로 리포트 POST.
    Returns: 성공 True / 실패 False (예외는 전파하지 않음).
    """
    url = settings.spring_backend_url.rstrip("/") + settings.spring_report_endpoint
    session_label = payload.get("userId") or "unknown"

    headers = {}
    if settings.spring_internal_api_key:
        headers["X-Internal-API-Key"] = settings.spring_internal_api_key
    else:
        logger.warning("[Spring] SPRING_INTERNAL_API_KEY 미설정 — 인증 없이 호출")

    try:
        async with httpx.AsyncClient(timeout=settings.spring_request_timeout) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.is_success:
            logger.info(
                f"[Spring] 리포트 전송 성공 (user={session_label}, "
                f"status={response.status_code})"
            )
            return True

        logger.warning(
            f"[Spring] 리포트 전송 실패 (user={session_label}, "
            f"status={response.status_code}, body={response.text[:200]})"
        )
        return False

    except Exception:
        logger.exception(f"[Spring] 리포트 전송 중 예외 (user={session_label})")
        return False
