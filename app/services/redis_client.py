"""
Redis 클라이언트 — ticket_id → userId 조회용.

키 포맷: `ticket:{ticket_id}`, 값은 userId 문자열.
스프링 CounselingService.startSession()에서 동일 포맷으로 저장 (TTL 3분).
"""

import logging
from typing import Optional

import redis.asyncio as redis_asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)

_redis_client: Optional[redis_asyncio.Redis] = None


def _get_client() -> redis_asyncio.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_asyncio.from_url(
            settings.redis_url, decode_responses=True
        )
        logger.info(f"[Redis] 클라이언트 생성: {settings.redis_url}")
    return _redis_client


def _user_id_key(ticket_id: str) -> str:
    return f"ticket:{ticket_id}"


async def get_user_id(ticket_id: str) -> Optional[int]:
    """
    ticket_id 로 Redis에서 userId 조회.
    Returns: userId(int) 또는 None (키 없음/Redis 에러).
    """
    try:
        client = _get_client()
        raw = await client.get(_user_id_key(ticket_id))
        if raw is None:
            logger.warning(f"[Redis] ticket_id={ticket_id} userId 없음")
            return None
        return int(raw)
    except Exception:
        logger.exception(f"[Redis] ticket_id={ticket_id} 조회 중 예외")
        return None


async def close() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
