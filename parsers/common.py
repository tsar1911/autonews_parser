from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar('T')


def safe_decode(text: str | None) -> str | None:
    if not text:
        return text
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, AttributeError):
        return text


def with_retry(func: Callable[[], T], retries: int = 3, delay_seconds: int = 2) -> T:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay_seconds)
    if last_error is None:
        raise RuntimeError('Неизвестная ошибка retry')
    raise last_error
