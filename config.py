from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    telegram_chat_id: str
    admin_chat_id: str
    openrouter_api_key: str | None
    database_path: Path
    log_file: Path
    log_max_bytes: int
    queue_max_size: int
    publish_delay_seconds: int
    publish_jitter_min_seconds: float
    publish_jitter_max_seconds: float
    cycle_sleep_seconds: int


def _must_getenv(name: str) -> str:
    value = os.getenv(name, '').strip()
    if not value:
        raise RuntimeError(f'Не задана переменная окружения {name}')
    return value


SETTINGS = Settings(
    telegram_token=_must_getenv('TELEGRAM_TOKEN'),
    telegram_chat_id=_must_getenv('TELEGRAM_CHAT_ID'),
    admin_chat_id=_must_getenv('ADMIN_CHAT_ID'),
    openrouter_api_key=os.getenv('OPENROUTER_API_KEY', '').strip() or None,
    database_path=BASE_DIR / 'published_articles.db',
    log_file=BASE_DIR / 'log_parser.txt',
    log_max_bytes=int(os.getenv('LOG_MAX_BYTES', 5 * 1024 * 1024)),
    queue_max_size=int(os.getenv('MAX_QUEUE', 100)),
    publish_delay_seconds=int(os.getenv('PUBLISH_DELAY_SECONDS', 600)),
    publish_jitter_min_seconds=float(os.getenv('PUBLISH_JITTER_MIN_SECONDS', 2)),
    publish_jitter_max_seconds=float(os.getenv('PUBLISH_JITTER_MAX_SECONDS', 5)),
    cycle_sleep_seconds=int(os.getenv('CYCLE_SLEEP_SECONDS', 10800)),
)
