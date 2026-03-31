from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


LOGGER_NAME = 'autonews_bot'


def setup_logging(log_file: Path, max_bytes: int) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=3,
        encoding='utf-8',
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def json_log(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {'event': event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False))
