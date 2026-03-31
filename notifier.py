from __future__ import annotations

import asyncio
import hashlib
import logging
import random

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError, TelegramRetryAfter

from logging_utils import json_log


class TelegramNotifier:
    def __init__(
        self,
        bot: Bot,
        logger: logging.Logger,
        chat_id: str,
        admin_chat_id: str,
        jitter_min: float,
        jitter_max: float,
    ):
        self.bot = bot
        self.logger = logger
        self.chat_id = chat_id
        self.admin_chat_id = admin_chat_id
        self.jitter_min = jitter_min
        self.jitter_max = jitter_max
        self.sent_error_hashes: set[str] = set()

    async def startup_message(self) -> None:
        await self.bot.send_message(self.admin_chat_id, '🚀 Парсер автоновостей запущен и начал работу.')
        json_log(self.logger, 'startup_notification_sent', admin_chat_id=self.admin_chat_id)

    async def notify_admin(self, text: str) -> None:
        error_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if error_hash in self.sent_error_hashes:
            json_log(self.logger, 'admin_notification_skipped_duplicate')
            return
        self.sent_error_hashes.add(error_hash)
        try:
            await self.bot.send_message(self.admin_chat_id, text)
            json_log(self.logger, 'admin_notification_sent')
        except Exception as exc:
            json_log(self.logger, 'admin_notification_error', error=str(exc))

    async def send_post(self, title: str, lead: str, image_url: str, link: str, source: str) -> None:
        caption = f'<b>{title}</b>\n\n{lead}\n\nИсточник: <a href="{link}">{source}</a>'
        caption = caption[:1024]
        await asyncio.sleep(random.uniform(self.jitter_min, self.jitter_max))

        try:
            if image_url and image_url.startswith('http'):
                await self.bot.send_photo(self.chat_id, image_url, caption=caption, parse_mode='HTML')
                json_log(self.logger, 'telegram_post_sent', mode='photo', title=title, source=source)
            else:
                await self.bot.send_message(self.chat_id, caption, parse_mode='HTML', disable_web_page_preview=False)
                json_log(self.logger, 'telegram_post_sent', mode='text', title=title, source=source)
        except TelegramRetryAfter as exc:
            json_log(self.logger, 'telegram_retry_after', timeout=exc.retry_after)
            await asyncio.sleep(exc.retry_after)
            raise
        except (TelegramBadRequest, TelegramNetworkError) as exc:
            json_log(self.logger, 'telegram_send_error', error=str(exc), title=title, source=source)
            raise
