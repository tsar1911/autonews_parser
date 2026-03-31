from __future__ import annotations

import asyncio
from typing import Callable

from aiogram import Bot
from playwright.sync_api import sync_playwright

from config import SETTINGS
from dedup import DuplicateDetector
from logging_utils import json_log, setup_logging
from notifier import TelegramNotifier
from parsers.sites import (
    parse_auto_article,
    parse_auto_ru,
    parse_autostat_article,
    parse_autostat_ru,
    parse_avtonovostidnya_article,
    parse_avtonovostidnya_ru,
    parse_kolesa_article,
    parse_kolesa_ru,
)
from queue_manager import LimitedPostQueue, PostItem
from storage import PublishedStorage


logger = setup_logging(SETTINGS.log_file, SETTINGS.log_max_bytes)
storage = PublishedStorage(SETTINGS.database_path)
detector = DuplicateDetector(logger, SETTINGS.openrouter_api_key)
queue = LimitedPostQueue(SETTINGS.queue_max_size)


async def publish_loop(notifier: TelegramNotifier) -> None:
    while True:
        item = queue.pop()
        if item is None:
            await asyncio.sleep(5)
            continue
        try:
            await notifier.send_post(item.title, item.lead, item.image_url, item.link, item.source)
        except Exception as exc:
            json_log(logger, 'publish_error', error=str(exc), title=item.title, source=item.source)
            await notifier.notify_admin(f'❌ Ошибка публикации\n\n{exc}\n\n{item.title}')
            queue.push(item)
            await asyncio.sleep(60)
            continue
        await asyncio.sleep(SETTINGS.publish_delay_seconds)


def collect_articles() -> list[tuple[str, str, str, str, str]]:
    parsed_articles: list[tuple[str, str, str, str, str]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
            ],
        )
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=1,
            is_mobile=False,
            has_touch=False,
            locale='ru-RU',
            timezone_id='Europe/Moscow',
            color_scheme='light',
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        )
        context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'language', {get: () => 'ru-RU'});
            Object.defineProperty(navigator, 'languages', {get: () => ['ru-RU', 'ru']});
            Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});
            """
        )
        page = context.new_page()

        sources: list[tuple[str, Callable, Callable]] = [
            ('auto.ru', parse_auto_ru, parse_auto_article),
            ('kolesa.ru', parse_kolesa_ru, parse_kolesa_article),
            ('autostat.ru', parse_autostat_ru, parse_autostat_article),
            ('avtonovostidnya.ru', parse_avtonovostidnya_ru, parse_avtonovostidnya_article),
        ]

        for source_name, links_parser, article_parser in sources:
            links = links_parser(page, logger)
            for link in links:
                title, lead, image_url = article_parser(page, link, logger)
                if title and lead:
                    parsed_articles.append((source_name, title, lead, image_url or '', link))

        context.close()
        browser.close()
    return parsed_articles


async def main() -> None:
    published = storage.load_all()
    detector.build_index(published)

    bot = Bot(token=SETTINGS.telegram_token)
    notifier = TelegramNotifier(
        bot=bot,
        logger=logger,
        chat_id=SETTINGS.telegram_chat_id,
        admin_chat_id=SETTINGS.admin_chat_id,
        jitter_min=SETTINGS.publish_jitter_min_seconds,
        jitter_max=SETTINGS.publish_jitter_max_seconds,
    )
    await notifier.startup_message()
    publisher_task = asyncio.create_task(publish_loop(notifier))

    try:
        while True:
            json_log(logger, 'cycle_start')
            parsed_articles = await asyncio.to_thread(collect_articles)
            json_log(logger, 'cycle_articles_collected', count=len(parsed_articles))

            for source, title, lead, image_url, link in parsed_articles:
                if storage.link_exists(link):
                    json_log(logger, 'skip_existing_link', source=source, link=link)
                    continue

                text = f'{title.strip()} {lead.strip()}'
                embedding = detector.encode_text(text)
                current_published = storage.load_all()

                if detector.is_duplicate(title, lead) or detector.llm_check_last_10(text.lower(), current_published):
                    storage.add_article(
                        link=link,
                        title=title,
                        lead=lead,
                        text=text.lower(),
                        embedding=embedding,
                        source=source,
                        is_duplicate=True,
                    )
                    detector.add_embedding(link, text, embedding)
                    json_log(logger, 'skip_duplicate', source=source, link=link)
                    continue

                pushed = queue.push(
                    PostItem(
                        title=title.strip(),
                        lead=lead[:500].strip(),
                        image_url=image_url if image_url.startswith('http') else '',
                        link=link,
                        source=source,
                    )
                )
                if not pushed:
                    json_log(logger, 'queue_full_skip', source=source, link=link, max_size=SETTINGS.queue_max_size)
                    continue

                storage.add_article(
                    link=link,
                    title=title,
                    lead=lead,
                    text=text.lower(),
                    embedding=embedding,
                    source=source,
                    is_duplicate=False,
                )
                detector.add_embedding(link, text, embedding)
                json_log(logger, 'queued_post', source=source, link=link, queue_size=len(queue))

            json_log(logger, 'cycle_complete', sleep_seconds=SETTINGS.cycle_sleep_seconds)
            await asyncio.sleep(SETTINGS.cycle_sleep_seconds)
    finally:
        publisher_task.cancel()
        await bot.session.close()


if __name__ == '__main__':
    asyncio.run(main())
