from __future__ import annotations

import logging
import time
from playwright.sync_api import Page

from logging_utils import json_log
from parsers.common import safe_decode, with_retry


def parse_kolesa_ru(page: Page, logger: logging.Logger) -> list[str]:
    json_log(logger, 'site_start', site='kolesa.ru')
    try:
        def _work() -> list[str]:
            page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'Accept-Language': 'ru-RU,ru;q=0.9',
            })
            page.goto('https://www.kolesa.ru/news', timeout=40000, wait_until='domcontentloaded')
            time.sleep(3)
            return page.evaluate("""() => {
                const links = new Set();
                const xpathResult = document.evaluate(
                    "//a[contains(@href, '/news/') and not(contains(@href, '/news/archive/'))]",
                    document,
                    null,
                    XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
                    null
                );
                for (let i = 0; i < Math.min(xpathResult.snapshotLength, 15); i++) {
                    const link = xpathResult.snapshotItem(i);
                    if (link.textContent.trim().length > 10) {
                        links.add(link.href.startsWith('http') ? link.href : 'https://www.kolesa.ru' + link.href);
                    }
                }
                return Array.from(links);
            }""")[:5]
        links = with_retry(_work)
        json_log(logger, 'site_links_found', site='kolesa.ru', count=len(links))
        return links
    except Exception as exc:
        json_log(logger, 'site_error', site='kolesa.ru', error=str(exc))
        return []


def parse_kolesa_article(page: Page, url: str, logger: logging.Logger) -> tuple[str | None, str | None, str | None]:
    try:
        def _work() -> tuple[str | None, str | None, str | None]:
            page.goto(url, timeout=30000, wait_until='domcontentloaded')
            time.sleep(3)
            title = page.locator('h1').first.inner_text(timeout=5000)
            lead = ''
            paragraphs = page.locator('p')
            for i in range(paragraphs.count()):
                try:
                    text = paragraphs.nth(i).inner_text(timeout=1000).strip()
                    if len(text) > 30:
                        lead = text
                        break
                except Exception:
                    continue
            image_url = page.locator("meta[property='og:image']").get_attribute('content')
            if not image_url:
                image_url = page.locator('img').first.get_attribute('src')
            return safe_decode(title), safe_decode(lead), image_url
        return with_retry(_work)
    except Exception as exc:
        json_log(logger, 'article_error', site='kolesa.ru', url=url, error=str(exc))
        return None, None, None


def parse_autostat_ru(page: Page, logger: logging.Logger) -> list[str]:
    json_log(logger, 'site_start', site='autostat.ru')
    try:
        def _work() -> list[str]:
            page.goto('https://www.autostat.ru/news/', timeout=60000, wait_until='domcontentloaded')
            time.sleep(5)
            return page.evaluate("""() => {
                const found = [];
                document.querySelectorAll('a[href^="/news/"]').forEach(a => {
                    const href = a.getAttribute('href');
                    if (/^\/news\/\d+\/$/.test(href)) {
                        found.push('https://www.autostat.ru' + href);
                    }
                });
                return found.slice(0, 5);
            }""")
        links = with_retry(_work)
        json_log(logger, 'site_links_found', site='autostat.ru', count=len(links))
        return links
    except Exception as exc:
        json_log(logger, 'site_error', site='autostat.ru', error=str(exc))
        return []


def parse_autostat_article(page: Page, url: str, logger: logging.Logger) -> tuple[str | None, str | None, str | None]:
    try:
        def _work() -> tuple[str | None, str | None, str | None]:
            page.goto(url, timeout=30000, wait_until='domcontentloaded')
            time.sleep(3)
            title = page.locator('h1').first.inner_text(timeout=5000)
            lead = ''
            paragraphs = page.locator('p')
            for i in range(paragraphs.count()):
                try:
                    text = paragraphs.nth(i).inner_text(timeout=1000).strip()
                    if 50 < len(text) < 500 and all(x not in text.lower() for x in ['e-mail', 'регистрация', 'нажмите', 'подпис', 'источник']):
                        lead = text
                        break
                except Exception:
                    continue
            image_url = page.locator("meta[property='og:image']").get_attribute('content')
            if not image_url:
                image_url = page.locator('img').first.get_attribute('src')
            return safe_decode(title), safe_decode(lead), image_url
        return with_retry(_work)
    except Exception as exc:
        json_log(logger, 'article_error', site='autostat.ru', url=url, error=str(exc))
        return None, None, None


def parse_avtonovostidnya_ru(page: Page, logger: logging.Logger) -> list[str]:
    json_log(logger, 'site_start', site='avtonovostidnya.ru')
    try:
        def _work() -> list[str]:
            page.goto('https://avtonovostidnya.ru/', timeout=30000, wait_until='domcontentloaded')
            time.sleep(3)
            return page.evaluate("""() => {
                const items = Array.from(document.querySelectorAll('article'));
                return items.map(item => {
                    const link = item.querySelector('h2 a, h3 a');
                    return link ? link.href : null;
                }).filter(href => href && href.includes('avtonovostidnya.ru')).slice(0, 5);
            }""")
        links = with_retry(_work)
        json_log(logger, 'site_links_found', site='avtonovostidnya.ru', count=len(links))
        return links
    except Exception as exc:
        json_log(logger, 'site_error', site='avtonovostidnya.ru', error=str(exc))
        return []


def parse_avtonovostidnya_article(page: Page, url: str, logger: logging.Logger) -> tuple[str | None, str | None, str | None]:
    try:
        def _work() -> tuple[str | None, str | None, str | None]:
            page.goto(url, timeout=30000, wait_until='domcontentloaded')
            time.sleep(3)
            title = page.evaluate("""() => {
                let t = document.querySelector('h1') || document.querySelector('.entry-title');
                return t ? t.innerText.trim() : '';
            }""")
            lead = page.evaluate("""() => {
                let ps = Array.from(document.querySelectorAll('.entry-content p, .post-content p, article p'));
                for (const p of ps) {
                    let text = p.innerText.trim();
                    if (text.length > 30) return text;
                }
                return '';
            }""")
            image_url = page.evaluate("""() => {
                let img = document.querySelector('meta[property="og:image"]');
                if (img && img.content) return img.content;
                img = document.querySelector('.wp-post-image, img, article img');
                return img && img.src ? img.src : '';
            }""")
            return safe_decode(title), safe_decode(lead), image_url
        return with_retry(_work)
    except Exception as exc:
        json_log(logger, 'article_error', site='avtonovostidnya.ru', url=url, error=str(exc))
        return None, None, None


def parse_auto_ru(page: Page, logger: logging.Logger) -> list[str]:
    json_log(logger, 'site_start', site='auto.ru')
    try:
        def _work() -> list[str]:
            page.set_extra_http_headers({
                'Accept-Language': 'ru-RU,ru;q=0.9',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
            })
            page.goto('https://auto.ru/mag/theme/news/', timeout=60000, wait_until='domcontentloaded')
            time.sleep(2)
            try:
                page.click('#confirm-button', timeout=5000)
                json_log(logger, 'auto_ru_confirm_clicked')
                time.sleep(1)
            except Exception:
                pass
            for _ in range(10):
                page.mouse.wheel(0, 2200)
                time.sleep(1.0)

            html = page.content().lower()
            if 'доступ временно ограничен' in html or 'робот' in html or 'captcha' in html:
                json_log(logger, 'auto_ru_block_detected')

            strategies = [
                """() => Array.from(document.querySelectorAll('a'))
                    .map(a => a.href)
                    .filter(h => h.includes('/mag/article/') && !h.includes('video') && !h.includes('gallery'))""",
                """() => Array.from(document.querySelectorAll('a[href*="/mag/article/"]'))
                    .map(a => a.href)""",
                """() => Array.from(document.querySelectorAll('[data-testid] a, article a, main a'))
                    .map(a => a.href)
                    .filter(Boolean)
                    .filter(h => h.includes('/mag/article/'))""",
            ]
            found: list[str] = []
            for i, script in enumerate(strategies, start=1):
                try:
                    current = page.evaluate(script)
                    json_log(logger, 'auto_ru_strategy_result', strategy=i, count=len(current))
                    found.extend(current)
                except Exception as exc:
                    json_log(logger, 'auto_ru_strategy_error', strategy=i, error=str(exc))
            links = []
            seen = set()
            for link in found:
                if not link or '/mag/article/' not in link:
                    continue
                if 'video' in link or 'gallery' in link:
                    continue
                if link in seen:
                    continue
                seen.add(link)
                links.append(link)
            return links[:5]
        links = with_retry(_work)
        json_log(logger, 'site_links_found', site='auto.ru', count=len(links))
        return links
    except Exception as exc:
        json_log(logger, 'site_error', site='auto.ru', error=str(exc))
        return []


def parse_auto_article(page, url, logger):
    try:
        page.goto(url, timeout=60000, wait_until='domcontentloaded')

        # ждем заголовок
        page.wait_for_selector("h1", timeout=10000)

        title = page.locator("h1").first.inner_text()

        if title and "главное за день" in title.lower():
            return None, None, None

        # получаем первый нормальный абзац
        paragraphs = page.locator("p")
        lead = ""

        for i in range(paragraphs.count()):
            try:
                text = paragraphs.nth(i).inner_text(timeout=1000).strip()

                if (
                    len(text) > 100
                    and "фото" not in text.lower()
                    and "источник" not in text.lower()
                    and not text.startswith("Читайте")
                ):
                    lead = text
                    break
            except:
                continue

        # картинка
        image = page.locator("meta[property='og:image']")
        image_url = image.get_attribute("content") if image else ""

        return title, lead, image_url

    except Exception as e:
        log(f"❌ auto.ru ошибка: {e}")
        return None, None, None