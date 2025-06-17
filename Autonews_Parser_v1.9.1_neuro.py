import json
import time
import threading
import telebot
import random
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import logging
import sys
import torch

log_file = "log_parser.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def log(msg):
    logging.info(msg)

model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Настройки
TELEGRAM_TOKEN = 
TELEGRAM_CHAT_ID = 
PUBLISHED_FILE = "published_articles.json"

bot = telebot.TeleBot(TELEGRAM_TOKEN)
post_queue = []
last_sent_time = 0

def load_published():
    try:
        with open(PUBLISHED_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_published(data):
    with open(PUBLISHED_FILE, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def is_duplicate(title, lead, published_data, threshold=0.9):
    embedding = model.encode(f"{title.strip()} {lead.strip()}", convert_to_tensor=True, dtype=torch.float32)
    for item in published_data:
        if "embedding" not in item:
            continue
        stored_embedding = torch.tensor(item["embedding"], dtype=torch.float32)
        similarity = util.cos_sim(embedding, stored_embedding)[0][0].item()
        log(f"🔎 Сравнение с {item['link']} — сходство: {similarity:.3f}")
        if similarity >= threshold:
            log(f"⚠️ Похожий заголовок найден: {item['link']} ({similarity:.2f})")
            return True
    return False

def enqueue_post(title, lead, image_url, link, source_name):
    if not title or not lead:
        return
    post = {
        "title": title.strip(),
        "lead": lead[:500].strip(),
        "image_url": image_url if image_url and image_url.startswith('http') else "https://via.placeholder.com/800x400?text=No+Image",
        "link": link,
        "source": source_name
    }
    post_queue.append(post)
    log(f"📝 Добавлено в очередь: {post['title']}")

def publish_loop():
    global last_sent_time
    while True:
        random.shuffle(post_queue)
        if post_queue:
            post = post_queue.pop(0)
            try:
                caption = f"<b>{post['title']}</b>\n\n{post['lead']}\n\nИсточник: <a href=\"{post['link']}\">{post['source']}</a>"
                bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=post['image_url'],
                    caption=caption,
                    parse_mode="HTML"
                )
                log(f"📤 Опубликовано: {post['title']}")
                last_sent_time = time.time()
                time.sleep(300)  # 5 минут между постами
            except Exception as e:
                log(f"❌ Ошибка публикации: {e}")
                post_queue.insert(0, post)
                time.sleep(60)
        else:
            time.sleep(5)

def parse_kolesa_ru(page):
    log("🌐 kolesa.ru/news")
    try:
        page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept-Language': 'ru-RU,ru;q=0.9'
        })
        page.goto("https://www.kolesa.ru/news", timeout=40000, wait_until="domcontentloaded")
        time.sleep(3)
        links = page.evaluate("""() => {
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
        }""")
        log(f"✅ kolesa.ru: {len(links)} ссылок")
        return links[:5]
    except Exception as e:
        log(f"❌ Ошибка kolesa.ru: {e}")
        return []

def parse_kolesa_article(page, url):
    try:
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        time.sleep(3)

        title = page.locator("h1").first.inner_text(timeout=5000)
        lead = ""
        paragraphs = page.locator("p")
        count = paragraphs.count()
        for i in range(count):
            try:
                text = paragraphs.nth(i).inner_text(timeout=1000).strip()
                if len(text) > 30:
                    lead = text
                    break
            except:
                continue

        image_url = page.locator("meta[property='og:image']").get_attribute("content")
        if not image_url:
            image_url = page.locator("img").first.get_attribute("src")

        if not title or not lead:
            log(f"⚠️ kolesa.ru: Нет данных для {url}")
            return None, None, None
        return title.strip(), lead, image_url
    except Exception as e:
        log(f"❌ Ошибка парсинга статьи kolesa.ru: {e}")
        return None, None, None

def parse_autostat_ru(page):
    log("🌐 autostat.ru")
    try:
        page.goto("https://www.autostat.ru/news/", timeout=60000)
        time.sleep(5)
        links = page.evaluate("""() => {
            const found = [];
            document.querySelectorAll('a[href^="/news/"]').forEach(a => {
                const href = a.getAttribute('href');
                if (/^\\/news\\/\\d+\\/$/.test(href)) {
                    found.push('https://www.autostat.ru' + href);
                }
            });
            return found.slice(0, 10);
        }""")
        log(f"✅ autostat.ru: {len(links)} ссылок")
        return links[:5]
    except Exception as e:
        log(f"❌ Ошибка autostat.ru: {str(e)[:200]}")
        return []

def parse_autostat_article(page, url):
    try:
        page.goto(url, timeout=30000)
        time.sleep(3)

        title = page.locator("h1").first.inner_text(timeout=5000)
        lead = ""
        paragraphs = page.locator("p")
        count = paragraphs.count()
        for i in range(count):
            try:
                text = paragraphs.nth(i).inner_text(timeout=1000).strip()
                if (
                    50 < len(text) < 500
                    and "e-mail" not in text.lower()
                    and "регистрация" not in text.lower()
                    and "нажмите" not in text.lower()
                    and "подпис" not in text.lower()
                    and "источник" not in text.lower()
                ):
                    lead = text
                    break
            except:
                continue

        image_url = page.locator("meta[property='og:image']").get_attribute("content")
        if not image_url:
            image_url = page.locator("img").first.get_attribute("src")

        if not title or not lead:
            log(f"⚠️ autostat.ru: Нет валидных данных для {url}")
            return None, None, None
        return title.strip(), lead, image_url
    except Exception as e:
        log(f"❌ Ошибка парсинга статьи autostat.ru: {e}")
        return None, None, None

def parse_avtonovostidnya_ru(page):
    log("🌐 avtonovostidnya.ru")
    try:
        page.goto("https://avtonovostidnya.ru/", timeout=30000)
        time.sleep(3)
        links = page.evaluate("""() => {
            const items = Array.from(document.querySelectorAll('article'));
            return items.map(item => {
                const link = item.querySelector('h2 a, h3 a');
                return link ? link.href : null;
            }).filter(href => href && href.includes('avtonovostidnya.ru')).slice(0, 5);
        }""")
        log(f"✅ avtonovostidnya.ru: {len(links)} ссылок")
        return links
    except Exception as e:
        log(f"❌ Ошибка avtonovostidnya.ru: {str(e)[:200]}")
        return []

def parse_avtonovostidnya_article(page, url):
    try:
        page.goto(url, timeout=30000)
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
        if not title or not lead:
            log(f"⚠️ avtonovostidnya.ru: Нет данных для {url}")
            return None, None, None
        return title.strip(), lead.strip(), image_url
    except Exception as e:
        log(f"❌ Ошибка парсинга статьи avtonovostidnya.ru: {e}")
        return None, None, None

# def parse_motor_ru(page):
#     log("🌐 motor.ru/pulse")
#     try:
#         page.goto("https://motor.ru/pulse", timeout=60000, wait_until="domcontentloaded")
#         for _ in range(5):
#             page.mouse.wheel(0, 1500)
#             time.sleep(1.5)
#     except Exception as e:
#         log(f"❌ Ошибка motor.ru: {e}")
#         return []

#     soup = BeautifulSoup(page.content(), "html.parser")
#     links = set()
#     for a in soup.find_all("a", href=True):
#         href = a["href"]
#         if href.startswith("/news") or href.startswith("/pulse"):
#             links.add("https://motor.ru" + href)
#     log(f"✅ motor.ru: {len(links)} ссылок")
#     return list(links)[:5]

# def parse_motor_article(page, url):
#     try:
#         page.goto(url, timeout=60000, wait_until="domcontentloaded")
#         page.wait_for_selector("h1", timeout=10000)
#         time.sleep(1)

#         title = page.query_selector("h1").inner_text()
#         lead = page.query_selector("article p")
#         lead_text = lead.inner_text().strip() if lead else ""

#         image = page.query_selector("article img")
#         image_url = image.get_attribute("src") if image else ""

#         if image_url and image_url.startswith("/"):
#             image_url = urljoin(url, image_url)

#         if not title or not lead_text:
#             log(f"⚠️ motor.ru: title={bool(title)}, lead={bool(lead_text)}, img={bool(image_url)}")
#             return title.strip(), lead_text, ""  # Даже если нет картинки, возвращаем текст

#         return title.strip(), lead_text, image_url

#     except Exception as e:
#         log(f"❌ motor.ru error: {e}")
#         return None, None, None

# === ПАРСИНГ auto.ru ===
def parse_auto_ru(page):
    log("🌐 auto.ru/mag/theme/news/")
    try:
        page.goto("https://auto.ru/mag/theme/news/", timeout=60000, wait_until="domcontentloaded")
        for _ in range(7):
            page.mouse.wheel(0, 2000)
            time.sleep(1.2)
    except Exception as e:
        log(f"❌ Ошибка auto.ru: {e}")
        return []

    try:
        links = page.evaluate("""
            () => Array.from(document.querySelectorAll('a'))
                .map(a => a.href)
                .filter(h => h.includes('/mag/article/') && !h.includes('video') && !h.includes('gallery'))
        """)
        log(f"✅ auto.ru: {len(links)} ссылок")
        return list(set(links))[:5]
    except Exception as e:
        log(f"❌ Ошибка JS auto.ru: {e}")
        return []

def parse_auto_article(page, url):
    try:
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        page.wait_for_selector("h1", timeout=10000)
        time.sleep(2)

        title = page.query_selector("h1").inner_text()
        image = page.query_selector("meta[property='og:image']")
        image_url = image.get_attribute("content") if image else ""

        lead_text = page.evaluate("""
            () => {
                const raw = document.body.innerText;
                if (!raw) return "";

                const lines = raw
                    .split('\\n')
                    .map(t => t.trim())
                    .filter(t =>
                        t.length > 100 &&
                        !t.toLowerCase().includes("фото") &&
                        !t.startsWith("Читайте также") &&
                        !t.toLowerCase().includes("источник") &&
                        !t.startsWith("Смотрите также")
                    );

                return lines.length ? lines[0] : "";
            }
        """)

        if not all([title, lead_text, image_url]):
            log(f"⚠️ auto.ru: title={bool(title)}, lead={bool(lead_text)}, img={bool(image_url)}")
            return None, None, None

        return title.strip(), lead_text.strip(), image_url

    except Exception as e:
        log(f"❌ auto.ru error: {e}")
        return None, None, None

# === ОСНОВНАЯ ЛОГИКА ===
def main():
    published = load_published()
    threading.Thread(target=publish_loop, daemon=True).start()

    while True:
        log("🔄 Запуск нового цикла сбора и публикации новостей...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent="Mozilla/5.0")
            page = context.new_page()

            # motor.ru
            # for link in parse_motor_ru(page):
            #     if link in published:
            #         log(f"⏭ Уже есть motor.ru: {link}")
            #         continue
            #     log(f"🔍 motor.ru: {link}")
            #     title, lead, image_url = parse_motor_article(page, link)
            #     if title and lead and image_url:
            #         enqueue_post(title, lead, image_url, link, "motor.ru")
            #         published.append(link)
            #         save_published(published)

            # auto.ru
            for link in parse_auto_ru(page):
                if any(isinstance(item, dict) and link == item.get("link") for item in published):
                    log(f"⏭ Уже есть auto.ru: {link}")
                    continue
                log(f"🔍 auto.ru: {link}")
                result = parse_auto_article(page, link)
                if result:
                    title, lead, image_url = result
                    if title and lead and image_url:
                        if not is_duplicate(title, lead, published):
                            embedding = model.encode(f"{title.strip()} {lead.strip()}").tolist()
                            enqueue_post(title, lead, image_url, link, "auto.ru")
                            published.append({"link": link, "embedding": embedding})
                            save_published(published)
                        else:
                            log(f"🚫 Пропущена как дубликат: {link}")

            # kolesa.ru
            for link in parse_kolesa_ru(page):
                if any(isinstance(item, dict) and link == item.get("link") for item in published):
                    log(f"⏭ Уже есть kolesa.ru: {link}")
                    continue
                log(f"🔍 kolesa.ru: {link}")
                result = parse_kolesa_article(page, link)
                if result:
                    title, lead, image_url = result
                    if title and lead and image_url:
                        if not is_duplicate(title, lead, published):
                            embedding = model.encode(f"{title.strip()} {lead.strip()}").tolist()
                            enqueue_post(title, lead, image_url, link, "kolesa.ru")
                            published.append({"link": link, "embedding": embedding})
                            save_published(published)
                        else:
                            log(f"🚫 Пропущена как дубликат: {link}")   

            # autostat.ru
            for link in parse_autostat_ru(page):
                if any(isinstance(item, dict) and link == item.get("link") for item in published):
                    log(f"⏭ Уже есть autostat.ru: {link}")
                    continue
                log(f"🔍 autostat.ru: {link}")
                result = parse_autostat_article(page, link)
                if result:
                    title, lead, image_url = result
                    if title and lead and image_url:
                        if not is_duplicate(title, lead, published):
                            embedding = model.encode(f"{title.strip()} {lead.strip()}").tolist()
                            enqueue_post(title, lead, image_url, link, "autostat.ru")
                            published.append({"link": link, "embedding": embedding})
                            save_published(published)
                        else:
                            log(f"🚫 Пропущена как дубликат: {link}")

            # avtonovostidnya.ru
            for link in parse_avtonovostidnya_ru(page):
                if any(isinstance(item, dict) and link == item.get("link") for item in published):
                    log(f"⏭ Уже есть avtonovostidnya.ru: {link}")
                    continue
                log(f"🔍 avtonovostidnya.ru: {link}")
                result = parse_avtonovostidnya_article(page, link)
                if result:
                    title, lead, image_url = result
                    if title and lead and image_url:
                        if not is_duplicate(title, lead, published):
                            embedding = model.encode(f"{title.strip()} {lead.strip()}").tolist()
                            enqueue_post(title, lead, image_url, link, "avtonovostidnya.ru")
                            published.append({"link": link, "embedding": embedding})
                            save_published(published)
                        else:
                            log(f"🚫 Пропущена как дубликат: {link}")
                            
            browser.close()
            log("✅ Цикл завершён. Жду 30 минут...\n")

        time.sleep(1800)  # 30 минут

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⛔ Скрипт остановлен вручную. До встречи!")
