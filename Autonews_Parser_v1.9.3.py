import json
import time
import threading
import traceback
import telebot
import random
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
import sys
import torch
import requests
import hashlib
from datetime import datetime, timedelta, timezone
import telebot.apihelper
from sentence_transformers import CrossEncoder
import faiss
import os

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

model = SentenceTransformer('all-mpnet-base-v2')
sent_error_hashes = set()
session_embeddings = []
SESSION_EMBEDDING_LIMIT = 50
cross_model = CrossEncoder("cross-encoder/stsb-roberta-large")

# Настройки
TELEGRAM_TOKEN = ''
TELEGRAM_CHAT_ID = ''
PUBLISHED_FILE = "published_articles.json"
ADMIN_CHAT_ID = ''
OPENROUTER_API_KEY = ''

bot = telebot.TeleBot(TELEGRAM_TOKEN)
post_queue = []
all_parsed_articles = []
last_sent_time = 0
faiss_index = None
faiss_texts = []

def safe_decode(text):
    if not text:
        return text
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, AttributeError):
        return text

def is_valid_image_url(url):
    try:
        r = requests.head(url, timeout=5, allow_redirects=True)
        content_type = r.headers.get("Content-Type", "")
        return content_type.startswith("image/")
    except:
        return False

def notify_admin(text, image_url=None, stop_on_error=False):
    error_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

    if error_hash in sent_error_hashes:
        log(f"🚫 Уведомление уже отправлялось: {text.splitlines()[0]}")
        return

    sent_error_hashes.add(error_hash)

    try:
        if image_url and is_valid_image_url(image_url):
            bot.send_photo(chat_id=ADMIN_CHAT_ID, photo=image_url, caption=text, parse_mode="HTML")
        else:
            bot.send_message(chat_id=ADMIN_CHAT_ID, text=text, parse_mode="HTML")
        log("🔔 Уведомление отправлено админу")

        if stop_on_error:
            log("⛔ Скрипт остановлен после отправки уведомления админу.")
            sys.exit(1)

    except Exception as e:
        log(f"❌ Ошибка при отправке уведомления админу: {e}")
        if stop_on_error:
            log("⛔ Скрипт принудительно остановлен из-за ошибки при отправке уведомления.")
            sys.exit(1)

def load_published():
    if not os.path.exists(PUBLISHED_FILE):
        return []
    with open(PUBLISHED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_published(data):
    with open(PUBLISHED_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_faiss_index(published_data):
    dimension = 768
    index = faiss.IndexFlatIP(dimension)  # создается всегда
    texts = []
    vectors = []

    for item in published_data:
        if "embedding" in item:
            emb = np.array(item["embedding"], dtype="float32")
            norm = np.linalg.norm(emb)
            if norm != 0:
                emb = emb / norm
            vectors.append(emb)
            texts.append((item.get("link", ""), item.get("text", "")))

    if vectors:
        index.add(np.vstack(vectors))
        log(f"✅ FAISS индекс построен: {len(vectors)} эмбеддингов")
    else:
        log(f"⚠️ FAISS: нет эмбеддингов для построения индекса (создан пустой)")

    return index, texts

def is_duplicate(title, lead, published_data, threshold_faiss=0.9, threshold_cross=0.9, llm_min=0.8):
    global faiss_index, faiss_texts
    if not title or not lead:
        log("⚠️ Пропуск дубликата: пустой title или lead")
        return False

    text = f"{title.strip()} {lead.strip()}".lower()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    # -------------------- FAISS --------------------
    if faiss_index is None or faiss_index.ntotal == 0:
        log(f"⚠️ FAISS индекс пуст, проверка невозможна")
        return False

    D, I = faiss_index.search(np.expand_dims(embedding, axis=0), k=faiss_index.ntotal)
    cross_candidates = []

    for score, idx in zip(D[0], I[0]):
        score = float(score)
        if idx >= len(faiss_texts):
            continue

        top_link, top_text = faiss_texts[idx]
        log(f"🔍 FAISS сравнение с {top_link} — score: {score:.3f}")

        if score >= threshold_faiss:
            log(f"⚠️ Подтверждено FAISS как дубликат")
            return True

        cross_candidates.append((top_link, top_text, score))

    # -------------------- CrossEncoder --------------------
    if not cross_candidates:
        return False

    # Отбираем топ-5 по FAISS
    cross_candidates = sorted(cross_candidates, key=lambda x: x[2], reverse=True)[:5]
    llm_candidates = []

    for top_link, top_text, faiss_score in cross_candidates:
        cross_score = cross_model.predict([(text, top_text)])[0]
        log(f"🔍 CrossEncoder сравнение с {top_link} — score: {cross_score:.3f}")

        if cross_score >= threshold_cross:
            log(f"⚠️ Подтверждено CrossEncoder как дубликат")
            return True

        if llm_min <= cross_score < threshold_cross:
            llm_candidates.append((top_link, top_text, cross_score))

    # -------------------- LLM --------------------
    # Отбираем топ-2 для LLM
    llm_candidates = sorted(llm_candidates, key=lambda x: x[2], reverse=True)[:2]

    for top_link, top_text, cross_score in llm_candidates:
        llm_result = llm_check(text, top_text)
        log(f"🔍 LLM сравнение с {top_link}, текст: {text[:20]}...\n, текст из базы: {top_text[:20]}  — результат: {llm_result}")

        if llm_result is True:
            log(f"⚠️ Подтверждено LLM как дубликат")
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
    global faiss_index, faiss_texts
    global last_sent_time
    while True:
        random.shuffle(post_queue)
        if post_queue:
            post = post_queue.pop(0)
            try:
                caption = (
                    f"<b>{post['title']}</b>\n\n"
                    f"{post['lead']}\n\n"
                    f"Источник: <a href=\"{post['link']}\">{post['source']}</a>"
                )

                if is_valid_image_url(post["image_url"]):
                    bot.send_photo(
                        chat_id=TELEGRAM_CHAT_ID,
                        photo=post["image_url"],
                        caption=caption,
                        parse_mode="HTML"
                    )
                    log(f"📤 Опубликовано с изображением: {post['title']}")
                else:
                    log(f"⚠️ Невалидное изображение: {post['image_url']}. Отправка без фото.")
                    bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=caption,
                        parse_mode="HTML"
                    )
                    log(f"📤 Опубликовано без изображения: {post['title']}")

                # ⏱️ Обновляем время отправки
                last_sent_time = time.time()

                # 🧠 Добавляем в published_articles
                embedding = model.encode(f"{post['title']} {post['lead']}", convert_to_numpy=True).tolist()
                published_data = load_published()
                new_entry = {
                    "title": post["title"],
                    "lead": post["lead"],
                    "link": post["link"],
                    "embedding": embedding,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "text": f"{post['title']} {post['lead']}".lower()
                }
                published_data.append(new_entry)
                save_published(published_data)

                # 🔥 Добавляем только новый эмбеддинг в FAISS
                new_emb = np.array(embedding, dtype="float32")
                norm = np.linalg.norm(new_emb)
                if norm != 0:
                    new_emb = new_emb / norm
                if faiss_index is None:
                    dimension = 768
                    faiss_index = faiss.IndexFlatIP(dimension)
                faiss_index.add(np.expand_dims(new_emb, axis=0))
                faiss_texts.append((post["link"], new_entry["text"]))

                # ⏳ Пауза между публикациями
                time.sleep(600)

            except telebot.apihelper.ApiTelegramException as e:
                log(f"❌ Ошибка Telegram API: {e}")
                notify_admin(f"<b>❌ Ошибка Telegram API</b>\n\n{e}\n\n<i>{post['title']}</i>", post["image_url"])
                post_queue.insert(0, post)
                time.sleep(60)

            except Exception as e:
                log(f"❌ Общая ошибка публикации: {e}")
                notify_admin(f"<b>❌ Ошибка публикации</b>\n\n{e}\n\n<i>{post['title']}</i>", post["image_url"])
                post_queue.insert(0, post)
                time.sleep(60)

        else:
            time.sleep(5)

# === ПАРСИНГ kolesa.ru ===

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

        title = safe_decode(title)
        lead = safe_decode(lead)

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

# === ПАРСИНГ autostat.ru ===

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

        title = safe_decode(title)
        lead = safe_decode(lead)

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

# === ПАРСИНГ avtonovostidnya.ru ===
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

        title = safe_decode(title)
        lead = safe_decode(lead)

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

        title = safe_decode(page.query_selector("h1").inner_text())

        # 🚫 Пропуск, если в заголовке есть "Главное за день"
        if "главное за день" or "запросы отправляли вы, а не робот" in title.lower():
            log(f"⏭ Пропущена новость с фразой 'Главное за день' или 'вы не робот': {title}")
            return None, None, None

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

        lead_text = safe_decode(lead_text)

        if not all([title, lead_text, image_url]):
            log(f"⚠️ auto.ru: title={bool(title)}, lead={bool(lead_text)}, img={bool(image_url)}")
            return None, None, None

        return title.strip(), lead_text.strip(), image_url

    except Exception as e:
        log(f"❌ auto.ru error: {e}")
        return None, None, None

def llm_check(text1, text2):
    prompt = f"""
Ты профессиональный редактор автомобильных новостей. Твоя задача: определить, описывают ли эти два текста **одно и то же событие**, даже если слова и формулировки разные. Ответь только **"Да" или "Нет"**, без пояснений.

Текст 1:
{text1}

Текст 2:
{text2}
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek/deepseek-chat-v3-0324:free",  # бесплатная DeepSeek через OpenRouter
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip().lower()
            log(f"🤖 LLM (DeepSeek free) ответ: {answer}")
            return "да" in answer
        else:
            log(f"❌ LLM ошибка: {response.status_code} {response.text}")
            return False

    except Exception as e:
        log(f"❌ Ошибка запроса к LLM: {e}")
        return False

# === ОСНОВНАЯ ЛОГИКА ===
def main():
    published = load_published()
    global faiss_index, faiss_texts
    faiss_index, faiss_texts = build_faiss_index(published)
    threading.Thread(target=publish_loop, daemon=True).start()

    while True:
        log("🔄 Запуск нового цикла сбора и публикации новостей...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent="Mozilla/5.0")
            page = context.new_page()

            # Сбор всех статей со всех источников
            parsed_articles = []

            # auto.ru
            for link in parse_auto_ru(page):
                result = parse_auto_article(page, link)
                if result:
                    title, lead, image_url = result
                    parsed_articles.append(("auto.ru", title, lead, image_url, link))

            # kolesa.ru
            for link in parse_kolesa_ru(page):
                result = parse_kolesa_article(page, link)
                if result:
                    title, lead, image_url = result
                    parsed_articles.append(("kolesa.ru", title, lead, image_url, link))

            # autostat.ru
            for link in parse_autostat_ru(page):
                result = parse_autostat_article(page, link)
                if result:
                    title, lead, image_url = result
                    parsed_articles.append(("autostat.ru", title, lead, image_url, link))

            # avtonovostidnya.ru
            for link in parse_avtonovostidnya_ru(page):
                result = parse_avtonovostidnya_article(page, link)
                if result:
                    title, lead, image_url = result
                    parsed_articles.append(("avtonovostidnya.ru", title, lead, image_url, link))

            for source, title, lead, image_url, link in parsed_articles:
                if any(isinstance(item, dict) and link == item.get("link") for item in published):
                    log(f"⏭ Уже есть {source}: {link}")
                    continue

                log(f"🔍 Проверка {source}: {link}")

                # Проверка FAISS
                if is_duplicate(title, lead, published):
                    log(f"🚫 Пропущена как дубликат: {link}")
                    continue

                # Если не дубликат, добавляем в очередь публикаций
                if not title or not lead:
                    log("⚠️ Пропуск: пустой title или lead в main")
                    continue
                text = f"{title.strip()} {lead.strip()}"
                embedding = model.encode(text).tolist()
                enqueue_post(title, lead, image_url, link, source)
                published.append({
                    "link": link,
                    "embedding": embedding,
                    "text": text
                })
                save_published(published)

                # === ✅ Добавление эмбеддинга в FAISS ===
                if faiss_index is not None and embedding is not None:
                    import numpy as np
                    emb_array = np.array(embedding, dtype="float32")
                    faiss_index.add(np.expand_dims(emb_array, axis=0))
                    faiss_texts.append((link, f"{title.strip()} {lead.strip()}".lower()))
                    log(f"✅ FAISS индекс обновлён: добавлен {link}")

            browser.close()
            log("✅ Цикл завершён. Жду 3 часа...\n")

        time.sleep(10800)  # 3 часа

def notify_startup():
    try:
        bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text="🚀 Парсер автоновостей запущен и начал работу.",
            parse_mode="HTML"
        )
        log("🟢 Уведомление о запуске отправлено админу")
    except Exception as e:
        log(f"❌ Не удалось отправить уведомление о запуске: {e}")

if __name__ == "__main__":
    try:
        notify_startup()
        main()
    except KeyboardInterrupt:
        log("⛔ Скрипт остановлен вручную.")
