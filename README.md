# AutoNewsBot Parser

AutoNewsBot is a fully automated parser and Telegram publisher that collects the latest automotive news from several Russian websites, filters duplicates using neural embeddings, and publishes posts at timed intervals. It includes logging, error notifications, and embedding-based duplicate detection.

---

## 🔧 Features

- Parses news from:
  - [auto.ru](https://auto.ru/mag/theme/news/)
  - [kolesa.ru](https://www.kolesa.ru/news)
  - [autostat.ru](https://www.autostat.ru/news/)
  - [avtonovostidnya.ru](https://avtonovostidnya.ru/)
- Uses `sentence-transformers` with the `all-mpnet-base-v2` model for semantic duplicate detection
- Detects duplicates based on cosine similarity of article embeddings
- Posts include:
  - title
  - lead paragraph
  - image
  - source link
- Posts one article every 10 minutes (with delay logic)
- Logs all activity to both console and `log_parser.txt`
- Sends startup and error notifications to the admin via Telegram
- Falls back to a placeholder image if the source image is invalid
- Runs continuous collection and posting loop every hour
- Stores all published articles in `published_articles.json` with title, lead, link, embedding, and timestamp

---

## 🛠 Installation

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
playwright install
```

2. `requirements.txt` contains:

```text
beautifulsoup4==4.12.2
requests==2.31.0
telebot==0.0.5
torch==2.1.2
sentence-transformers==2.5.1
numpy==1.26.4
playwright==1.43.0
```

---

## ⚙️ Configuration

Open `Autonews_Parser_v1.9.2.py` and set the following variables:

```python
TELEGRAM_TOKEN = 'your_bot_token'
TELEGRAM_CHAT_ID = 'your_channel_or_group_id'
ADMIN_CHAT_ID = 'your_telegram_user_id'
```

- `TELEGRAM_TOKEN`: your Telegram Bot API token  
- `TELEGRAM_CHAT_ID`: the ID of the channel or group where posts will be sent  
- `ADMIN_CHAT_ID`: your personal Telegram user ID for notifications

---

## 🚀 How It Works

1. Loads previously published articles from `published_articles.json`
2. Starts a new parsing cycle every hour
3. For each site:
   - Fetches a list of article links
   - Visits the article
   - Extracts title, lead paragraph, and image
   - Encodes the combined text using `sentence-transformers`
   - Compares embedding with:
     - previously published embeddings
     - last 50 session embeddings
   - If not a duplicate, queues the article for posting
4. A separate loop publishes one article every 10 minutes to the Telegram channel
5. After successful post:
   - Adds the embedding and article metadata to `published_articles.json`
   - Logs the action

---

## 🧪 Duplicate Detection

Each article undergoes two levels of duplicate checking:

1. **Against published history:** compares with saved embeddings in `published_articles.json`
2. **Within current session:** compares with last 50 embeddings processed during current runtime

An article is skipped if cosine similarity ≥ 0.9 in either comparison.

---

## 📂 Logging

All logs are written to `log_parser.txt`. You can follow the log in real-time:

```bash
tail -f log_parser.txt
```

The log includes:

- Article collection start
- Link counts
- Parsing results
- Error messages
- Cosine similarity scores
- Posting confirmations

---

## ▶️ Running the Bot

Use Python to run the script:

```bash
python Autonews_Parser_v1.9.2.py
```

If you're running this on a server, consider using `tmux` or `screen`:

```bash
tmux new -s autonews
python Autonews_Parser_v1.9.2.py
```

---

## 🧹 Maintenance Tips

- `published_articles.json` will grow over time. To prevent memory issues:
  - Manually delete articles older than 2–4 weeks
  - Or implement automatic cleanup logic
- `log_parser.txt` can also be rotated or cleared periodically

If needed, ask for automatic pruning instructions.

---

## 📄 License

MIT — free to use and modify.

---

## 👤 Author

Developed by [your_name]  
Telegram: [@your_handle]  
GitHub: [https://github.com/your_username](https://github.com/your_username)
