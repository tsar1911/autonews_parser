# Autonews Parser Bot v1.9.3

## Description

This bot scrapes automotive news from multiple Russian websites, checks for duplicates using FAISS, CrossEncoder, and LLM-based semantic comparisons, and publishes unique articles to a Telegram channel.

### Key features

- Multi-source parsing: auto.ru, kolesa.ru, autostat.ru, avtonovostidnya.ru
- Duplicate detection pipeline:
  - FAISS index for vector similarity search (all published articles)
  - CrossEncoder (`stsb-roberta-large`) for refined semantic similarity
  - LLM final check for borderline cases
- Normalizes and updates FAISS index dynamically after each publication
- Sends notifications to admin on script start and critical errors
- Stops on any Telegram API error
- Logs all FAISS, CrossEncoder, and LLM comparisons for debugging

### Installation

1. Clone repository
2. Install dependencies:

```bash
pip install -r requirements.txt
playwright install
```

3. Run script:

```bash
python Autonews_Parser_v1.9.3.py
```

### Configuration

- Model: `all-mpnet-base-v2` (embeddings), `stsb-roberta-large` (CrossEncoder)
- Uses FAISS index with normalized vectors for efficient duplicate search
- Environment variables or hardcoded configs for:
  - Telegram bot token
  - Admin chat ID

### Usage notes

- Ensure GPU or sufficient CPU for model inference speed
- LLM API should return clear boolean "TRUE"/"FALSE" outputs for duplicate confirmation
- Schedule script in a persistent session (tmux, systemd) for continuous parsing

---

**Version:** 1.9.3  
**License:** MIT
