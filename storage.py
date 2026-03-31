from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class PublishedStorage:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS published_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    link TEXT NOT NULL UNIQUE,
                    title TEXT,
                    lead TEXT,
                    text TEXT NOT NULL,
                    embedding TEXT,
                    source TEXT,
                    is_duplicate INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_published_created_at ON published_articles(created_at)'
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_published_is_duplicate ON published_articles(is_duplicate)'
            )
            conn.commit()

    def load_all(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                '''
                SELECT link, title, lead, text, embedding, source, is_duplicate, created_at
                FROM published_articles
                ORDER BY id ASC
                '''
            ).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            embedding_raw = row['embedding']
            result.append(
                {
                    'link': row['link'],
                    'title': row['title'],
                    'lead': row['lead'],
                    'text': row['text'],
                    'embedding': json.loads(embedding_raw) if embedding_raw else None,
                    'source': row['source'],
                    'is_duplicate': bool(row['is_duplicate']),
                    'created_at': row['created_at'],
                }
            )
        return result

    def link_exists(self, link: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT 1 FROM published_articles WHERE link = ? LIMIT 1',
                (link,),
            ).fetchone()
        return row is not None

    def add_article(
        self,
        *,
        link: str,
        title: str,
        lead: str,
        text: str,
        embedding: list[float] | None,
        source: str,
        is_duplicate: bool,
    ) -> None:
        embedding_json = json.dumps(embedding, ensure_ascii=False) if embedding is not None else None
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT OR IGNORE INTO published_articles (
                    link, title, lead, text, embedding, source, is_duplicate
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (link, title, lead, text, embedding_json, source, int(is_duplicate)),
            )
            conn.commit()
