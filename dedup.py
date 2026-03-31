from __future__ import annotations

import logging
from typing import Any

import faiss
import numpy as np
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

from logging_utils import json_log


class DuplicateDetector:
    def __init__(self, logger: logging.Logger, openrouter_api_key: str | None = None):
        self.logger = logger
        self.openrouter_api_key = openrouter_api_key
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.cross_model = CrossEncoder('cross-encoder/stsb-roberta-large')
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.faiss_texts: list[tuple[str, str]] = []

    def build_index(self, published_data: list[dict[str, Any]]) -> None:
        dimension = 768
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_texts = []
        vectors: list[np.ndarray] = []

        for item in published_data:
            embedding = item.get('embedding')
            if not embedding:
                continue
            emb = np.array(embedding, dtype='float32')
            norm = np.linalg.norm(emb)
            if norm != 0:
                emb = emb / norm
            vectors.append(emb)
            self.faiss_texts.append((item.get('link', ''), item.get('text', '')))

        if vectors:
            self.faiss_index.add(np.vstack(vectors))
        json_log(self.logger, 'faiss_index_built', total=len(vectors))

    def encode_text(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def add_embedding(self, link: str, text: str, embedding: list[float]) -> None:
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(768)
        emb = np.array(embedding, dtype='float32')
        norm = np.linalg.norm(emb)
        if norm != 0:
            emb = emb / norm
        self.faiss_index.add(np.expand_dims(emb, axis=0))
        self.faiss_texts.append((link, text.lower()))
        json_log(self.logger, 'faiss_index_updated', link=link, total=self.faiss_index.ntotal)

    def is_duplicate(
        self,
        title: str,
        lead: str,
        threshold_faiss: float = 0.9,
        threshold_cross: float = 0.9,
        llm_min: float = 0.8,
    ) -> bool:
        if not title or not lead:
            json_log(self.logger, 'duplicate_skip_empty')
            return False
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            json_log(self.logger, 'duplicate_skip_empty_faiss')
            return False

        text = f'{title.strip()} {lead.strip()}'.lower()
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.faiss_index.search(np.expand_dims(embedding, axis=0), k=self.faiss_index.ntotal)

        cross_candidates: list[tuple[str, str, float]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx >= len(self.faiss_texts):
                continue
            score = float(score)
            top_link, top_text = self.faiss_texts[idx]
            json_log(self.logger, 'faiss_compare', link=top_link, score=round(score, 4))
            if score >= threshold_faiss:
                json_log(self.logger, 'duplicate_by_faiss', link=top_link, score=round(score, 4))
                return True
            cross_candidates.append((top_link, top_text, score))

        if not cross_candidates:
            return False

        llm_candidates: list[tuple[str, str, float]] = []
        for top_link, top_text, faiss_score in sorted(cross_candidates, key=lambda x: x[2], reverse=True)[:5]:
            cross_score = float(self.cross_model.predict([(text, top_text)])[0])
            json_log(self.logger, 'cross_compare', link=top_link, score=round(cross_score, 4))
            if cross_score >= threshold_cross:
                json_log(self.logger, 'duplicate_by_cross', link=top_link, score=round(cross_score, 4))
                return True
            if llm_min <= cross_score < threshold_cross:
                llm_candidates.append((top_link, top_text, cross_score))

        for top_link, top_text, cross_score in sorted(llm_candidates, key=lambda x: x[2], reverse=True)[:2]:
            llm_result = self.llm_check(text, top_text)
            json_log(self.logger, 'llm_compare', link=top_link, cross_score=round(cross_score, 4), result=llm_result)
            if llm_result:
                json_log(self.logger, 'duplicate_by_llm', link=top_link)
                return True
        return False

    def llm_check(self, text1: str, text2: str) -> bool:
        if not self.openrouter_api_key:
            json_log(self.logger, 'llm_disabled_no_key')
            return False
        prompt = (
            'Ты профессиональный редактор автомобильных новостей. Твоя задача: определить, описывают ли эти два текста ' 
            'одно и то же событие, даже если слова и формулировки разные. Ответь только Да или Нет, без пояснений.\n\n'
            f'Текст 1:\n{text1}\n\nТекст 2:\n{text2}'
        )
        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json',
        }
        data = {
            'model': 'qwen/qwen3.6-plus-preview:free',
            'messages': [{'role': 'user', 'content': prompt}],
        }
        try:
            response = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30,
            )
            if response.status_code != 200:
                json_log(self.logger, 'llm_error', status_code=response.status_code, body=response.text[:300])
                return False
            answer = response.json()['choices'][0]['message']['content'].strip().lower()
            json_log(self.logger, 'llm_response', answer=answer)
            return 'да' in answer
        except Exception as exc:
            json_log(self.logger, 'llm_exception', error=str(exc))
            return False

    def llm_check_last_10(self, text: str, published_data: list[dict[str, Any]]) -> bool:
        if not self.openrouter_api_key:
            json_log(self.logger, 'llm_last10_disabled_no_key')
            return False
        recent_entries = published_data[-10:]
        for entry in recent_entries:
            top_link = entry.get('link', '')
            top_text = entry.get('text', '')
            if not top_text:
                continue
            llm_result = self.llm_check(text, top_text)
            json_log(self.logger, 'llm_last10_compare', link=top_link, result=llm_result)
            if llm_result:
                json_log(self.logger, 'duplicate_by_llm_last10', link=top_link)
                return True
        return False
