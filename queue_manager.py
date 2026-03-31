from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class PostItem:
    title: str
    lead: str
    image_url: str
    link: str
    source: str


class LimitedPostQueue:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._queue: deque[PostItem] = deque()

    def push(self, item: PostItem) -> bool:
        if len(self._queue) >= self.max_size:
            return False
        self._queue.append(item)
        return True

    def pop(self) -> PostItem | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def __len__(self) -> int:
        return len(self._queue)
