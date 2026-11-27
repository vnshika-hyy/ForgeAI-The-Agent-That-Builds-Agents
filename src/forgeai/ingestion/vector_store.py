import math
import re
from collections import Counter

from pydantic import BaseModel

from .chunker import Chunk


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class LocalTextStore:
    """Small deterministic lexical retriever requiring no external model."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._document_frequency: Counter[str] = Counter()
        for chunk in chunks:
            for token in set(self._tokens(chunk.text)):
                self._document_frequency[token] += 1

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _score(self, query: str, text: str) -> float:
        query_terms = set(self._tokens(query))
        terms = self._tokens(text)
        if not query_terms or not terms:
            return 0.0
        counts = Counter(terms)
        length = len(self.chunks) or 1
        score = 0.0
        for term in query_terms:
            if term not in counts:
                continue
            frequency = counts[term]
            df = self._document_frequency.get(term, 0)
            idf = math.log((length + 1) / (df + 1)) + 1
            score += (1 + math.log(frequency)) * idf
        return score / math.sqrt(len(terms))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ranked = [
            RetrievedChunk(chunk=chunk, score=self._score(query, chunk.text))
            for chunk in self.chunks
        ]
        ranked.sort(key=lambda result: (-result.score, result.chunk.source, result.chunk.chunk_index))
        return [result for result in ranked[:top_k] if result.score > 0]

