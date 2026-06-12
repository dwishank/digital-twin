"""
retrieval.py — chunking + hybrid BM25/vector retrieval for the Digital Twin.
"""

import re, math
from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- 1. Sliding window chunking ---
def sliding_window_chunks(text: str, size: int = 300, overlap: int = 60) -> List[str]:
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i+size]) for i in range(0, len(words), step) if words[i:i+size]]


# --- 2. Semantic chunking (paragraph-aware) ---
def semantic_chunks(text: str, max_words: int = 300) -> List[str]:
    chunks, buf = [], ""
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        if len((buf + " " + para).split()) <= max_words:
            buf = (buf + " " + para).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks or [text]


# --- 3. BM25 (no external deps) ---
def bm25_scores(corpus: List[str], query: str, k1=1.5, b=0.75) -> List[float]:
    tok = lambda s: re.findall(r"\w+", s.lower())
    docs = [tok(d) for d in corpus]
    avgdl = sum(len(d) for d in docs) / len(docs)
    df = {}
    for d in docs:
        for t in set(d): df[t] = df.get(t, 0) + 1
    N = len(docs)
    scores = []
    for doc in docs:
        freq = {t: doc.count(t) for t in doc}
        s = sum(
            math.log((N - df.get(t,0) + 0.5) / (df.get(t,0) + 0.5) + 1)
            * (freq.get(t,0) * (k1+1))
            / (freq.get(t,0) + k1 * (1 - b + b * len(doc) / avgdl))
            for t in tok(query) if t in freq
        )
        scores.append(s)
    return scores


# --- 4. Vector embed + cosine ---
def embed(texts: List[str]) -> List[List[float]]:
    return [r.embedding for r in client.embeddings.create(model="text-embedding-3-small", input=texts).data]

def cosine(a, b) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)) + 1e-10)


# --- 5. KnowledgeBase: index once, retrieve with hybrid BM25 + vector ---
class KnowledgeBase:
    def __init__(self, text: str):
        self.chunks = semantic_chunks(text)
        self.embeddings = embed(self.chunks)          # pre-computed at startup

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[str]:
        bm25 = bm25_scores(self.chunks, query)
        vecs  = [cosine(embed([query])[0], e) for e in self.embeddings]

        def norm(s):
            mn, mx = min(s), max(s)
            return [(x-mn)/(mx-mn or 1) for x in s]

        scores = [alpha*b + (1-alpha)*v for b, v in zip(norm(bm25), norm(vecs))]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_k]]
