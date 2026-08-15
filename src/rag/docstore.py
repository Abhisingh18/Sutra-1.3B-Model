"""In-memory index over documents a user uploads at runtime.

This is where retrieval actually pays off for a model this small. The Wikipedia
index covers under 1% of the encyclopedia, so half of all questions retrieve
the wrong passage and the answer gets worse rather than better. An uploaded
document has 100% coverage by construction: whatever the user asks about is,
by definition, in the corpus they just handed over.

Documents live in memory and disappear when the server restarts. That is
deliberate -- persisting other people's uploads to disk is a decision that
should be made on purpose, not inherited from a demo.
"""

import io
import time
import uuid

import numpy as np

from .ingest import chunk_text

# Guard rails. Embedding is fast but not free, and the whole store sits in RAM.
MAX_BYTES = 10 * 1024 * 1024
MAX_CHUNKS = 4000
MAX_DOCS = 40


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    # Everything else is treated as text. errors="ignore" keeps a stray byte in
    # an otherwise fine file from failing the whole upload.
    return data.decode("utf-8", errors="ignore")


class DocStore:
    """Holds one embedded document per id, newest evicting oldest."""

    def __init__(self, encoder):
        self.enc = encoder
        self.docs = {}

    def add(self, filename: str, data: bytes):
        if len(data) > MAX_BYTES:
            raise ValueError(f"file is larger than {MAX_BYTES // 1024 // 1024} MB")

        text = extract_text(filename, data)
        if not text.strip():
            raise ValueError("no readable text found (a scanned PDF needs OCR)")

        chunks = chunk_text(text)[:MAX_CHUNKS]
        if not chunks:
            raise ValueError("nothing to index")

        embeds = self.enc.encode(chunks, batch_size=64, convert_to_numpy=True,
                                 normalize_embeddings=True).astype(np.float32)

        doc_id = uuid.uuid4().hex[:12]
        self.docs[doc_id] = {"name": filename, "chunks": chunks,
                             "embeds": embeds, "added": time.time()}

        if len(self.docs) > MAX_DOCS:
            oldest = min(self.docs, key=lambda k: self.docs[k]["added"])
            del self.docs[oldest]

        return {"doc_id": doc_id, "name": filename, "chunks": len(chunks),
                "characters": len(text)}

    def search(self, doc_id: str, query: str, k: int = 4):
        doc = self.docs.get(doc_id)
        if not doc:
            return []
        q = self.enc.encode([query], convert_to_numpy=True,
                            normalize_embeddings=True).astype(np.float32)
        scores = doc["embeds"] @ q[0]
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        # No relevance floor here, unlike the Wikipedia retriever. The user
        # uploaded this document to ask about it, so the best passage in it is
        # the right answer to show even when the similarity is middling.
        return [(float(scores[i]), doc["chunks"][i], doc["name"]) for i in idx]
