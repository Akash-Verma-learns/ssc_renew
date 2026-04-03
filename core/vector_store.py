"""
Vector Store
------------
Handles ChromaDB ingestion and semantic retrieval of RFP chunks.
Uses sentence-transformers for local, free embeddings (no API key needed).

Model used: all-MiniLM-L6-v2
  - 80MB, runs on CPU
  - Fast and accurate for legal/contract text retrieval

v2 additions
  - get_all_chunks_for_doc(doc_name) — returns ALL chunks for a document
    sorted by page_no.  Used by tq_extractor v5 for schema extraction so
    that the full evaluation table reaches the LLM without embedding-search
    truncation.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List
from core.parser import Chunk


# ─────────────────────────────────────────────────────────────────────────────
# Singleton vector store (persists to disk)
# ─────────────────────────────────────────────────────────────────────────────

_client     = None
_collection = None

COLLECTION_NAME = "rfp_chunks"
DB_PATH         = "./chroma_db"


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ─────────────────────────────────────────────────────────────────────────────
# Ingest
# ─────────────────────────────────────────────────────────────────────────────

def ingest_chunks(chunks: List[Chunk], doc_id: str) -> int:
    """
    Ingest parsed chunks into ChromaDB.
    Deletes old chunks for the same doc first (upsert behaviour).
    Returns count of chunks ingested.
    """
    collection = _get_collection()

    try:
        existing = collection.get(where={"doc_name": doc_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"[VectorStore] Deleted {len(existing['ids'])} old chunks for '{doc_id}'")
    except Exception:
        pass

    if not chunks:
        return 0

    batch_size = 100
    total      = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "page_no":         c.page_no,
                    "section_heading": c.section_heading,
                    "clause_ref":      c.clause_ref,
                    "doc_name":        c.doc_name,
                }
                for c in batch
            ],
        )
        total += len(batch)

    print(f"[VectorStore] Ingested {total} chunks for '{doc_id}'")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Semantic retrieval (for clause extraction and evidence queries)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(query: str, doc_name: str, top_k: int = 5) -> List[dict]:
    """
    Semantic search for the most relevant chunks in a specific document.

    Returns list of:
    {
        "text": str,
        "page_no": int,
        "section_heading": str,
        "clause_ref": str,
        "score": float   (0–1, higher = more similar)
    }
    """
    collection = _get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"doc_name": doc_name},
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text":            doc,
            "page_no":         meta.get("page_no", 0),
            "section_heading": meta.get("section_heading", ""),
            "clause_ref":      meta.get("clause_ref", ""),
            "score":           round(1 - dist, 4),
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Full-document retrieval (used by TQ schema extraction — no embedding bias)
# ─────────────────────────────────────────────────────────────────────────────

def get_all_chunks_for_doc(doc_name: str) -> List[dict]:
    """
    Return ALL stored chunks for a given document, sorted by page_no.

    Unlike retrieve(), this does NOT use embedding similarity — it returns
    every chunk in page order.  Used by TQ schema extraction to guarantee
    the full evaluation table reaches the LLM without top-k truncation.

    Returns list of same shape as retrieve() but without a 'score' key.
    """
    collection = _get_collection()
    try:
        result = collection.get(
            where={"doc_name": doc_name},
            include=["documents", "metadatas"],
        )
    except Exception as e:
        print(f"[VectorStore] get_all_chunks_for_doc error: {e}")
        return []

    chunks = []
    for doc, meta in zip(result.get("documents", []), result.get("metadatas", [])):
        if not doc:
            continue
        chunks.append({
            "text":            doc,
            "page_no":         meta.get("page_no", 0) if meta else 0,
            "section_heading": meta.get("section_heading", "") if meta else "",
            "clause_ref":      meta.get("clause_ref", "") if meta else "",
        })

    chunks.sort(key=lambda x: x["page_no"])
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def list_docs() -> List[str]:
    """List all document names currently in the vector store."""
    collection = _get_collection()
    all_meta   = collection.get(include=["metadatas"])["metadatas"]
    return list({m["doc_name"] for m in all_meta if m})


def delete_doc(doc_name: str) -> int:
    """Remove all chunks for a specific document."""
    collection = _get_collection()
    existing   = collection.get(where={"doc_name": doc_name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        return len(existing["ids"])
    return 0

def get_all_chunks_for_doc(doc_name: str) -> list:
    """
    Return ALL chunks for a document in page-number order.
    Uses ChromaDB .get() (no embedding query) so nothing is filtered out.
 
    Returns list of dicts with keys: text, page_no, section_heading,
    clause_ref, score (always 1.0 for direct retrieval).
    """
    collection = _get_collection()
    try:
        result = collection.get(
            where={"doc_name": doc_name},
            include=["documents", "metadatas"],
        )
    except Exception as e:
        print(f"[VectorStore] get_all_chunks_for_doc failed for '{doc_name}': {e}")
        return []
 
    if not result or not result.get("ids"):
        return []
 
    chunks = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        chunks.append({
            "text":            doc,
            "page_no":         meta.get("page_no", 0),
            "section_heading": meta.get("section_heading", ""),
            "clause_ref":      meta.get("clause_ref", ""),
            "score":           1.0,   # direct retrieval — not similarity-ranked
        })
 
    chunks.sort(key=lambda x: (x["page_no"], x["section_heading"]))
    return chunks
 