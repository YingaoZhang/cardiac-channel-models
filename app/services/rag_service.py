from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import warnings
from contextlib import redirect_stderr
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from app.settings import (
    BASE_DIR,
    BGE_M3_MODEL,
    KNOWLEDGE_INDEX_PATH,
    KNOWLEDGE_RAW_DIR,
    PGVECTOR_DATABASE_URL,
    RAG_HYBRID_DENSE_WEIGHT,
    RAG_HYBRID_SPARSE_WEIGHT,
    RAG_SOURCE_DIR,
    RAG_VECTOR_DIMENSION,
)


HERG_DATA_PATH = BASE_DIR / "data" / "dev" / "herg_dev.csv"
SUPPORTED_KNOWLEDGE_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".pdf"}
POSTGRES_CANDIDATE_LIMIT = 48


def _risk_from_label(label: float) -> str:
    return "known hERG blocker" if float(label) >= 0.5 else "known non-blocker"


def _canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=1)
def _load_herg_cases() -> tuple[dict[str, Any], ...]:
    if not HERG_DATA_PATH.exists():
        return ()

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cases: list[dict[str, Any]] = []

    with HERG_DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles = str(row.get("SMILES", "")).strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            try:
                label = 1.0 if row.get("Activity_Label") == "Blocker" else 0.0
            except (TypeError, ValueError):
                continue

            cases.append(
                {
                    "kind": "structure_case",
                    "smiles": smiles,
                    "label": label,
                    "fingerprint": generator.GetFingerprint(mol),
                }
            )

    return tuple(cases)


def retrieve_herg_cases(smiles: str, top_k: int = 5) -> list[dict[str, Any]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    cases = _load_herg_cases()
    if not cases:
        return []

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query_fp = generator.GetFingerprint(mol)

    ranked = []
    for case in cases:
        similarity = DataStructs.TanimotoSimilarity(query_fp, case["fingerprint"])
        ranked.append((similarity, case))

    ranked.sort(key=lambda item: item[0], reverse=True)

    results = []
    query_canonical = _canonicalize(smiles)
    for similarity, case in ranked[:top_k]:
        results.append(
            {
                "kind": "structure_case",
                "source": "data/dev/herg_dev.csv",
                "smiles": case["smiles"],
                "label": case["label"],
                "risk": _risk_from_label(case["label"]),
                "score": round(float(similarity), 4),
                "similarity": round(float(similarity), 4),
                "is_exact_match": query_canonical == case["smiles"],
            }
        )

    return results


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _postgres_text(value: Any) -> str:
    return str(value).replace("\x00", " ")


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 160) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")


def _read_json_file(path: Path) -> str:
    try:
        data = json.loads(_read_text_file(path))
    except json.JSONDecodeError:
        return _read_text_file(path)
    return json.dumps(data, ensure_ascii=False, indent=2)


def _read_csv_file(path: Path) -> str:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if index > 5000:
                break
            rows.append("; ".join(f"{key}: {value}" for key, value in row.items()))
    return "\n".join(rows)


def _read_pdf_file(path: Path) -> str:
    with redirect_stderr(io.StringIO()):
        reader = PdfReader(str(path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {page_number}]\n{text}")
    return "\n\n".join(pages)


def _file_to_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return _read_text_file(path)
    if suffix == ".json":
        return _read_json_file(path)
    if suffix == ".csv":
        return _read_csv_file(path)
    if suffix == ".pdf":
        return _read_pdf_file(path)
    return ""


def _iter_source_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    return [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_KNOWLEDGE_SUFFIXES
    ]


def _collect_documents(source_dir: Path) -> list[dict[str, Any]]:
    documents = []
    for path in _iter_source_files(source_dir):
        try:
            text = _file_to_text(path)
        except Exception as exc:
            documents.append(
                {
                    "id": hashlib.sha1(str(path).encode("utf-8")).hexdigest(),
                    "kind": "ingest_error",
                    "source": str(path),
                    "chunk_index": 0,
                    "text": "",
                    "error": str(exc),
                }
            )
            continue

        for chunk_index, chunk in enumerate(_chunk_text(text)):
            chunk_id = hashlib.sha1(f"{path}:{chunk_index}:{chunk}".encode("utf-8")).hexdigest()
            documents.append(
                {
                    "id": chunk_id,
                    "kind": "knowledge_chunk",
                    "source": str(path),
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )
    return documents


def _normalize_vector(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array.tolist()
    return (array / norm).astype(float).tolist()


@lru_cache(maxsize=1)
def _load_bge_encoder() -> Any:
    warnings.filterwarnings(
        "ignore",
        message="Failed to load image Python extension.*",
        category=UserWarning,
    )
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(
            BGE_M3_MODEL,
            model_kwargs={"use_safetensors": True},
        )
    except TypeError:
        return SentenceTransformer(BGE_M3_MODEL)


def _encode_bge_m3(text: str) -> tuple[list[float], dict[str, float]]:
    model = _load_bge_encoder()

    dense = model.encode([text], normalize_embeddings=True)[0]
    return list(map(float, dense)), {}


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _sparse_dot(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    if len(left) > len(right):
        left, right = right, left

    return sum(weight * float(right.get(token, 0.0)) for token, weight in left.items())


def _connect_pg():
    import psycopg

    return psycopg.connect(PGVECTOR_DATABASE_URL, connect_timeout=3)


def ensure_pgvector_schema() -> None:
    with _connect_pg() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({RAG_VECTOR_DIMENSION}) NOT NULL,
                    sparse_embedding JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    text_search TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(text, ''))
                    ) STORED
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS rag_documents_embedding_hnsw
                ON rag_documents USING hnsw (embedding vector_cosine_ops)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS rag_documents_text_search_gin
                ON rag_documents USING gin (text_search)
                """
            )
        conn.commit()


def build_knowledge_index(source_dir: Path | None = None) -> dict[str, Any]:
    source_dir = source_dir or RAG_SOURCE_DIR
    KNOWLEDGE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    documents = _collect_documents(source_dir)
    valid_documents = [doc for doc in documents if doc.get("kind") == "knowledge_chunk" and doc.get("text")]

    payload = {
        "version": 3,
        "mode": "postgres_pgvector_bge_m3_hybrid",
        "source_dir": str(source_dir),
        "document_count": len(valid_documents),
        "error_count": len(documents) - len(valid_documents),
        "documents": documents,
    }
    KNOWLEDGE_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _load_knowledge_documents.cache_clear()
    return payload


def build_vector_index(source_dir: Path | None = None) -> dict[str, Any]:
    source_dir = source_dir or RAG_SOURCE_DIR
    text_payload = build_knowledge_index(source_dir=source_dir)
    documents = [doc for doc in text_payload["documents"] if doc.get("kind") == "knowledge_chunk" and doc.get("text")]

    ensure_pgvector_schema()

    rows = []
    total_documents = len(documents)
    for index, doc in enumerate(documents, start=1):
        if index == 1 or index % 10 == 0 or index == total_documents:
            print(f"[RAG] Encoding chunk {index}/{total_documents} with {BGE_M3_MODEL}...")
        dense, sparse = _encode_bge_m3(doc["text"])
        if len(dense) != RAG_VECTOR_DIMENSION:
            raise ValueError(
                f"BGE-M3 embedding dimension is {len(dense)}, "
                f"but RAG_VECTOR_DIMENSION={RAG_VECTOR_DIMENSION}."
            )
        rows.append(
            (
                doc["id"],
                _postgres_text(doc["source"]),
                int(doc["chunk_index"]),
                _postgres_text(doc["text"]),
                _vector_literal(dense),
                json.dumps(sparse),
                json.dumps({"model": BGE_M3_MODEL}, ensure_ascii=False),
            )
        )

    with _connect_pg() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE rag_documents")
            cur.executemany(
                """
                INSERT INTO rag_documents (
                    id, source, chunk_index, text, embedding, sparse_embedding, metadata
                )
                VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    source = EXCLUDED.source,
                    chunk_index = EXCLUDED.chunk_index,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    sparse_embedding = EXCLUDED.sparse_embedding,
                    metadata = EXCLUDED.metadata,
                    indexed_at = now()
                """,
                rows,
            )
        conn.commit()

    return {
        "version": 3,
        "mode": "postgres_pgvector_bge_m3_hybrid",
        "model": BGE_M3_MODEL,
        "database_url": _safe_database_url(PGVECTOR_DATABASE_URL),
        "source_dir": str(source_dir),
        "document_count": len(documents),
        "embedding_shape": [len(documents), RAG_VECTOR_DIMENSION],
        "dense_weight": RAG_HYBRID_DENSE_WEIGHT,
        "sparse_weight": RAG_HYBRID_SPARSE_WEIGHT,
    }


def _safe_database_url(url: str) -> str:
    return re.sub(r"://([^:/]+):([^@]+)@", r"://\1:***@", url)


def pgvector_health() -> dict[str, Any]:
    try:
        ensure_pgvector_schema()
        with _connect_pg() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM rag_documents")
                document_count = cur.fetchone()[0]
        return {
            "status": "ready",
            "document_count": document_count,
            "model": BGE_M3_MODEL,
            "database_url": _safe_database_url(PGVECTOR_DATABASE_URL),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "document_count": 0,
            "model": BGE_M3_MODEL,
            "database_url": _safe_database_url(PGVECTOR_DATABASE_URL),
            "error": str(exc),
        }


@lru_cache(maxsize=1)
def _load_knowledge_documents() -> tuple[dict[str, Any], ...]:
    if not KNOWLEDGE_INDEX_PATH.exists():
        return ()

    try:
        payload = json.loads(KNOWLEDGE_INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ()

    docs = payload.get("documents", [])
    if not isinstance(docs, list):
        return ()
    return tuple(doc for doc in docs if isinstance(doc, dict) and doc.get("text"))


def retrieve_knowledge_context_pgvector(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    dense_query, sparse_query = _encode_bge_m3(query)
    if len(dense_query) != RAG_VECTOR_DIMENSION:
        raise ValueError(
            f"BGE-M3 embedding dimension is {len(dense_query)}, "
            f"but RAG_VECTOR_DIMENSION={RAG_VECTOR_DIMENSION}."
        )

    vector = _vector_literal(dense_query)
    candidates: dict[str, dict[str, Any]] = {}

    with _connect_pg() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, source, chunk_index, text,
                    1 - (embedding <=> %s::vector) AS dense_score,
                    ts_rank_cd(text_search, websearch_to_tsquery('english', %s)) AS text_score,
                    sparse_embedding
                FROM rag_documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, query, vector, POSTGRES_CANDIDATE_LIMIT),
            )
            for row in cur.fetchall():
                candidates[row[0]] = _candidate_from_row(row)

            cur.execute(
                """
                SELECT
                    id, source, chunk_index, text,
                    1 - (embedding <=> %s::vector) AS dense_score,
                    ts_rank_cd(text_search, websearch_to_tsquery('english', %s)) AS text_score,
                    sparse_embedding
                FROM rag_documents
                WHERE text_search @@ websearch_to_tsquery('english', %s)
                ORDER BY text_score DESC
                LIMIT %s
                """,
                (vector, query, query, POSTGRES_CANDIDATE_LIMIT),
            )
            for row in cur.fetchall():
                candidates[row[0]] = _candidate_from_row(row)

    if not candidates:
        return []

    max_dense = max(abs(item["dense_score"]) for item in candidates.values()) or 1.0
    max_sparse = max(abs(item["text_score"]) for item in candidates.values()) or 1.0

    for item in candidates.values():
        sparse_model_score = _sparse_dot(sparse_query, item.pop("sparse_embedding", {}) or {})
        sparse_score = max(item["text_score"] / max_sparse, sparse_model_score)
        dense_score = item["dense_score"] / max_dense
        item["score"] = (
            RAG_HYBRID_DENSE_WEIGHT * dense_score
            + RAG_HYBRID_SPARSE_WEIGHT * sparse_score
        )
        item["dense_score"] = round(float(item["dense_score"]), 4)
        item["sparse_score"] = round(float(sparse_score), 4)

    ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    return [
        {
            "kind": "knowledge_chunk",
            "retriever": "pgvector_bge_m3_hybrid",
            "source": item["source"],
            "chunk_index": item["chunk_index"],
            "score": round(float(item["score"]), 4),
            "dense_score": item["dense_score"],
            "sparse_score": item["sparse_score"],
            "text": item["text"],
        }
        for item in ranked[:top_k]
    ]


def _candidate_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
    sparse = row[6]
    if isinstance(sparse, str):
        sparse = json.loads(sparse)
    return {
        "id": row[0],
        "source": row[1],
        "chunk_index": row[2],
        "text": row[3],
        "dense_score": float(row[4] or 0.0),
        "text_score": float(row[5] or 0.0),
        "sparse_embedding": sparse,
    }


def retrieve_knowledge_context_tfidf(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = _load_knowledge_documents()
    if not docs or not query.strip():
        return []

    corpus = [doc["text"] for doc in docs]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()

    ranked_indices = scores.argsort()[::-1]
    results = []
    for index in ranked_indices[:top_k]:
        score = float(scores[index])
        if math.isclose(score, 0.0):
            continue
        doc = docs[int(index)]
        results.append(
            {
                "kind": "knowledge_chunk",
                "retriever": "tfidf_fallback",
                "source": doc.get("source", "unknown"),
                "chunk_index": doc.get("chunk_index", 0),
                "score": round(score, 4),
                "text": doc.get("text", ""),
            }
        )
    return results


def retrieve_knowledge_context(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    try:
        pgvector_results = retrieve_knowledge_context_pgvector(query, top_k=top_k)
        if pgvector_results:
            return pgvector_results
    except Exception as exc:
        print(f"[RAG] pgvector retrieval failed, using TF-IDF fallback: {exc}")

    return retrieve_knowledge_context_tfidf(query, top_k=top_k)


def retrieve_rag_context(smiles: str, top_k_cases: int = 5, top_k_knowledge: int = 5) -> list[dict[str, Any]]:
    query = (
        f"hERG cardiotoxicity medicinal chemistry optimization SAR ion channel "
        f"lipophilicity basic amine aromatic fragment SMILES {smiles}"
    )
    return [
        *retrieve_herg_cases(smiles, top_k=top_k_cases),
        *retrieve_knowledge_context(query, top_k=top_k_knowledge),
    ]


def summarize_herg_context(context: list[dict[str, Any]]) -> str:
    if not context:
        return "No RAG context was retrieved."

    lines = []
    for index, item in enumerate(context, start=1):
        if item.get("kind") == "structure_case":
            lines.append(
                (
                    f"{index}. [Structure case] score={item['score']}, "
                    f"label={item['label']} ({item['risk']}), "
                    f"smiles={item['smiles']}"
                )
            )
            continue

        snippet = _normalize_text(item.get("text", ""))[:420]
        lines.append(
            (
                f"{index}. [Knowledge/{item.get('retriever', 'text')}] score={item.get('score')}, "
                f"dense={item.get('dense_score', '-')}, sparse={item.get('sparse_score', '-')}, "
                f"source={item.get('source')}#{item.get('chunk_index')}: {snippet}"
            )
        )

    return "\n".join(lines)


retrieve_herg_context = retrieve_rag_context
