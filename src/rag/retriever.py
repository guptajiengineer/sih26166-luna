"""RAG retriever — indexes and queries lunar registration knowledge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.knowledge_base.documents import KNOWLEDGE_DOCUMENTS


class LunarRegistrationRetriever:
    """
    Retrieval-Augmented Generation backend for parameter and strategy selection.
    Uses ChromaDB + sentence-transformers when available; keyword fallback otherwise.
    """

    def __init__(
        self,
        collection_name: str = "lunar_registration_kb",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str | Path = "data/knowledge_base/chroma",
    ) -> None:
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.persist_dir = Path(persist_dir)
        self._collection = None
        self._use_chroma = False
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = client.get_or_create_collection(self.collection_name)
            self._embedder = SentenceTransformer(self.embedding_model_name)
            self._use_chroma = True
            if self._collection.count() == 0:
                self.index_documents(KNOWLEDGE_DOCUMENTS)
        except ImportError:
            self._docs = KNOWLEDGE_DOCUMENTS

    def index_documents(self, documents: list[dict[str, Any]]) -> int:
        if not self._use_chroma:
            self._docs = documents
            return len(documents)

        ids, texts, metadatas = [], [], []
        for doc in documents:
            ids.append(doc["id"])
            texts.append(f"{doc['title']}. {doc['content']}")
            metadatas.append({"title": doc["title"], "tags": ",".join(doc.get("tags", []))})

        embeddings = self._embedder.encode(texts).tolist()
        self._collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        return len(ids)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._use_chroma:
            q_emb = self._embedder.encode([query]).tolist()
            results = self._collection.query(query_embeddings=q_emb, n_results=top_k)
            docs = []
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                    "relevance": 1.0 - (results["distances"][0][i] if results.get("distances") else 0.0),
                })
            return docs

        # Keyword fallback
        query_lower = query.lower()
        scored = []
        for doc in self._docs:
            text = f"{doc['title']} {doc['content']} {' '.join(doc.get('tags', []))}".lower()
            score = sum(1 for w in query_lower.split() if w in text) / max(len(query_lower.split()), 1)
            scored.append({**doc, "relevance": score})
        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return scored[:top_k]

    def suggest_parameters(self, ref_sensor: str, mov_sensor: str, sun_diff: float) -> dict[str, Any]:
        query = (
            f"Register {ref_sensor} to {mov_sensor} lunar images with sun angle difference "
            f"{sun_diff:.1f} degrees. Best features, matcher, estimator, thresholds."
        )
        chunks = self.retrieve(query, top_k=3)
        params: dict[str, Any] = {
            "feature_extractors": ["phase_congruency", "contour"],
            "matcher": "semi_dense",
            "geometric_estimator": "magsac",
            "reproj_threshold": 2.0,
            "ratio_threshold": 0.75,
            "rag_context": chunks,
            "rag_mean_relevance": sum(c.get("relevance", 0) for c in chunks) / max(len(chunks), 1),
        }

        combined = " ".join(c.get("content", c.get("title", "")) for c in chunks).lower()
        if "graph" in combined:
            params["geometric_estimator"] = "graph_matching"
        if "spatial" in combined:
            params["secondary_filter"] = "spatial_consistency"
        if "deep" in combined or "iirs" in mov_sensor.lower():
            params["feature_extractors"].append("deep_embedding")
        if sun_diff > 30:
            params["preprocess_illumination"] = True
        if "300" in combined or ("ohrc" in ref_sensor.lower() and "iirs" in mov_sensor.lower()):
            params["hierarchical"] = True

        return params
