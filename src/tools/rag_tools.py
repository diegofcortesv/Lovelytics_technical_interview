"""
RAG retrieval tools for the Fraud Copilot agent.

Supports two backends:
1. Databricks Vector Search (primary) — embedding-based similarity search
2. TF-IDF fallback — keyword-based retrieval when Vector Search is unavailable

The tool automatically detects Vector Search availability and falls back
to TF-IDF if the index is not ready or the endpoint is unreachable.
"""

import mlflow
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGTool:
    """Financial document search with dual-backend retrieval."""

    def __init__(
        self,
        chunks_df: pd.DataFrame,
        vs_endpoint_name: Optional[str] = None,
        vs_index_name: Optional[str] = None,
    ):
        self.chunks_df = chunks_df
        self.use_vector_search = False
        self.vs_index = None

        # Build TF-IDF index (always available as fallback)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.chunk_vectors = self.vectorizer.fit_transform(
            chunks_df["content"].tolist()
        )

        # Try to connect to Vector Search
        if vs_endpoint_name and vs_index_name:
            self._try_connect_vs(vs_endpoint_name, vs_index_name)

    def _try_connect_vs(self, endpoint_name: str, index_name: str) -> None:
        """Attempt to connect to Databricks Vector Search."""
        try:
            from databricks.vector_search.client import VectorSearchClient

            client = VectorSearchClient()
            self.vs_index = client.get_index(
                endpoint_name=endpoint_name,
                index_name=index_name,
            )
            status = self.vs_index.describe().get("status", {}).get("ready", False)
            if status:
                self.use_vector_search = True
        except Exception:
            pass  # Fall back to TF-IDF silently

    @mlflow.trace(span_type="RETRIEVER", name="search_financial_docs")
    def search(self, query: str, top_k: int = 5) -> dict:
        """Search financial knowledge base with citations."""
        if self.use_vector_search:
            result = self._search_vector(query, top_k)
            if result is not None:
                return result

        return self._search_tfidf(query, top_k)

    def _search_vector(self, query: str, top_k: int) -> Optional[dict]:
        """Search via Databricks Vector Search."""
        try:
            results_raw = self.vs_index.similarity_search(
                query_text=query,
                columns=["chunk_id", "content", "source", "doc_name"],
                num_results=top_k,
            )

            results = []
            sources = set()
            for row in results_raw.get("result", {}).get("data_array", []):
                content = str(row[1])[:1500]
                source = str(row[2])
                score = float(row[-1]) if len(row) > 4 else 0.0
                results.append({"text": content, "source": source, "score": score})
                sources.add(source)

            context = "\n\n---\n\n".join(
                [f"[Source: {r['source']}]\n{r['text']}" for r in results]
            )
            return {
                "answer_context": context,
                "sources": list(sources),
                "num_results": len(results),
                "backend": "vector_search",
            }
        except Exception:
            return None  # Fall through to TF-IDF

    def _search_tfidf(self, query: str, top_k: int) -> dict:
        """Fallback search via TF-IDF cosine similarity."""
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.chunk_vectors).flatten()
        top_idx = np.argsort(sims)[-top_k:][::-1]

        results = []
        sources = set()
        for idx in top_idx:
            if sims[idx] > 0.05:
                row = self.chunks_df.iloc[idx]
                results.append({
                    "text": row["content"][:1500],
                    "source": row["source"],
                    "score": float(sims[idx]),
                })
                sources.add(row["source"])

        context = "\n\n---\n\n".join(
            [f"[Source: {r['source']}]\n{r['text']}" for r in results]
        )
        return {
            "answer_context": context,
            "sources": list(sources),
            "num_results": len(results),
            "backend": "tfidf",
        }


# Module-level convenience function

_rag_tool: Optional[RAGTool] = None


def init_rag_tool(
    chunks_df: pd.DataFrame,
    vs_endpoint_name: Optional[str] = None,
    vs_index_name: Optional[str] = None,
) -> None:
    """Initialize the RAG tool."""
    global _rag_tool
    _rag_tool = RAGTool(chunks_df, vs_endpoint_name, vs_index_name)


def search_financial_docs(query: str, top_k: int = 5) -> dict:
    if _rag_tool is None:
        raise RuntimeError("Call init_rag_tool() first")
    return _rag_tool.search(query, top_k)
