"""
rag/retriever.py + rag/synthesizer.py

Hybrid RAG pipeline combining:
  1. GNN structural similarity (FAISS over feature store embeddings)
     → finds companies with similar positions in the ownership/board graph
  2. BM25 + dense semantic search (OpenSearch) over filing text
     → finds relevant filing language (risk factors, related party disclosures)

Results combined via Reciprocal Rank Fusion (RRF):
  score(d) = 1/(k + rank_gnn(d)) + 1/(k + rank_semantic(d))
  k=60 smoothing constant (standard from the RRF paper)

Why RRF over learned fusion?
  Simple, no training needed, no distribution shift.
  Works well when the two ranking lists have complementary signals.
  GNN finds structurally similar companies; semantic finds relevant language.
  These are complementary — RRF makes sense here.
"""

import anthropic
import numpy as np
import faiss
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from typing import Optional

from feature_store.client import FeatureStoreClient


# ------------------------------------------------------------------
# Hybrid Retriever
# ------------------------------------------------------------------

class HybridRetriever:
    """
    Two-signal retriever: GNN structural + BM25/dense semantic.

    Build once on startup:
      - Load all company embeddings from feature store into FAISS index
      - Connect to OpenSearch for filing text search

    Query time:
      - GNN retrieval: O(log N) FAISS ANN search (~0.1ms for 45k entities)
      - Semantic retrieval: OpenSearch hybrid query (~10-50ms)
      - RRF fusion: O(k) — negligible
    """

    def __init__(
        self,
        feature_store_client: FeatureStoreClient,
        opensearch_host: str = "localhost:9200",
    ):
        self.feature_store = feature_store_client
        self.os_client     = OpenSearch(hosts=[opensearch_host])
        self.encoder       = SentenceTransformer("all-MiniLM-L6-v2")

        # FAISS index built from feature store embeddings
        # Rebuilt on startup and refreshed periodically (every 5 min)
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.entity_ids:  list[str] = []

        self._build_faiss_index()

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    def _build_faiss_index(self) -> None:
        """Load all company embeddings into FAISS for ANN search."""
        # In production: paginate through feature store's entity list
        # Here: batch fetch a known set of company CIKs
        company_ids = self.feature_store.list_entities("company_embeddings")
        if not company_ids:
            print("[retriever] No embeddings in feature store — FAISS index empty")
            return

        embeddings = self.feature_store.batch_get_embeddings(
            "company_embeddings", company_ids)

        if not embeddings:
            return

        self.entity_ids = [cik for cik in company_ids if cik in embeddings]
        matrix = np.vstack([embeddings[cik] for cik in self.entity_ids]).astype(np.float32)

        # IndexFlatIP: exact inner product search on L2-normalized vectors = cosine similarity.
        # For 45k entities × 64 dims: 11MB index, < 1ms search.
        # At 1M entities: use IndexHNSWFlat for O(log N) ANN.
        self.faiss_index = faiss.IndexFlatIP(matrix.shape[1])
        self.faiss_index.add(matrix)
        print(f"[retriever] FAISS index: {self.faiss_index.ntotal:,} companies, "
              f"dim={matrix.shape[1]}")

    def refresh_index(self) -> None:
        """Refresh FAISS index with latest embeddings from feature store."""
        self._build_faiss_index()

    # ------------------------------------------------------------------
    # Main retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query:     str,
        entity_id: Optional[str] = None,   # CIK for structural search
        k:         int = 5,
    ) -> list[dict]:
        """
        Hybrid retrieval.

        If entity_id provided: structural similarity via FAISS + semantic filing search.
        Query only: semantic + keyword search over filing corpus.

        Returns top-k results ranked by RRF score.
        """
        raw_results = []

        # GNN structural retrieval
        if entity_id and self.faiss_index is not None:
            gnn_results = self._gnn_retrieve(entity_id, k=k * 2)
            raw_results.extend(gnn_results)

        # Hybrid BM25 + dense semantic retrieval
        semantic_results = self._hybrid_retrieve(query, k=k * 2)
        raw_results.extend(semantic_results)

        # Reciprocal Rank Fusion
        return self._rrf_combine(raw_results, k=k)

    # ------------------------------------------------------------------
    # GNN structural retrieval
    # ------------------------------------------------------------------

    def _gnn_retrieve(self, entity_id: str, k: int = 10) -> list[dict]:
        """Find structurally similar companies via FAISS."""
        embedding = self.feature_store.get_embedding("company_embeddings", entity_id)
        if embedding is None or self.faiss_index is None:
            return []

        query_vec = embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_index.search(query_vec, k + 1)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.entity_ids):
                continue
            cik = self.entity_ids[idx]
            if cik == entity_id:   # exclude self
                continue
            results.append({
                "type":   "gnn_structural",
                "cik":    cik,
                "score":  float(score),
                "source": "faiss",
                # Fetch filing text for this similar company for RAG context
                "context": f"Structurally similar company (cosine={score:.3f})",
            })

        return results[:k]

    # ------------------------------------------------------------------
    # Hybrid BM25 + dense retrieval
    # ------------------------------------------------------------------

    def _hybrid_retrieve(self, query: str, k: int = 10) -> list[dict]:
        """
        OpenSearch hybrid query: BM25 keyword + kNN dense vector.

        BM25 strength: exact term matching (restatement, material weakness,
          Section 16 violation) — legal language is precise.
        Dense strength: semantic similarity — "executive changed" matches
          "officer resigned" even without shared terms.
        Combined: neither alone is sufficient for compliance queries.
        """
        query_embedding = self.encoder.encode(query).tolist()

        os_query = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "chunk_text": {
                                    "query": query,
                                    "boost": 1.0,
                                }
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k":      k,
                                    "boost":  1.0,
                                }
                            }
                        },
                    ]
                }
            },
            "_source": ["chunk_text", "company_cik", "company_name",
                        "form_type", "filed_at", "chunk_type"],
        }

        try:
            response = self.os_client.search(index="edgar_filings", body=os_query)
        except Exception as e:
            print(f"[retriever] OpenSearch error: {e}")
            return []

        results = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            results.append({
                "type":         "semantic",
                "chunk_text":   src.get("chunk_text", ""),
                "company_cik":  src.get("company_cik", ""),
                "company_name": src.get("company_name", ""),
                "form_type":    src.get("form_type", ""),
                "filed_at":     src.get("filed_at", ""),
                "chunk_type":   src.get("chunk_type", ""),
                "score":        hit["_score"],
                "source":       "opensearch",
            })

        return results

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _rrf_combine(self, results: list[dict], k: int = 5,
                     rrf_k: int = 60) -> list[dict]:
        """
        Reciprocal Rank Fusion: score(d) = sum_i 1/(k + rank_i(d))

        rrf_k=60 is the standard from Cormack et al. 2009.
        Higher rrf_k → smooths rank differences, lower → rewards top ranks more.
        60 is the empirical sweet spot across retrieval benchmarks.

        RRF advantage over score normalization:
          Scores from FAISS (cosine 0-1) and OpenSearch (BM25 unbounded) are
          not directly comparable. RRF uses only ranks, not raw scores, so
          it's robust to score scale differences between retrievers.
        """
        # Assign rank within each source separately
        gnn_results  = [r for r in results if r["type"] == "gnn_structural"]
        sem_results  = [r for r in results if r["type"] == "semantic"]

        # Sort each list by score (descending)
        gnn_results.sort(key=lambda x: -x["score"])
        sem_results.sort(key=lambda x: -x["score"])

        # Build rank maps
        gnn_ranks = {r["cik"]:  i for i, r in enumerate(gnn_results)}
        sem_ranks = {r.get("company_cik", r.get("cik", "")): i
                     for i, r in enumerate(sem_results)}

        # Collect all unique documents
        all_docs: dict[str, dict] = {}
        for r in gnn_results + sem_results:
            doc_id = r.get("cik") or r.get("company_cik", "")
            if doc_id and doc_id not in all_docs:
                all_docs[doc_id] = r

        # Compute RRF scores
        scored = []
        for doc_id, doc in all_docs.items():
            rrf_score = 0.0
            if doc_id in gnn_ranks:
                rrf_score += 1.0 / (rrf_k + gnn_ranks[doc_id])
            if doc_id in sem_ranks:
                rrf_score += 1.0 / (rrf_k + sem_ranks[doc_id])
            scored.append((rrf_score, doc))

        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:k]]


# ------------------------------------------------------------------
# RAG Synthesizer
# ------------------------------------------------------------------

class ComplianceRAGSynthesizer:
    """
    Synthesize compliance risk assessments from retrieved context.
    Uses Claude API with retrieved filing text + structural similarity context.
    """

    SYSTEM_PROMPT = """You are a financial compliance analyst with deep expertise in
SEC EDGAR filings. You analyze corporate networks, insider trading patterns, and
regulatory filings to assess compliance risk.

When given retrieved filing text and structural similarity information, synthesize
a concise, grounded compliance risk assessment. Always cite specific filing evidence.
Do not make claims beyond what the retrieved evidence supports.
"""

    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(api_key=api_key)

    def synthesize(
        self,
        query:    str,
        results:  list[dict],
        entity_id: Optional[str] = None,
    ) -> str:
        """
        Generate a compliance risk assessment from retrieved context.

        Context is assembled from:
          - GNN structural similarity (network position of similar companies)
          - Filing text chunks (risk factors, related party transactions, MDA)
        """
        context_parts = []

        for i, r in enumerate(results, 1):
            if r["type"] == "gnn_structural":
                context_parts.append(
                    f"[{i}] Structurally similar company (CIK: {r['cik']}): "
                    f"cosine similarity {r['score']:.3f} in executive/ownership network."
                )
            elif r["type"] == "semantic":
                context_parts.append(
                    f"[{i}] {r.get('form_type', 'Filing')} ({r.get('filed_at', '')[:10]}) "
                    f"— {r.get('company_name', 'Unknown')} "
                    f"[{r.get('chunk_type', 'general')}]:\n"
                    f"{r.get('chunk_text', '')[:500]}"
                )

        context = "\n\n".join(context_parts)

        user_message = f"""Query: {query}

{"Entity under analysis: " + entity_id if entity_id else ""}

Retrieved Evidence:
{context}

Based on the above evidence, provide a compliance risk assessment. Include:
1. Risk level (Low / Medium / High) with justification
2. Key structural risk signals (if any structural similarity context provided)
3. Specific filing language that supports the assessment
4. Recommended follow-up actions"""

        message = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return message.content[0].text


# ------------------------------------------------------------------
# OpenSearch indexer
# ------------------------------------------------------------------

class FilingIndexer:
    """
    Indexes EDGAR filing text chunks in OpenSearch for hybrid retrieval.
    kNN field for dense search + text field for BM25.
    """

    INDEX_NAME = "edgar_filings"
    INDEX_SETTINGS = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "chunk_text":       {"type": "text", "analyzer": "english"},
                "chunk_type":       {"type": "keyword"},
                "accession_number": {"type": "keyword"},
                "company_cik":      {"type": "keyword"},
                "company_name":     {"type": "text"},
                "form_type":        {"type": "keyword"},
                "filed_at":         {"type": "date"},
                "embedding": {
                    "type":      "knn_vector",
                    "dimension": 384,
                    "method": {
                        "name":       "hnsw",
                        "space_type": "cosinesimil",
                        "engine":     "nmslib",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
            }
        },
    }

    def __init__(self, opensearch_host: str = "localhost:9200"):
        self.client  = OpenSearch(hosts=[opensearch_host])
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._ensure_index()

    def _ensure_index(self):
        if not self.client.indices.exists(self.INDEX_NAME):
            self.client.indices.create(
                index=self.INDEX_NAME, body=self.INDEX_SETTINGS)
            print(f"Created OpenSearch index: {self.INDEX_NAME}")

    def index_chunk(self, chunk: dict, company_info: dict):
        embedding = self.encoder.encode(chunk["chunk_text"]).tolist()
        doc = {
            "chunk_text":       chunk["chunk_text"],
            "chunk_type":       chunk.get("chunk_type", "general"),
            "accession_number": chunk["accession_number"],
            "company_cik":      company_info.get("cik", ""),
            "company_name":     company_info.get("name", ""),
            "form_type":        chunk.get("form_type", ""),
            "filed_at":         chunk.get("filed_at", ""),
            "embedding":        embedding,
        }
        doc_id = f"{chunk['accession_number']}_{chunk.get('chunk_index', 0)}"
        self.client.index(index=self.INDEX_NAME, body=doc, id=doc_id)

    def bulk_index(self, chunks: list[dict], company_info: dict,
                   batch_size: int = 100):
        """Batch index using OpenSearch bulk API."""
        from opensearchpy import helpers

        def _gen():
            for chunk in chunks:
                embedding = self.encoder.encode(chunk["chunk_text"]).tolist()
                yield {
                    "_index": self.INDEX_NAME,
                    "_id":    f"{chunk['accession_number']}_{chunk.get('chunk_index',0)}",
                    "_source": {
                        "chunk_text":       chunk["chunk_text"],
                        "chunk_type":       chunk.get("chunk_type", "general"),
                        "accession_number": chunk["accession_number"],
                        "company_cik":      company_info.get("cik", ""),
                        "company_name":     company_info.get("name", ""),
                        "embedding":        embedding,
                    },
                }

        success, failed = helpers.bulk(self.client, _gen(), chunk_size=batch_size)
        print(f"Indexed {success} chunks ({failed} failed)")
        return success
