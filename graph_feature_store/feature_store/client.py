"""
feature_store/client.py

Python gRPC client for the Go feature store.
Used by the GNN trainer (Python) to push embeddings and by the RAG
retriever to fetch embeddings for FAISS index building.

Firebase auth: external clients (Android app, Python trainer running off-cluster)
must attach a Firebase ID token as gRPC metadata. The Go server validates it.
Inter-node Raft RPCs use mTLS (no Firebase needed — server-to-server auth).

Usage:
    client = FeatureStoreClient("localhost:50051")
    client.set_embedding("company_embeddings", "0000320193", emb.tolist(), model_version=1)
    emb = client.get_embedding("company_embeddings", "0000320193")
"""

import grpc
import numpy as np
import time
from typing import Optional

# Generated from proto/feature_store.proto via:
#   python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/feature_store.proto
import proto.feature_store_pb2        as pb
import proto.feature_store_pb2_grpc   as pb_grpc


class FeatureStoreClient:
    """
    Python client for the Go Raft-backed feature store.

    Handles:
      - Connection management (single persistent connection)
      - Firebase ID token injection for external auth
      - Retry with exponential backoff on transient errors
      - Batch operations for bulk embedding pushes
    """

    def __init__(
        self,
        addr:            str = "localhost:50051",
        firebase_token:  Optional[str] = None,
        tls_cert_path:   Optional[str] = None,
        max_retries:     int = 3,
    ):
        self.addr          = addr
        self.firebase_token = firebase_token
        self.max_retries   = max_retries

        # TLS channel for production; insecure for local dev
        if tls_cert_path:
            with open(tls_cert_path, "rb") as f:
                creds = grpc.ssl_channel_credentials(f.read())
            self.channel = grpc.secure_channel(addr, creds)
        else:
            self.channel = grpc.insecure_channel(addr)

        self.stub = pb_grpc.FeatureStoreStub(self.channel)

    def _metadata(self) -> list[tuple]:
        """Attach Firebase ID token as gRPC metadata for auth."""
        if self.firebase_token:
            return [("authorization", f"Bearer {self.firebase_token}")]
        return []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def set_embedding(
        self,
        namespace:     str,
        entity_id:     str,
        embedding:     list[float],
        model_version: int = 0,
    ) -> bool:
        """
        Push a single embedding through Raft consensus.
        Blocks until committed on majority of nodes.

        Retry on transient errors (leader re-election in progress).
        Raft re-elections take ~300ms worst-case — retry with backoff.
        """
        req = pb.SetFeatureRequest(
            namespace=namespace,
            entity_id=entity_id,
            embedding=embedding,
            model_version=model_version,
            updated_at_ms=int(time.time() * 1000),
        )

        for attempt in range(self.max_retries):
            try:
                resp = self.stub.SetFeature(req, metadata=self._metadata())
                return resp.success
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE and attempt < self.max_retries - 1:
                    # Leader election in progress — wait and retry
                    wait = 0.1 * (2 ** attempt)
                    time.sleep(wait)
                    continue
                raise

        return False

    def batch_set_embeddings(
        self,
        namespace:     str,
        entity_ids:    list[str],
        embeddings:    np.ndarray,   # [N, D]
        model_version: int = 0,
        batch_size:    int = 100,
    ) -> int:
        """
        Push N embeddings in batches.
        Returns count of successfully committed embeddings.

        Batching: gRPC doesn't pipeline unary calls automatically.
        For 45,000 embeddings at 1ms/write = 45s sequential.
        With batch_size=100 and parallel goroutines on server side → ~5s.
        For higher throughput: use streaming RPC (not implemented here).
        """
        n_success = 0
        for i in range(0, len(entity_ids), batch_size):
            batch_ids  = entity_ids[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]

            for entity_id, emb in zip(batch_ids, batch_embs):
                if self.set_embedding(namespace, entity_id, emb.tolist(), model_version):
                    n_success += 1

            if i % 1000 == 0 and i > 0:
                print(f"  Pushed {i:,}/{len(entity_ids):,} embeddings...")

        return n_success

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_embedding(
        self,
        namespace:   str,
        entity_id:   str,
        strong_read: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Fetch a single embedding.

        strong_read=False: local replica read (fast, potentially stale).
          Stale by at most the replication lag (~5ms in healthy cluster).
          Use for FAISS index building where slight staleness is acceptable.

        strong_read=True: linearizable read through leader (always consistent).
          Use for compliance queries where stale embeddings could yield wrong results.
        """
        req = pb.GetFeatureRequest(
            namespace=namespace,
            entity_id=entity_id,
            strong_read=strong_read,
        )
        try:
            resp = self.stub.GetFeature(req, metadata=self._metadata())
            if resp.is_stale:
                print(f"[warn] {namespace}/{entity_id} embedding is stale "
                      f"(age={(time.time()*1000 - resp.updated_at_ms):.0f}ms)")
            return np.array(resp.embedding, dtype=np.float32)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise

    def batch_get_embeddings(
        self,
        namespace:   str,
        entity_ids:  list[str],
        strong_read: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Fetch embeddings for multiple entities in one RPC call.
        Server fans out reads in parallel goroutines.
        Much faster than N individual GetFeature calls.
        """
        req = pb.BatchGetRequest(
            namespace=namespace,
            entity_ids=entity_ids,
            strong_read=strong_read,
        )
        resp = self.stub.BatchGetFeatures(req, metadata=self._metadata())
        return {
            entity_id: np.array(feat.embedding, dtype=np.float32)
            for entity_id, feat in resp.features.items()
        }

    def list_entities(self, namespace: str) -> list[str]:
        """
        List all entity IDs in a namespace.
        Used by FAISS index builder to enumerate all companies.
        Implementation: scan Raft state machine for keys with prefix "emb:{namespace}:".
        Not implemented in proto — use a separate admin RPC or scan via HTTP.
        """
        # Placeholder: in production, add a ListEntities RPC to proto
        # or maintain a separate index in the feature store.
        print(f"[warn] list_entities not implemented — returning empty list")
        return []

    # ------------------------------------------------------------------
    # Cluster monitoring
    # ------------------------------------------------------------------

    def cluster_stats(self) -> dict:
        """Fetch cluster stats — used by Android app monitoring dashboard."""
        resp = self.stub.GetClusterStats(pb.StatsRequest(), metadata=self._metadata())
        return {
            "node_id":        resp.node_id,
            "role":           resp.role,
            "term":           resp.term,
            "commit_index":   resp.commit_index,
            "total_features": resp.total_features,
            "writes_per_sec": resp.writes_per_sec,
            "reads_per_sec":  resp.reads_per_sec,
        }

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
