"""
gnn/trainer.py

Multi-task GNN trainer:
  Task 1 — Node classification: predict company SIC sector from graph structure.
  Task 2 — Link prediction: predict which executives share boards.

Multi-task training produces richer embeddings than either task alone.
The GNN must learn structural patterns that generalize across both tasks —
this forces it to capture genuine network structure rather than overfitting
to a single supervision signal.

After training: embeddings are pushed to the Raft-backed feature store.
Every company and executive gets a 64-dim vector stored with strong consistency.

Usage:
    python gnn/trainer.py --epochs 50 --hidden_dim 128 [--use_synthetic]
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling

from gnn.model import FinancialGNN


class GNNTrainer:
    """
    Trains FinancialGNN on the EDGAR heterogeneous graph.
    Pushes final embeddings to the Raft-backed feature store via gRPC.
    """

    def __init__(
        self,
        feature_store_client=None,  # FeatureStoreClient — optional for offline runs
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hidden_dim: int = 128,
        embed_dim:  int = 64,
    ):
        self.device        = torch.device(device)
        self.feature_store = feature_store_client
        self.model_version = 0

        self.model = FinancialGNN(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5)

        # Linear classifier on top of embeddings (task 1)
        # Trained jointly — gradients flow back through GNN
        self.sector_head = nn.Linear(embed_dim, 10).to(self.device)  # 10 SIC sectors

        print(f"GNN: {self.model}")

    def train(self, graph: HeteroData, epochs: int = 100,
               checkpoint_dir: str = "checkpoints") -> dict:
        """
        Full training on complete EDGAR graph.

        Returns dict of final metrics.
        """
        Path(checkpoint_dir).mkdir(exist_ok=True)
        graph = graph.to(self.device)

        best_loss = float("inf")
        history   = []

        for epoch in range(epochs):
            t0 = time.perf_counter()
            self.model.train()
            self.optimizer.zero_grad()

            embeddings = self.model(graph.x_dict, graph.edge_index_dict)

            cls_loss = self._classification_loss(embeddings, graph)
            lp_loss  = self._link_prediction_loss(embeddings, graph)
            loss     = cls_loss + 0.5 * lp_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            step_ms = (time.perf_counter() - t0) * 1000
            history.append({
                "epoch":    epoch,
                "loss":     loss.item(),
                "cls_loss": cls_loss.item(),
                "lp_loss":  lp_loss.item(),
                "step_ms":  step_ms,
            })

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | loss={loss.item():.4f} | "
                      f"cls={cls_loss.item():.4f} | lp={lp_loss.item():.4f} | "
                      f"{step_ms:.0f}ms")

            # Save best checkpoint
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    "epoch":       epoch,
                    "model":       self.model.state_dict(),
                    "optimizer":   self.optimizer.state_dict(),
                    "loss":        loss.item(),
                    "model_version": self.model_version,
                }, f"{checkpoint_dir}/best_model.pt")

        # Push final embeddings to feature store
        self.model_version += 1
        n_pushed = self._push_embeddings(graph)
        print(f"\nPushed {n_pushed:,} embeddings to feature store "
              f"(version {self.model_version})")

        return {"final_loss": best_loss, "epochs": epochs, "history": history}

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _classification_loss(self, embeddings: dict,
                               graph: HeteroData) -> torch.Tensor:
        """
        Task 1: predict company SIC sector from company embeddings.
        Uses first 2 digits of SIC code as sector label (10 sectors).

        If no labels on graph (synthetic data), returns 0.
        """
        if not hasattr(graph["company"], "y"):
            return torch.tensor(0.0, device=self.device)

        company_emb = embeddings["company"]
        labels      = graph["company"].y.to(self.device)

        logits = self.sector_head(company_emb)
        return F.cross_entropy(logits, labels)

    def _link_prediction_loss(self, embeddings: dict,
                               graph: HeteroData) -> torch.Tensor:
        """
        Task 2: predict co-director relationships from executive embeddings.

        Positive pairs:  actual co-director edges in the graph.
        Negative pairs:  randomly sampled executive pairs.

        Binary cross-entropy pushes positive pair scores → 1,
        negative pair scores → 0.

        Why link prediction for compliance?
        Co-director relationships are the key structural signal.
        Making the GNN predict them forces it to learn the embedding
        space where structurally connected executives cluster together.
        """
        exec_emb   = embeddings["executive"]
        edge_index = graph["executive", "co_director", "executive"].edge_index

        if exec_emb.size(0) < 2 or edge_index.size(1) == 0:
            return torch.tensor(0.0, device=self.device)

        # Positive scores
        src_emb    = exec_emb[edge_index[0]]
        dst_emb    = exec_emb[edge_index[1]]
        pos_scores = (src_emb * dst_emb).sum(dim=-1)  # dot product (cosine since L2-normed)

        # Negative sampling — random executive pairs
        neg_index  = negative_sampling(
            edge_index, num_nodes=exec_emb.size(0),
            num_neg_samples=edge_index.size(1))
        neg_src    = exec_emb[neg_index[0]]
        neg_dst    = exec_emb[neg_index[1]]
        neg_scores = (neg_src * neg_dst).sum(dim=-1)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores))

        return (pos_loss + neg_loss) / 2.0

    # ------------------------------------------------------------------
    # Feature store push
    # ------------------------------------------------------------------

    def _push_embeddings(self, graph: HeteroData) -> int:
        """Push all node embeddings to the Raft-backed feature store."""
        if not self.feature_store:
            print("[warn] No feature store client — skipping push.")
            return 0

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(graph.x_dict, graph.edge_index_dict)

        count = 0

        # Company embeddings
        company_embs = embeddings["company"].cpu().numpy()
        company_ids  = graph["company"].cik
        for i, cik in enumerate(company_ids):
            self.feature_store.set_embedding(
                namespace="company_embeddings",
                entity_id=cik,
                embedding=company_embs[i].tolist(),
                model_version=self.model_version,
            )
            count += 1

        # Executive embeddings
        exec_embs = embeddings["executive"].cpu().numpy()
        exec_names = graph["executive"].name
        for i, name in enumerate(exec_names):
            self.feature_store.set_embedding(
                namespace="executive_embeddings",
                entity_id=name,
                embedding=exec_embs[i].tolist(),
                model_version=self.model_version,
            )
            count += 1

        return count

    # ------------------------------------------------------------------
    # Incremental push (after online retraining)
    # ------------------------------------------------------------------

    def push_subset(self, graph: HeteroData, affected_ids: list[str]) -> int:
        """Push embeddings only for a subset of affected nodes."""
        if not self.feature_store:
            return 0

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(graph.x_dict, graph.edge_index_dict)

        affected_set = set(affected_ids)
        count = 0

        company_embs = embeddings["company"].cpu().numpy()
        for i, cik in enumerate(graph["company"].cik):
            if cik in affected_set:
                self.feature_store.set_embedding(
                    "company_embeddings", cik,
                    company_embs[i].tolist(), self.model_version)
                count += 1

        exec_embs = embeddings["executive"].cpu().numpy()
        for i, name in enumerate(graph["executive"].name):
            if name in affected_set:
                self.feature_store.set_embedding(
                    "executive_embeddings", name,
                    exec_embs[i].tolist(), self.model_version)
                count += 1

        return count


# Missing import needed for sector_head
import torch.nn as nn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,  default=50)
    parser.add_argument("--hidden_dim",   type=int,  default=128)
    parser.add_argument("--embed_dim",    type=int,  default=64)
    parser.add_argument("--use_synthetic",action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    from data.graph_builder import GraphBuilder

    if args.use_synthetic:
        print("Building synthetic graph...")
        graph = GraphBuilder.build_synthetic(n_companies=1000, n_executives=5000)
    else:
        import psycopg2
        db = psycopg2.connect(os.environ.get("DATABASE_URL",
                              "postgresql://localhost/edgar"))
        graph = GraphBuilder(db).build()
        db.close()

    trainer = GNNTrainer(
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
    )
    metrics = trainer.train(graph, epochs=args.epochs,
                             checkpoint_dir=args.checkpoint_dir)
    print(f"\nTraining complete. Best loss: {metrics['final_loss']:.4f}")
