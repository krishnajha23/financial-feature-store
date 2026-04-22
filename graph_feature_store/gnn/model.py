"""
gnn/model.py

Heterogeneous GNN for financial relationship graphs.

Architecture: 3-layer HeteroConv
  Layer 1: GAT (Graph Attention Network) — learns which relationships matter most.
           Not all edges are equally informative for compliance risk.
           Board membership from a Delaware shell → higher attention weight.
  Layer 2: GraphSAGE — wider neighborhood aggregation, more stable gradients.
  Layer 3: GraphSAGE → 64-dim final embeddings (L2 normalized for FAISS search).

Why GAT in layer 1?
  The attention mechanism learns per-edge importance weights.
  A company's risk is more influenced by its board connections than
  random employment relationships. GAT discovers this automatically.

Why SAGE in layers 2-3?
  SAGEConv is more memory efficient than GAT for deep message passing.
  It concatenates self-features with aggregated neighbor mean, which
  is more stable than attention-weighted sum at deeper layers.

Output: L2-normalized 64-dim embeddings for company and executive nodes.
These are stored in the Raft-backed feature store and searched via FAISS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv
from torch_geometric.data import HeteroData


class FinancialGNN(nn.Module):
    """
    Heterogeneous GNN on SEC EDGAR financial relationship graphs.

    Input:
      company nodes:   [N_c, 5]  (log_employees, log_trades, juris_risk, log_rev, log_execs)
      executive nodes: [N_e, 4]  (log_boards, log_trades, avg_trade_val, is_director)

    Output:
      {'company': [N_c, 64], 'executive': [N_e, 64]}  — L2 normalized embeddings

    GNN beats XGBoost because it propagates risk across the graph:
      company A looks legitimate alone → but 2-hop neighbors are shell companies
      → GNN embedding absorbs neighbor risk → correct high-risk classification.
    XGBoost treats each node independently — it sees node features but not
    this 2-hop structural context.
    """

    EDGE_TYPES_L1 = [
        ("executive", "employed_at",    "company"),
        ("executive", "board_member",   "company"),
        ("executive", "traded",         "company"),
        ("executive", "co_director",    "executive"),
        ("company",   "rev_employed_at",  "executive"),
        ("company",   "rev_board_member", "executive"),
    ]

    EDGE_TYPES_L2 = EDGE_TYPES_L1  # same set

    EDGE_TYPES_L3 = [
        ("executive", "employed_at",    "company"),
        ("executive", "board_member",   "company"),
        ("executive", "co_director",    "executive"),
        ("company",   "rev_employed_at",  "executive"),
    ]

    def __init__(
        self,
        company_feat_dim: int = 5,
        exec_feat_dim:    int = 4,
        hidden_dim:       int = 128,
        embed_dim:        int = 64,
        num_heads:        int = 4,
        dropout:          float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout   = nn.Dropout(dropout)

        # Input projections — map raw features to hidden_dim
        # Without these, node types with different feature dims can't share layers.
        self.company_proj   = nn.Linear(company_feat_dim, hidden_dim)
        self.executive_proj = nn.Linear(exec_feat_dim,    hidden_dim)

        head_dim = hidden_dim // num_heads  # output per attention head

        # Layer 1: Multi-head GAT — learns edge importance weights
        self.conv1 = HeteroConv({
            et: GATConv(hidden_dim, head_dim, heads=num_heads,
                        dropout=dropout, add_self_loops=False)
            for et in self.EDGE_TYPES_L1
        }, aggr="sum")

        # Layer 2: GraphSAGE — wider, more stable aggregation
        self.conv2 = HeteroConv({
            et: SAGEConv(hidden_dim, hidden_dim)
            for et in self.EDGE_TYPES_L2
        }, aggr="sum")

        # Layer 3: Project to embedding space
        self.conv3 = HeteroConv({
            et: SAGEConv(hidden_dim, embed_dim)
            for et in self.EDGE_TYPES_L3
        }, aggr="sum")

    def forward(self, x_dict: dict[str, torch.Tensor],
                edge_index_dict: dict) -> dict[str, torch.Tensor]:
        """
        Forward pass through 3 GNN layers.

        Args:
            x_dict:          {'company': [N_c, 5], 'executive': [N_e, 4]}
            edge_index_dict: {edge_type: [2, E]}

        Returns:
            {'company': [N_c, 64], 'executive': [N_e, 64]}  — L2 normalized
        """
        # Project raw features to shared hidden_dim
        x_dict = {
            "company":   F.relu(self.company_proj(x_dict["company"])),
            "executive": F.relu(self.executive_proj(x_dict["executive"])),
        }

        # Layer 1: GAT with dropout
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict.items()}

        # Layer 2: SAGE
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict.items()}

        # Layer 3: final embeddings
        x_dict = self.conv3(x_dict, edge_index_dict)

        # L2 normalize — required for cosine similarity search via FAISS IndexFlatIP
        x_dict = {k: F.normalize(v, p=2, dim=-1) for k, v in x_dict.items()}

        return x_dict

    def embed(self, graph: HeteroData, device: torch.device = None) -> dict:
        """Convenience: embed a full graph."""
        if device:
            graph = graph.to(device)
        self.eval()
        with torch.no_grad():
            return self.forward(graph.x_dict, graph.edge_index_dict)

    def extra_repr(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"params={total:,}, embed_dim={self.embed_dim}"
