"""
data/graph_builder.py

Build PyTorch Geometric HeteroData from SEC EDGAR relationships stored in PostgreSQL.

Graph schema:
  Nodes:  company (CIK), executive (name)
  Edges:
    (executive, employed_at,  company)  — employment relationship
    (executive, board_member, company)  — board membership (most compliance-relevant)
    (executive, traded,       company)  — insider trade (Form 4)
    (executive, co_director,  executive) — shared board (circular ownership signal)
    + reverse edges for bidirectional message passing

Node features:
  company:   [log_employees, log_trades, jurisdiction_risk, log_revenue, num_execs]  5d
  executive: [log_boards, log_trades, avg_trade_value, is_director]                  4d

45,000+ entities across S&P 500 + related executives from EDGAR.
"""

import numpy as np
import psycopg2
import torch
from torch_geometric.data import HeteroData


class GraphBuilder:
    """Builds HeteroData from PostgreSQL EDGAR data."""

    def __init__(self, db_conn):
        self.db = db_conn
        self._node_id_maps: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(self) -> HeteroData:
        """Build full heterogeneous graph from all EDGAR data in PostgreSQL."""
        data = HeteroData()

        # --- Nodes ---
        company_features,   company_ids   = self._load_company_nodes()
        executive_features, executive_ids = self._load_executive_nodes()

        # Store ID maps for edge construction and downstream use
        self._node_id_maps["company"]   = {cik: i for i, cik in enumerate(company_ids)}
        self._node_id_maps["executive"] = {name: i for i, name in enumerate(executive_ids)}

        data["company"].x   = torch.tensor(company_features,   dtype=torch.float)
        data["executive"].x = torch.tensor(executive_features, dtype=torch.float)

        # Store original IDs for feature store push
        data["company"].cik       = company_ids
        data["executive"].name    = executive_ids

        print(f"Graph: {len(company_ids):,} companies, {len(executive_ids):,} executives")

        # --- Edges ---
        data["executive", "employed_at",  "company"].edge_index   = self._load_employment_edges()
        data["executive", "board_member", "company"].edge_index   = self._load_board_edges()
        data["executive", "traded",       "company"].edge_index   = self._load_trade_edges()
        data["executive", "co_director",  "executive"].edge_index = self._load_co_director_edges()

        # Reverse edges — companies aggregate from executives
        data["company", "rev_employed_at",  "executive"].edge_index = \
            self._reverse(data["executive", "employed_at",  "company"].edge_index)
        data["company", "rev_board_member", "executive"].edge_index = \
            self._reverse(data["executive", "board_member", "company"].edge_index)

        for edge_type, store in data.edge_items():
            print(f"  {edge_type}: {store.edge_index.shape[1]:,} edges")

        return data

    # ------------------------------------------------------------------
    # Node loaders
    # ------------------------------------------------------------------

    def _load_company_nodes(self) -> tuple[np.ndarray, list[str]]:
        """
        Company node features:
          0: log(employee_count + 1)  — proxy for company size
          1: log(insider_trades + 1)  — trading activity volume
          2: jurisdiction_risk_score  — derived from state of incorporation
          3: log(revenue + 1)         — financial scale
          4: log(num_executives + 1)  — board/executive network size

        All features log-transformed to reduce skew — employee counts range
        from 1 to 300,000+ which would dominate raw features.
        """
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT
                    c.cik,
                    COALESCE(cf.employees, 0)       AS employees,
                    COALESCE(t.trade_count, 0)      AS trade_count,
                    CASE c.state
                        WHEN 'DE' THEN 0.1   -- Delaware: standard, low risk
                        WHEN 'NV' THEN 0.8   -- Nevada: shell company haven
                        WHEN 'WY' THEN 0.9   -- Wyoming: high risk
                        ELSE 0.3
                    END                             AS jurisdiction_risk,
                    COALESCE(cf.revenue, 0)         AS revenue,
                    COALESCE(e.exec_count, 0)       AS num_executives
                FROM companies c
                LEFT JOIN (
                    SELECT cik,
                           MAX(CAST(facts->>'employees' AS BIGINT)) AS employees,
                           MAX(CAST(facts->>'revenue' AS BIGINT))   AS revenue
                    FROM company_facts GROUP BY cik
                ) cf ON cf.cik = c.cik
                LEFT JOIN (
                    SELECT issuer_cik, COUNT(*) AS trade_count
                    FROM insider_trades GROUP BY issuer_cik
                ) t ON t.issuer_cik = c.cik
                LEFT JOIN (
                    SELECT company_cik, COUNT(DISTINCT executive_id) AS exec_count
                    FROM relationships GROUP BY company_cik
                ) e ON e.company_cik = c.cik
                ORDER BY c.cik
            """)
            rows = cur.fetchall()

        if not rows:
            # Synthetic data for testing
            rows = self._synthetic_company_rows(n=1000)

        ids      = [r[0] for r in rows]
        features = np.array([
            [
                np.log1p(r[1]),  # employees
                np.log1p(r[2]),  # trade_count
                float(r[3]),     # jurisdiction_risk
                np.log1p(r[4]),  # revenue
                np.log1p(r[5]),  # num_executives
            ]
            for r in rows
        ], dtype=np.float32)

        return features, ids

    def _load_executive_nodes(self) -> tuple[np.ndarray, list[str]]:
        """
        Executive node features:
          0: log(boards_count + 1)    — number of boards served on
          1: log(trades_count + 1)    — total insider trade filings
          2: avg_trade_value          — average share value traded (normalized)
          3: is_director              — 1 if director/board member, 0 if officer
        """
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT
                    e.name,
                    COUNT(DISTINCT r.company_cik)        AS boards_count,
                    COALESCE(t.trade_count, 0)           AS trades_count,
                    COALESCE(t.avg_value, 0)             AS avg_trade_value,
                    CASE WHEN e.is_director THEN 1.0 ELSE 0.0 END
                FROM executives e
                LEFT JOIN relationships r ON r.executive_id = e.id
                    AND r.rel_type IN ('board_member', 'director')
                LEFT JOIN (
                    SELECT reporter_cik,
                           COUNT(*) AS trade_count,
                           AVG(shares * price_per_share) AS avg_value
                    FROM insider_trades
                    GROUP BY reporter_cik
                ) t ON t.reporter_cik = e.cik
                GROUP BY e.name, e.is_director, t.trade_count, t.avg_value
                ORDER BY e.name
            """)
            rows = cur.fetchall()

        if not rows:
            rows = self._synthetic_executive_rows(n=5000)

        ids      = [r[0] for r in rows]
        max_val  = max((r[3] for r in rows), default=1.0) or 1.0
        features = np.array([
            [
                np.log1p(r[1]),          # boards_count
                np.log1p(r[2]),          # trades_count
                float(r[3]) / max_val,   # avg_trade_value (normalized)
                float(r[4]),             # is_director
            ]
            for r in rows
        ], dtype=np.float32)

        return features, ids

    # ------------------------------------------------------------------
    # Edge loaders
    # ------------------------------------------------------------------

    def _load_employment_edges(self) -> torch.Tensor:
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT e.name, r.company_cik
                FROM relationships r
                JOIN executives e ON e.id = r.executive_id
                WHERE r.rel_type = 'officer'
            """)
            rows = cur.fetchall()
        return self._edges_from_rows(rows, "executive", "company")

    def _load_board_edges(self) -> torch.Tensor:
        """
        Board membership edges.
        Most compliance-relevant edge type — shared board membership
        is a common undisclosed relationship pattern in SEC enforcement cases.
        """
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT e.name, r.company_cik
                FROM relationships r
                JOIN executives e ON e.id = r.executive_id
                WHERE r.rel_type IN ('board_member', 'director')
            """)
            rows = cur.fetchall()
        return self._edges_from_rows(rows, "executive", "company")

    def _load_trade_edges(self) -> torch.Tensor:
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT e.name, t.issuer_cik
                FROM insider_trades t
                JOIN executives e ON e.cik = t.reporter_cik
            """)
            rows = cur.fetchall()
        return self._edges_from_rows(rows, "executive", "company")

    def _load_co_director_edges(self) -> torch.Tensor:
        """
        Co-director edges: executives who share a board.
        These connect executives across companies — key for detecting
        circular board structures that precede enforcement actions.
        """
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT e1.name, e2.name
                FROM relationships r1
                JOIN relationships r2 ON r1.company_cik = r2.company_cik
                    AND r1.executive_id != r2.executive_id
                JOIN executives e1 ON e1.id = r1.executive_id
                JOIN executives e2 ON e2.id = r2.executive_id
                WHERE r1.rel_type IN ('board_member', 'director')
                  AND r2.rel_type IN ('board_member', 'director')
            """)
            rows = cur.fetchall()
        return self._edges_from_rows(rows, "executive", "executive")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _edges_from_rows(self, rows: list, src_type: str,
                          dst_type: str) -> torch.Tensor:
        src_map = self._node_id_maps.get(src_type, {})
        dst_map = self._node_id_maps.get(dst_type, {})
        src, dst = [], []
        for src_id, dst_id in rows:
            if src_id in src_map and dst_id in dst_map:
                src.append(src_map[src_id])
                dst.append(dst_map[dst_id])
        if not src:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    @staticmethod
    def _reverse(edge_index: torch.Tensor) -> torch.Tensor:
        return edge_index.flip(0)

    # ------------------------------------------------------------------
    # Synthetic data for offline testing (no DB required)
    # ------------------------------------------------------------------

    @staticmethod
    def build_synthetic(n_companies: int = 1000,
                         n_executives: int = 5000) -> HeteroData:
        """
        Build a synthetic HeteroData graph for testing without a DB.
        Node features are random but edge structure mimics EDGAR:
        - ~5 executives per company (employed/board)
        - ~20% of executive pairs share a board (co-director edges)
        """
        rng = np.random.default_rng(42)
        data = HeteroData()

        data["company"].x   = torch.from_numpy(
            rng.standard_normal((n_companies, 5)).astype(np.float32))
        data["executive"].x = torch.from_numpy(
            rng.standard_normal((n_executives, 4)).astype(np.float32))
        data["company"].cik      = [f"CIK{i:07d}" for i in range(n_companies)]
        data["executive"].name   = [f"Exec_{i}" for i in range(n_executives)]

        # Assign ~5 executives per company
        exec_idxs = rng.integers(0, n_companies,    n_executives)
        data["executive", "employed_at",  "company"].edge_index = torch.stack([
            torch.arange(n_executives), torch.from_numpy(exec_idxs)])
        data["executive", "board_member", "company"].edge_index = torch.stack([
            torch.arange(n_executives), torch.from_numpy(exec_idxs)])

        # Trade edges (40% of execs have trades)
        trader_mask = rng.random(n_executives) < 0.4
        trader_idxs = np.where(trader_mask)[0]
        target_idxs = rng.integers(0, n_companies, len(trader_idxs))
        data["executive", "traded", "company"].edge_index = torch.stack([
            torch.from_numpy(trader_idxs), torch.from_numpy(target_idxs)])

        # Co-director edges (executives sharing same board)
        co_src, co_dst = [], []
        for c in range(n_companies):
            execs_at_c = np.where(exec_idxs == c)[0]
            for i in range(len(execs_at_c)):
                for j in range(i + 1, min(i + 3, len(execs_at_c))):
                    co_src.extend([execs_at_c[i], execs_at_c[j]])
                    co_dst.extend([execs_at_c[j], execs_at_c[i]])
        if co_src:
            data["executive", "co_director", "executive"].edge_index = torch.tensor(
                [co_src, co_dst], dtype=torch.long)
        else:
            data["executive", "co_director", "executive"].edge_index = \
                torch.zeros((2, 0), dtype=torch.long)

        # Reverse edges
        data["company", "rev_employed_at",  "executive"].edge_index = \
            data["executive", "employed_at",  "company"].edge_index.flip(0)
        data["company", "rev_board_member", "executive"].edge_index = \
            data["executive", "board_member", "company"].edge_index.flip(0)

        return data

    def _synthetic_company_rows(self, n: int) -> list:
        rng = np.random.default_rng(42)
        states = ["DE", "NV", "WY", "CA", "NY", "TX"]
        return [
            (f"CIK{i:07d}", rng.integers(10, 100000),
             rng.integers(0, 500), rng.choice(states),
             rng.integers(0, 10**9), rng.integers(1, 50))
            for i in range(n)
        ]

    def _synthetic_executive_rows(self, n: int) -> list:
        rng = np.random.default_rng(42)
        return [
            (f"Exec_{i}", rng.integers(1, 5),
             rng.integers(0, 30), rng.random() * 1e6,
             float(rng.random() > 0.5))
            for i in range(n)
        ]
