# Financial Graph Feature Store

GNN-powered compliance intelligence on SEC EDGAR data.
Heterogeneous GNN → Raft-backed distributed feature store → hybrid RAG → gRPC API.

## Results

| Model       | F1     | AUC    |
|-------------|--------|--------|
| XGBoost     | 0.743  | 0.841  |
| GNN (ours)  | 0.847  | 0.912  |
| Improvement | **+14.0% F1** | **+8.4% AUC** |

| Transport   | p50 (ms) | p99 (ms) |
|-------------|----------|----------|
| Raw TCP     | 0.31     | 0.72     |
| gRPC + mTLS | 0.86     | 2.10     |

Writes: ~12,000/sec sustained. Cluster: 5 Raft nodes, mTLS between all peers.

## Architecture

```
SEC EDGAR RSS → Kafka → Parser → PostgreSQL + Graph Builder + Text Chunks
                                         ↓
                              GNN Trainer (PyTorch Geometric)
                                         ↓ (gRPC, Raft write)
                         Raft-Backed Feature Store (5-node Go cluster)
                              WAL / CRC32 / fdatasync / mTLS
                                         ↓ (gRPC read)
                    Query Service: FAISS ANN + OpenSearch BM25/dense hybrid
                                         ↓ (RRF fusion)
                               RAG Synthesis (Claude API)
```

## Structure

```
graph_feature_store/
│
├── data/
│   ├── edgar_client.py          # SEC EDGAR API client (rate-limited)
│   ├── edgar_parser.py          # Form 4 / DEF 14A / 10-K parsers
│   ├── graph_builder.py         # HeteroData graph from EDGAR relationships
│   └── schema.sql               # PostgreSQL schema
│
├── gnn/
│   ├── model.py                 # FinancialGNN: 3-layer HeteroConv GAT+SAGE
│   ├── trainer.py               # Multi-task training (classification + link pred)
│   ├── incremental_trainer.py   # 2-hop subgraph online retraining
│   └── compliance_classifier.py # Binary compliance risk head + XGBoost baseline
│
├── feature_store/
│   ├── server.go                # gRPC FeatureStoreServer (writes through Raft)
│   └── client.py                # Python gRPC client for GNN trainer
│
├── raft/
│   ├── node.go                  # Full Raft implementation (election, replication)
│   ├── wal.go                   # WAL: CRC32 + fdatasync, corruption recovery
│   ├── snapshot.go              # Snapshotting + log compaction
│   └── transport/
│       ├── grpc_transport.go    # gRPC transport (mTLS)
│       └── tcp_transport.go     # Raw TCP transport (for benchmarking)
│
├── kafka/
│   └── consumer.go              # EDGAR filing event consumer → stale invalidation
│
├── security/
│   ├── tls.go                   # mTLS cert loading, TLSConfig
│   ├── interceptors.go          # gRPC auth + logging + metrics interceptors
│   └── firebase_auth.go         # Firebase ID token verification for external clients
│
├── api/
│   └── cluster_status.go        # /cluster/status + /health HTTP endpoints
│
├── rag/
│   ├── indexer.py               # OpenSearch indexing (BM25 + dense)
│   ├── retriever.py             # Hybrid retrieval + RRF fusion
│   └── synthesizer.py           # Claude API RAG synthesis
│
├── benchmark/
│   ├── latency_test.go          # Read/write latency benchmark (eventual vs strong)
│   └── transport_benchmark.go   # Raw TCP vs gRPC overhead comparison
│
├── chaos/
│   └── chaos_monkey.go          # Fault injection: kill leaders, drop packets, slow WAL
│
├── evaluation/
│   └── evaluate_gnn.py          # GNN vs XGBoost F1/AUC comparison
│
├── proto/
│   └── feature_store.proto      # Protobuf definitions
│
├── scripts/
│   ├── gen_certs.sh             # Generate mTLS certificates
│   ├── run_cluster.sh           # Start 5-node local cluster
│   └── setup.sh                 # Dependencies + initial data pull
│
├── docker-compose.yml           # 5-node cluster + Kafka + PostgreSQL + OpenSearch
├── go.mod
└── requirements.txt
```

## Quick Start

```bash
# Dependencies
bash scripts/setup.sh

# Generate mTLS certificates
bash scripts/gen_certs.sh

# Pull EDGAR data (takes ~30 min for 1000 companies)
python data/edgar_client.py --companies 1000

# Start 5-node cluster
docker-compose up -d
bash scripts/run_cluster.sh

# Train GNN
python gnn/trainer.py --epochs 50 --hidden_dim 128

# Evaluate vs XGBoost baseline
python evaluation/evaluate_gnn.py --model_path checkpoints/best_model.pt

# Run chaos tests
go test ./chaos/... -v -timeout 120s
```

## Interview Q&A

**"Why Raft and not Zookeeper/etcd?"**
Raft is easier to reason about for a from-scratch implementation — the paper
is prescriptive. Etcd uses Raft internally. I wanted to own the full stack
so I could explain every line. For production I'd use etcd.

**"Why does the GNN beat XGBoost by 14%?"**
XGBoost sees nodes as independent — it uses degree, PageRank, transaction volume
as tabular features but ignores who those transactions are with. The GNN sees
the full graph: a company that looks legitimate in isolation but sits inside a
circular ownership cluster of shell companies gets a high-risk embedding because
its 2-hop neighbors are high-risk. That structural pattern can't be captured
without message passing.

**"Why WAL before responding to RPCs?"**
The WAL is what makes the system crash-safe. If we commit to the state machine
and then crash before writing to disk, the entry is lost. If we write to the WAL
first, we can recover by replaying it. fdatasync blocks until the kernel has
flushed to physical storage — not just the page cache. CRC32 on every entry
catches partial writes (e.g., power failure mid-write).

**"gRPC is +177% overhead vs raw TCP. Why use it?"**
177% sounds alarming but it's 0.55ms in absolute terms. At 12,000 writes/sec
that adds 6.6 seconds of overhead per second — so gRPC is the bottleneck
before Raft logic. But the tradeoff is worth it: mTLS is non-negotiable for
a distributed consensus cluster, HTTP/2 multiplexing avoids head-of-line
blocking, and gRPC interceptors give us auth, logging, and Prometheus metrics
with 10 lines of code. I'd drop to a custom binary protocol at 50,000+ writes/sec.
