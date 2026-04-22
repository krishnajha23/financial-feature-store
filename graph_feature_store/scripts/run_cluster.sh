#!/bin/bash
# scripts/run_cluster.sh
# Start 5-node Raft cluster via Docker Compose and verify cluster health.

set -e

echo "=== Starting 5-node graph feature store cluster ==="

# Ensure certs exist
if [ ! -f certs/ca.crt ]; then
    echo "Generating mTLS certificates..."
    bash scripts/gen_certs.sh
fi

# Start infrastructure first
echo "Starting infrastructure (Postgres, Kafka, OpenSearch)..."
docker-compose up -d postgres kafka opensearch prometheus grafana
echo "Waiting for infrastructure to be healthy..."
sleep 15

# Start Raft nodes
echo "Starting Raft nodes..."
docker-compose up -d node0 node1 node2 node3 node4

echo "Waiting for leader election (up to 10s)..."
for i in $(seq 1 20); do
    sleep 0.5
    ROLE=$(curl -sf http://localhost:8080/cluster/status 2>/dev/null \
           | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['role'])" 2>/dev/null)
    if [ "$ROLE" = "leader" ]; then
        echo "  node0 is leader!"
        break
    fi
    # Try other nodes
    for PORT in 8081 8082 8083 8084; do
        ROLE=$(curl -sf http://localhost:$PORT/cluster/status 2>/dev/null \
               | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['role'])" 2>/dev/null)
        if [ "$ROLE" = "leader" ]; then
            echo "  node on port $PORT is leader!"
            break 2
        fi
    done
done

echo ""
echo "=== Cluster Status ==="
for PORT in 8080 8081 8082 8083 8084; do
    STATUS=$(curl -sf http://localhost:$PORT/cluster/status 2>/dev/null \
             | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'  {d[\"node_id\"]}: {d[\"role\"]:12s} term={d[\"term\"]}  commit={d[\"commit_index\"]}'
)" 2>/dev/null || echo "  port $PORT: not responding")
    echo "$STATUS"
done

echo ""
echo "=== Services ==="
echo "  gRPC (node0):       localhost:50051"
echo "  HTTP status (node0): http://localhost:8080/cluster/status"
echo "  Prometheus:          http://localhost:9090"
echo "  Grafana:             http://localhost:3000"
echo "  OpenSearch:          http://localhost:9200"
echo ""
echo "Next steps:"
echo "  python data/edgar_client.py --companies 100   # pull EDGAR data"
echo "  python gnn/trainer.py --use_synthetic         # train GNN"
echo "  python gnn/compliance_classifier.py --use_synthetic  # evaluate"
echo "  go test ./chaos/... -v -timeout 120s          # chaos tests"
