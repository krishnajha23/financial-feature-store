// api/cluster_status.go
// HTTP API consumed by the Android cluster monitoring dashboard.
//
// Endpoints:
//   GET /cluster/status  — full cluster state (role, term, commit index, metrics)
//   GET /health          — liveness check (returns 503 during leader election)
//   GET /metrics         — Prometheus metrics scrape endpoint
//
// The Android app polls /cluster/status every 5s and fires a push notification
// if: role changes, election count increases, replication lag > 50ms,
// or commit index stalls for > 10s.

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	feature_store "github.com/yourusername/graph-feature-store/feature_store"
	raft "github.com/yourusername/graph-feature-store/raft"
)

// ------------------------------------------------------------------
// Types
// ------------------------------------------------------------------

type ClusterStatus struct {
	NodeID           string                        `json:"node_id"`
	Role             string                        `json:"role"`
	Term             int64                         `json:"term"`
	CommitIndex      int64                         `json:"commit_index"`
	LeaderID         string                        `json:"leader_id"`
	WritesPerSec     float64                       `json:"writes_per_sec"`
	ReplicationLagMs float64                       `json:"replication_lag_ms"`
	ElectionCount    int64                         `json:"election_count"`
	TotalWrites      int64                         `json:"total_writes"`
	FeatureStats     feature_store.FeatureStoreStats `json:"feature_stats"`
	Timestamp        time.Time                     `json:"timestamp"`
}

// ------------------------------------------------------------------
// HTTP handlers
// ------------------------------------------------------------------

// ClusterStatusHandler serves /cluster/status — polled by Android app.
func ClusterStatusHandler(node *raft.Node,
	fs *feature_store.FeatureStoreServer) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {
		status := ClusterStatus{
			NodeID:           node.NodeID(),
			Role:             node.Role(),
			Term:             node.CurrentTerm(),
			CommitIndex:      node.CommitIndex(),
			LeaderID:         node.LeaderID(),
			ReplicationLagMs: node.ReplicationLagMs(),
			ElectionCount:    node.ElectionCount(),
			TotalWrites:      node.WriteCount(),
			FeatureStats:     fs.Stats(),
			Timestamp:        time.Now(),
		}

		// Approximate writes/sec from total writes over last minute
		// In production: use a proper rate counter (EWMA)
		status.WritesPerSec = float64(node.WriteCount()) / 60.0

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*") // allow Android WebView
		json.NewEncoder(w).Encode(status)
	}
}

// HealthHandler serves /health — used by load balancers and Docker health checks.
// Returns 503 during leader election to prevent traffic during instability.
func HealthHandler(node *raft.Node) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		role := node.Role()
		health := map[string]interface{}{
			"status":    "healthy",
			"role":      role,
			"term":      node.CurrentTerm(),
			"timestamp": time.Now(),
		}

		if role == "candidate" {
			// Election in progress — don't route traffic here
			w.WriteHeader(http.StatusServiceUnavailable)
			health["status"] = "election_in_progress"
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(health)
	}
}

// ------------------------------------------------------------------
// HTTP server
// ------------------------------------------------------------------

func StartHTTPServer(
	node *raft.Node,
	fs   *feature_store.FeatureStoreServer,
	port int,
	ctx  context.Context,
) *http.Server {

	mux := http.NewServeMux()
	mux.HandleFunc("/cluster/status", ClusterStatusHandler(node, fs))
	mux.HandleFunc("/health",         HealthHandler(node))
	mux.Handle("/metrics",            promhttp.Handler())

	srv := &http.Server{
		Addr:         fmt.Sprintf(":%d", port),
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	go func() {
		log.Printf("[http] listening on :%d", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("[http] server error: %v", err)
		}
	}()

	go func() {
		<-ctx.Done()
		srv.Shutdown(context.Background())
	}()

	return srv
}
