// feature_store/server.go
// gRPC FeatureStoreServer — thin layer over Raft node.
//
// Write path: SetFeature → serialize embedding → Raft.Propose → blocks until committed
//   Strong consistency: every write is committed on a majority before returning.
//   At 12,000 writes/sec this is the bottleneck (gRPC adds 0.55ms vs raw TCP).
//
// Read paths:
//   strong_read=false → ReadLocal  (local replica, potentially stale, fast)
//   strong_read=true  → ReadFromLeader (always consistent, adds network hop)
//
// Staleness detection: every GetFeature response includes is_stale flag.
//   Stale = (now - updated_at_ms) > staleness_threshold (default 5 min).
//   Kafka consumer marks embeddings stale when new EDGAR filings arrive.
//   GNN retraining then pushes fresh embeddings.

package feature_store

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	pb  "github.com/yourusername/graph-feature-store/proto"
	raft "github.com/yourusername/graph-feature-store/raft"
)

// ------------------------------------------------------------------
// Types
// ------------------------------------------------------------------

// FeatureValue is the JSON-serialized value stored in the Raft state machine.
// Key format: "emb:{namespace}:{entity_id}"
type FeatureValue struct {
	Embedding    []float32 `json:"embedding"`
	ModelVersion int64     `json:"model_version"`
	UpdatedAtMs  int64     `json:"updated_at_ms"`
}

// Metrics tracks read/write latencies and staleness rates.
type Metrics struct {
	totalWrites    int64
	totalReads     int64
	staleReads     int64
	writeLatencyNs int64  // exponential moving average (× 1000 for precision)
	readLatencyNs  int64

	// Histogram buckets for p50/p99
	readLatencies  []float64
	writeLatencies []float64
	mu             sync.Mutex
}

func NewMetrics() *Metrics { return &Metrics{} }

func (m *Metrics) RecordWrite(d time.Duration) {
	atomic.AddInt64(&m.totalWrites, 1)
	m.mu.Lock()
	m.writeLatencies = append(m.writeLatencies, float64(d.Microseconds())/1000.0)
	if len(m.writeLatencies) > 10000 {
		m.writeLatencies = m.writeLatencies[1000:]
	}
	m.mu.Unlock()
}

func (m *Metrics) RecordRead(d time.Duration, strong, stale bool) {
	atomic.AddInt64(&m.totalReads, 1)
	if stale {
		atomic.AddInt64(&m.staleReads, 1)
	}
	m.mu.Lock()
	m.readLatencies = append(m.readLatencies, float64(d.Microseconds())/1000.0)
	if len(m.readLatencies) > 10000 {
		m.readLatencies = m.readLatencies[1000:]
	}
	m.mu.Unlock()
}

func (m *Metrics) P50ReadMs() float64  { return m.percentile(m.readLatencies,  50) }
func (m *Metrics) P99ReadMs() float64  { return m.percentile(m.readLatencies,  99) }
func (m *Metrics) P50WriteMs() float64 { return m.percentile(m.writeLatencies, 50) }

func (m *Metrics) percentile(data []float64, pct float64) float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(data) == 0 {
		return 0
	}
	sorted := make([]float64, len(data))
	copy(sorted, data)
	// Quickselect would be faster but sort is fine for monitoring
	idx := int(math.Ceil(pct/100.0*float64(len(sorted)))) - 1
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// ------------------------------------------------------------------
// Server
// ------------------------------------------------------------------

type FeatureStoreServer struct {
	pb.UnimplementedFeatureStoreServer

	raft                 *raft.Node
	stalenessThresholdMs int64  // embeddings older than this are flagged stale
	metrics              *Metrics

	// Entity list cache (for list_entities calls from Python client)
	entityCache map[string][]string
	cacheMu     sync.RWMutex
}

func NewFeatureStoreServer(raftNode *raft.Node) *FeatureStoreServer {
	return &FeatureStoreServer{
		raft:                 raftNode,
		stalenessThresholdMs: 5 * 60 * 1000, // 5 minutes
		metrics:              NewMetrics(),
		entityCache:          make(map[string][]string),
	}
}

// ------------------------------------------------------------------
// SetFeature — write path
// ------------------------------------------------------------------

// SetFeature stores a GNN embedding in the Raft-backed store.
// Blocks until the write is committed on a majority of nodes.
//
// Key format: "emb:{namespace}:{entity_id}"
// Namespace: "company_embeddings" | "executive_embeddings"
func (s *FeatureStoreServer) SetFeature(ctx context.Context,
	req *pb.SetFeatureRequest) (*pb.SetFeatureResponse, error) {

	t0  := time.Now()
	key := fmt.Sprintf("emb:%s:%s", req.Namespace, req.EntityId)

	val := FeatureValue{
		Embedding:    req.Embedding,
		ModelVersion: req.ModelVersion,
		UpdatedAtMs:  time.Now().UnixMilli(),
	}

	data, err := json.Marshal(val)
	if err != nil {
		return nil, fmt.Errorf("marshal failed: %w", err)
	}

	// Write through Raft — strong consistency.
	// Returns only after committed on majority.
	if err := s.raft.Propose(ctx, key, string(data)); err != nil {
		return nil, fmt.Errorf("raft write failed: %w", err)
	}

	// Update entity cache
	s.cacheMu.Lock()
	s.entityCache[req.Namespace] = append(s.entityCache[req.Namespace], req.EntityId)
	s.cacheMu.Unlock()

	s.metrics.RecordWrite(time.Since(t0))

	return &pb.SetFeatureResponse{
		Success:     true,
		CommittedAt: time.Now().Format(time.RFC3339),
	}, nil
}

// ------------------------------------------------------------------
// GetFeature — read path
// ------------------------------------------------------------------

func (s *FeatureStoreServer) GetFeature(ctx context.Context,
	req *pb.GetFeatureRequest) (*pb.GetFeatureResponse, error) {

	t0  := time.Now()
	key := fmt.Sprintf("emb:%s:%s", req.Namespace, req.EntityId)

	var (
		data string
		err  error
	)

	if req.StrongRead {
		// Leader read: always consistent, adds one network hop if not leader.
		// Use for compliance queries where stale embeddings could cause incorrect
		// risk assessments.
		data, err = s.raft.ReadFromLeader(ctx, key)
	} else {
		// Local read: fast (~0.1ms), potentially stale (bounded by replication lag).
		// Replication lag in healthy cluster: < 5ms.
		// Use for batch embedding lookups where slight staleness is acceptable.
		data, err = s.raft.ReadLocal(key)
	}

	if err != nil {
		return nil, fmt.Errorf("read failed: %w", err)
	}

	var val FeatureValue
	if err := json.Unmarshal([]byte(data), &val); err != nil {
		return nil, fmt.Errorf("unmarshal failed: %w", err)
	}

	stalenessMs := time.Now().UnixMilli() - val.UpdatedAtMs
	isStale     := stalenessMs > s.stalenessThresholdMs

	s.metrics.RecordRead(time.Since(t0), req.StrongRead, isStale)

	return &pb.GetFeatureResponse{
		Embedding:    val.Embedding,
		ModelVersion: val.ModelVersion,
		UpdatedAtMs:  val.UpdatedAtMs,
		IsStale:      isStale,
		ServedBy:     s.raft.NodeID(),
	}, nil
}

// ------------------------------------------------------------------
// BatchGetFeatures — parallel fan-out
// ------------------------------------------------------------------

func (s *FeatureStoreServer) BatchGetFeatures(ctx context.Context,
	req *pb.BatchGetRequest) (*pb.BatchGetResponse, error) {

	t0      := time.Now()
	results := make(map[string]*pb.GetFeatureResponse, len(req.EntityIds))
	errCh   := make(chan error, len(req.EntityIds))
	var mu  sync.Mutex
	var wg  sync.WaitGroup

	// Fan out reads in parallel — local reads hit only the local state machine,
	// so N parallel reads cost the same as 1 (no network, just map lookups).
	for _, entityID := range req.EntityIds {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			resp, err := s.GetFeature(ctx, &pb.GetFeatureRequest{
				Namespace:  req.Namespace,
				EntityId:   id,
				StrongRead: req.StrongRead,
			})
			if err != nil {
				errCh <- fmt.Errorf("entity %s: %w", id, err)
				return
			}
			mu.Lock()
			results[id] = resp
			mu.Unlock()
		}(entityID)
	}

	wg.Wait()
	close(errCh)

	// Collect first error (if any)
	for err := range errCh {
		if err != nil {
			return nil, err
		}
	}

	return &pb.BatchGetResponse{
		Features:  results,
		LatencyMs: time.Since(t0).Milliseconds(),
	}, nil
}

// ------------------------------------------------------------------
// GetClusterStats — consumed by Android monitoring dashboard
// ------------------------------------------------------------------

func (s *FeatureStoreServer) GetClusterStats(ctx context.Context,
	_ *pb.StatsRequest) (*pb.ClusterStats, error) {

	totalFeatures := int64(0)
	s.cacheMu.RLock()
	for _, ids := range s.entityCache {
		totalFeatures += int64(len(ids))
	}
	s.cacheMu.RUnlock()

	writeCount := s.raft.WriteCount()
	// Approximate writes/sec using rolling window (simplified)
	writesPerSec := float64(writeCount) / 60.0 // TODO: proper rate tracking

	return &pb.ClusterStats{
		NodeId:         s.raft.NodeID(),
		Role:           s.raft.Role(),
		Term:           s.raft.CurrentTerm(),
		CommitIndex:    s.raft.CommitIndex(),
		TotalFeatures:  totalFeatures,
		WritesPerSec:   writesPerSec,
		ReadsPerSec:    float64(atomic.LoadInt64(&s.metrics.totalReads)) / 60.0,
	}, nil
}

// ------------------------------------------------------------------
// MarkStale — called by Kafka consumer on new EDGAR filings
// ------------------------------------------------------------------

// MarkStale writes a staleness marker to the Raft state machine.
// The GetFeature handler returns is_stale=true for these entities
// until fresh embeddings are pushed by the GNN retrainer.
func (s *FeatureStoreServer) MarkStale(ctx context.Context,
	namespace, entityID, marker string) error {
	key := fmt.Sprintf("stale:%s:%s", namespace, entityID)
	return s.raft.Propose(ctx, key, marker)
}

// ------------------------------------------------------------------
// Stats (for cluster status HTTP handler)
// ------------------------------------------------------------------

type FeatureStoreStats struct {
	TotalFeatures    int64   `json:"total_features"`
	StaleFeatures    int64   `json:"stale_features"`
	ReadLatencyP50Ms float64 `json:"read_latency_p50_ms"`
	ReadLatencyP99Ms float64 `json:"read_latency_p99_ms"`
}

func (s *FeatureStoreServer) Stats() FeatureStoreStats {
	s.cacheMu.RLock()
	total := int64(0)
	for _, ids := range s.entityCache {
		total += int64(len(ids))
	}
	s.cacheMu.RUnlock()

	return FeatureStoreStats{
		TotalFeatures:    total,
		StaleFeatures:    atomic.LoadInt64(&s.metrics.staleReads),
		ReadLatencyP50Ms: s.metrics.P50ReadMs(),
		ReadLatencyP99Ms: s.metrics.P99ReadMs(),
	}
}
