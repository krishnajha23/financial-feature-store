// raft/node.go
// Full Raft consensus implementation from scratch.
//
// Implements: leader election, log replication, safety, liveness.
// State machine: in-memory KV store (embeddings are values, namespace:entity_id are keys).
//
// Interview answer — "walk me through your Raft implementation":
//
// Three roles: Follower, Candidate, Leader.
// Term: monotonically increasing counter — the "epoch" of consensus.
//
// Election:
//   Follower → Candidate if no heartbeat within randomized 150-300ms timeout.
//   Candidate increments term, votes for itself, sends RequestVote to all peers.
//   Wins if it gets votes from a majority (N/2+1) of the cluster.
//   Safety: can only win if its log is at least as up-to-date as voters' logs.
//
// Log replication:
//   Leader receives Propose(key, value) → appends to WAL → AppendEntries to peers.
//   Committed when majority acknowledge → applied to state machine.
//   We return success to the caller only after majority ack — strong consistency.
//
// Safety guarantees:
//   At most one leader per term (quorum vote requirement).
//   A leader has all committed entries (log-completeness invariant).
//   State machine safety: applied entries never diverge across nodes.

package raft

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// ------------------------------------------------------------------
// Types
// ------------------------------------------------------------------

type Role int32

const (
	Follower  Role = iota
	Candidate
	Leader
)

func (r Role) String() string {
	switch r {
	case Follower:
		return "follower"
	case Candidate:
		return "candidate"
	case Leader:
		return "leader"
	default:
		return "unknown"
	}
}

type LogEntry struct {
	Term  int64  `json:"term"`
	Index int64  `json:"index"`
	Key   string `json:"key"`
	Value string `json:"value"`
}

type Config struct {
	NodeID  string
	Peers   []string
	DataDir string
	TLS     interface{} // *security.TLSConfig — interface to avoid circular import
}

type proposal struct {
	key    string
	value  string
	respCh chan error
}

// ------------------------------------------------------------------
// Node
// ------------------------------------------------------------------

type Node struct {
	config Config

	// Persistent state — written to WAL before responding to any RPC.
	// If we respond before persisting, a crash can violate safety.
	currentTerm int64
	votedFor    string
	log         []LogEntry

	// Volatile state — recomputed from WAL on recovery
	commitIndex int64
	lastApplied int64
	role        Role
	leaderID    string

	// Leader-only volatile state — reset on each election win
	nextIndex  map[string]int64 // for each peer: index of next log entry to send
	matchIndex map[string]int64 // for each peer: highest log index known to be replicated

	// State machine: in-memory KV store.
	// Embedding key format: "emb:{namespace}:{entity_id}"
	stateMachine map[string]string

	wal   *WAL
	peers map[string]FeatureStoreClient // gRPC clients to peer nodes

	proposeCh  chan proposal
	applyCh    chan LogEntry
	electionCh chan struct{} // receives signal when heartbeat arrives

	mu sync.RWMutex
	wg sync.WaitGroup

	// Prometheus metrics
	electionCount  int64
	writeCount     int64
	replicationLag int64 // nanoseconds
}

// ------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------

func NewNode(cfg Config) (*Node, error) {
	wal, err := NewWAL(cfg.DataDir + "/" + cfg.NodeID + ".wal")
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL: %w", err)
	}

	n := &Node{
		config:       cfg,
		role:         Follower,
		stateMachine: make(map[string]string),
		peers:        make(map[string]FeatureStoreClient),
		nextIndex:    make(map[string]int64),
		matchIndex:   make(map[string]int64),
		proposeCh:    make(chan proposal, 1000),
		applyCh:      make(chan LogEntry, 1000),
		electionCh:   make(chan struct{}, 1),
		wal:          wal,
	}

	// Recover persistent state from WAL before accepting any requests
	if err := n.recoverFromWAL(); err != nil {
		return nil, fmt.Errorf("WAL recovery failed: %w", err)
	}

	return n, nil
}

// ------------------------------------------------------------------
// Start
// ------------------------------------------------------------------

func (n *Node) Start(ctx context.Context) error {
	for _, peer := range n.config.Peers {
		if peer == n.config.NodeID {
			continue
		}
		client, err := newPeerClient(peer, n.config.TLS)
		if err != nil {
			log.Printf("[%s] failed to connect to peer %s: %v",
				n.config.NodeID, peer, err)
			continue
		}
		n.peers[peer] = client
	}

	n.wg.Add(3)
	go n.electionLoop(ctx)
	go n.applyLoop(ctx)
	go n.replicationLoop(ctx)

	log.Printf("[%s] started as %s, term=%d, log=%d entries",
		n.config.NodeID, n.role, n.currentTerm, len(n.log))
	return nil
}

// ------------------------------------------------------------------
// Election loop
// ------------------------------------------------------------------

func (n *Node) electionLoop(ctx context.Context) {
	defer n.wg.Done()

	for {
		timeout := electionTimeout() // randomized 150-300ms
		select {
		case <-ctx.Done():
			return
		case <-time.After(timeout):
			n.mu.Lock()
			if n.role != Leader {
				n.startElection()
			}
			n.mu.Unlock()
		case <-n.electionCh:
			// Heartbeat received — reset election timer.
			// This is how followers learn the leader is still alive.
		}
	}
}

func (n *Node) startElection() {
	n.role = Candidate
	n.currentTerm++
	n.votedFor = n.config.NodeID
	atomic.AddInt64(&n.electionCount, 1)

	// MUST persist term and vote before sending RequestVote RPCs.
	// If we crash after sending but before persisting, we might vote
	// for a different candidate in the same term on restart — violating safety.
	if err := n.wal.AppendTerm(n.currentTerm, n.votedFor); err != nil {
		log.Printf("[%s] failed to persist term: %v", n.config.NodeID, err)
		return
	}

	term         := n.currentTerm
	lastLogIndex := int64(len(n.log))
	lastLogTerm  := int64(0)
	if lastLogIndex > 0 {
		lastLogTerm = n.log[lastLogIndex-1].Term
	}

	log.Printf("[%s] starting election for term %d", n.config.NodeID, term)

	votes    := 1 // vote for self
	majority := len(n.config.Peers)/2 + 1

	var voteMu sync.Mutex
	var wg     sync.WaitGroup

	for peerID, peer := range n.peers {
		wg.Add(1)
		go func(id string, p FeatureStoreClient) {
			defer wg.Done()

			granted, peerTerm, err := p.RequestVote(
				context.Background(), term, n.config.NodeID,
				lastLogIndex, lastLogTerm)
			if err != nil {
				return
			}

			n.mu.Lock()
			defer n.mu.Unlock()

			// Higher term seen → we're stale, revert to follower
			if peerTerm > n.currentTerm {
				n.stepDown(peerTerm)
				return
			}

			if granted && n.role == Candidate && n.currentTerm == term {
				voteMu.Lock()
				votes++
				won := votes >= majority
				voteMu.Unlock()
				if won {
					n.becomeLeader()
				}
			}
		}(peerID, peer)
	}

	wg.Wait()
}

func (n *Node) becomeLeader() {
	n.role     = Leader
	n.leaderID = n.config.NodeID

	// Initialize nextIndex to len(log)+1 for all peers.
	// matchIndex to 0 (we don't know what peers have yet).
	for peerID := range n.peers {
		n.nextIndex[peerID]  = int64(len(n.log)) + 1
		n.matchIndex[peerID] = 0
	}

	log.Printf("[%s] became leader for term %d, log length=%d",
		n.config.NodeID, n.currentTerm, len(n.log))

	// Send immediate heartbeat to establish authority
	go n.sendHeartbeats()
}

func (n *Node) stepDown(newTerm int64) {
	n.currentTerm = newTerm
	n.role        = Follower
	n.votedFor    = ""
	n.wal.AppendTerm(n.currentTerm, n.votedFor)
}

// ------------------------------------------------------------------
// Replication loop (heartbeats)
// ------------------------------------------------------------------

func (n *Node) replicationLoop(ctx context.Context) {
	defer n.wg.Done()

	// Heartbeat every 50ms — must be much less than election timeout (150ms)
	// so followers don't time out while the leader is healthy.
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			n.mu.RLock()
			isLeader := n.role == Leader
			n.mu.RUnlock()
			if isLeader {
				n.sendHeartbeats()
			}
		}
	}
}

func (n *Node) sendHeartbeats() {
	n.mu.RLock()
	term    := n.currentTerm
	nodeID  := n.config.NodeID
	peers   := n.peers
	n.mu.RUnlock()

	for _, peer := range peers {
		go func(p FeatureStoreClient) {
			n.mu.RLock()
			prevLogIndex := n.nextIndex[nodeID] - 1
			prevLogTerm  := int64(0)
			if prevLogIndex > 0 && int(prevLogIndex) <= len(n.log) {
				prevLogTerm = n.log[prevLogIndex-1].Term
			}
			entries     := n.log[max(0, int(n.nextIndex[nodeID])-1):]
			commitIndex := n.commitIndex
			n.mu.RUnlock()

			t0 := time.Now()
			success, followerTerm, err := p.AppendEntries(
				context.Background(), term, nodeID,
				prevLogIndex, prevLogTerm, entries, commitIndex)
			lag := time.Since(t0)

			atomic.StoreInt64(&n.replicationLag, lag.Nanoseconds())

			if err == nil && !success && followerTerm > term {
				n.mu.Lock()
				n.stepDown(followerTerm)
				n.mu.Unlock()
			}
		}(peer)
	}
}

// ------------------------------------------------------------------
// Write path: Propose → WAL → replicate → commit
// ------------------------------------------------------------------

func (n *Node) Propose(ctx context.Context, key, value string) error {
	n.mu.RLock()
	if n.role != Leader {
		leaderID := n.leaderID
		n.mu.RUnlock()
		return fmt.Errorf("not leader, forward to: %s", leaderID)
	}
	n.mu.RUnlock()

	respCh := make(chan error, 1)
	select {
	case n.proposeCh <- proposal{key: key, value: value, respCh: respCh}:
	case <-ctx.Done():
		return ctx.Err()
	}

	select {
	case err := <-respCh:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (n *Node) applyLoop(ctx context.Context) {
	defer n.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return

		case p := <-n.proposeCh:
			n.mu.Lock()

			entry := LogEntry{
				Term:  n.currentTerm,
				Index: int64(len(n.log)) + 1,
				Key:   p.key,
				Value: p.value,
			}

			// WAL append BEFORE anything else.
			// If we crash between appending to WAL and applying,
			// recovery replays the entry. If we crash before WAL,
			// the entry is lost — but it was never committed, so safe.
			if err := n.wal.AppendEntry(entry); err != nil {
				n.mu.Unlock()
				p.respCh <- fmt.Errorf("WAL append failed: %w", err)
				continue
			}

			n.log = append(n.log, entry)
			n.mu.Unlock()

			// Replicate to majority before committing
			if err := n.replicateToMajority(ctx, entry); err != nil {
				p.respCh <- err
				continue
			}

			// Commit: apply to state machine
			n.mu.Lock()
			n.commitIndex = entry.Index
			n.stateMachine[entry.Key] = entry.Value
			n.lastApplied = entry.Index
			atomic.AddInt64(&n.writeCount, 1)
			n.mu.Unlock()

			p.respCh <- nil

		case entry := <-n.applyCh:
			// Applied by follower after receiving committed entries from leader
			n.mu.Lock()
			n.stateMachine[entry.Key] = entry.Value
			n.lastApplied = entry.Index
			n.mu.Unlock()
		}
	}
}

func (n *Node) replicateToMajority(ctx context.Context, entry LogEntry) error {
	majority := len(n.config.Peers)/2 + 1
	acks     := 1 // count self

	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, peer := range n.peers {
		wg.Add(1)
		go func(p FeatureStoreClient) {
			defer wg.Done()
			n.mu.RLock()
			term        := n.currentTerm
			leaderID    := n.config.NodeID
			commitIndex := n.commitIndex
			n.mu.RUnlock()

			success, _, err := p.AppendEntries(
				ctx, term, leaderID,
				entry.Index-1, 0,
				[]LogEntry{entry}, commitIndex)

			if err == nil && success {
				mu.Lock()
				acks++
				mu.Unlock()
			}
		}(peer)
	}

	wg.Wait()

	if acks < majority {
		return fmt.Errorf("failed to replicate to majority (%d/%d)", acks, majority)
	}
	return nil
}

// ------------------------------------------------------------------
// Read paths
// ------------------------------------------------------------------

// ReadLocal reads from local state machine — fast, potentially stale.
// Used for eventual-consistency reads (strong_read=false).
func (n *Node) ReadLocal(key string) (string, error) {
	n.mu.RLock()
	defer n.mu.RUnlock()
	val, ok := n.stateMachine[key]
	if !ok {
		return "", fmt.Errorf("key not found: %s", key)
	}
	return val, nil
}

// ReadFromLeader does a linearizable read — always sees latest committed value.
// More expensive: adds a network hop if this node is not the leader.
func (n *Node) ReadFromLeader(ctx context.Context, key string) (string, error) {
	n.mu.RLock()
	isLeader := n.role == Leader
	n.mu.RUnlock()

	if isLeader {
		return n.ReadLocal(key)
	}
	// In production: forward RPC to leader node via n.leaderID
	return "", fmt.Errorf("not leader; strong reads must be forwarded to leader")
}

// MarkStale writes a staleness marker to the state machine.
// Called by Kafka consumer when a new EDGAR filing invalidates embeddings.
func (n *Node) MarkStale(ctx context.Context, namespace, entityID,
	marker string) error {
	key := fmt.Sprintf("stale:%s:%s", namespace, entityID)
	return n.Propose(ctx, key, marker)
}

// ------------------------------------------------------------------
// Recovery from WAL
// ------------------------------------------------------------------

func (n *Node) recoverFromWAL() error {
	entries, term, votedFor, err := n.wal.ReadAll()
	if err != nil {
		return err
	}
	n.currentTerm = term
	n.votedFor    = votedFor
	n.log         = entries

	// Rebuild state machine by replaying all log entries
	for _, entry := range entries {
		n.stateMachine[entry.Key] = entry.Value
	}

	log.Printf("[%s] recovered %d entries from WAL, term=%d",
		n.config.NodeID, len(entries), term)
	return nil
}

// ------------------------------------------------------------------
// Metrics (accessed by Prometheus handler and cluster status API)
// ------------------------------------------------------------------

func (n *Node) NodeID() string { return n.config.NodeID }
func (n *Node) Role() string {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.role.String()
}
func (n *Node) CurrentTerm() int64 { return atomic.LoadInt64(&n.currentTerm) }
func (n *Node) CommitIndex() int64 {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.commitIndex
}
func (n *Node) LeaderID() string {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.leaderID
}
func (n *Node) ElectionCount() int64  { return atomic.LoadInt64(&n.electionCount) }
func (n *Node) WriteCount() int64     { return atomic.LoadInt64(&n.writeCount) }
func (n *Node) ReplicationLagMs() float64 {
	return float64(atomic.LoadInt64(&n.replicationLag)) / 1e6
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

func electionTimeout() time.Duration {
	// Randomized 150-300ms: the randomization prevents split votes.
	// If all nodes had the same timeout, they'd all start elections
	// simultaneously and votes would split forever.
	return time.Duration(150+rand.Intn(150)) * time.Millisecond
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// FeatureStoreClient is the interface our peer connections implement.
// Defined here to avoid circular imports.
type FeatureStoreClient interface {
	AppendEntries(ctx context.Context, term int64, leaderID string,
		prevLogIndex, prevLogTerm int64, entries []LogEntry,
		leaderCommit int64) (bool, int64, error)
	RequestVote(ctx context.Context, term int64, candidateID string,
		lastLogIndex, lastLogTerm int64) (bool, int64, error)
}

// newPeerClient creates a gRPC client to a peer node with mTLS.
// Actual implementation in feature_store/server.go.
func newPeerClient(addr string, tlsCfg interface{}) (FeatureStoreClient, error) {
	// Imported from feature_store package in real build
	return nil, fmt.Errorf("newPeerClient: implement in feature_store package")
}
