// raft/snapshot.go
// Snapshotting: capture full state machine state, discard old WAL entries.
//
// Without snapshotting: WAL grows without bound.
//   At 12,000 writes/sec × 256 bytes/entry = 3 MB/sec → 180 MB/min.
//   After 24 hours: 259 GB WAL. Restart takes hours to replay.
//
// With snapshotting: compact WAL every N entries (e.g., every 10,000 writes).
//   Snapshot = JSON of full state machine (all embeddings).
//   After snapshot: WAL only contains entries since snapshot.
//   Restart: load snapshot → replay only recent WAL entries.
//
// Snapshot file format: JSON with term, index, state_machine.
// File naming: {nodeID}.snap.{commitIndex} — allows multiple snapshots,
// easy to find the most recent (highest index).

package raft

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ------------------------------------------------------------------
// Snapshot types
// ------------------------------------------------------------------

// Snapshot captures the full state machine at a commit index.
// Once we have a snapshot at index N, all WAL entries with index ≤ N
// are redundant and can be discarded.
type Snapshot struct {
	Term         int64             `json:"term"`
	Index        int64             `json:"index"`
	StateMachine map[string]string `json:"state_machine"`
	CreatedAt    time.Time         `json:"created_at"`
	NodeID       string            `json:"node_id"`
}

// ------------------------------------------------------------------
// Taking snapshots
// ------------------------------------------------------------------

// TakeSnapshot captures the current state machine and writes it to disk.
// Should be called periodically (e.g., every 10,000 commits) or before
// restart to speed up recovery.
func (n *Node) TakeSnapshot() error {
	n.mu.RLock()
	snap := Snapshot{
		Term:         n.currentTerm,
		Index:        n.commitIndex,
		StateMachine: make(map[string]string, len(n.stateMachine)),
		CreatedAt:    time.Now(),
		NodeID:       n.config.NodeID,
	}
	for k, v := range n.stateMachine {
		snap.StateMachine[k] = v
	}
	n.mu.RUnlock()

	snapPath := filepath.Join(
		n.config.DataDir,
		fmt.Sprintf("%s.snap.%010d", n.config.NodeID, snap.Index),
	)

	data, err := json.Marshal(snap)
	if err != nil {
		return fmt.Errorf("snapshot marshal failed: %w", err)
	}

	// Write atomically: write to temp file, then rename.
	// Rename is atomic on POSIX — we never see a partial snapshot.
	tmpPath := snapPath + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("snapshot write failed: %w", err)
	}
	if err := os.Rename(tmpPath, snapPath); err != nil {
		return fmt.Errorf("snapshot rename failed: %w", err)
	}

	// Compact WAL: discard entries before snapshot index
	if err := n.wal.CompactBefore(snap.Index); err != nil {
		return fmt.Errorf("WAL compaction failed: %w", err)
	}

	// Trim in-memory log to entries after snapshot index
	n.mu.Lock()
	newLog := make([]LogEntry, 0)
	for _, e := range n.log {
		if e.Index > snap.Index {
			newLog = append(newLog, e)
		}
	}
	n.log = newLog
	n.mu.Unlock()

	fmt.Printf("[%s] snapshot taken at index=%d, term=%d, entries=%d, path=%s\n",
		n.config.NodeID, snap.Index, snap.Term,
		len(snap.StateMachine), snapPath)

	// Delete old snapshots (keep last 2 for safety)
	n.pruneOldSnapshots(2)

	return nil
}

// ------------------------------------------------------------------
// Loading snapshots (on restart)
// ------------------------------------------------------------------

// LoadLatestSnapshot finds and loads the most recent snapshot.
// Returns nil if no snapshots exist (fresh node).
func (n *Node) LoadLatestSnapshot() (*Snapshot, error) {
	pattern := filepath.Join(
		n.config.DataDir,
		fmt.Sprintf("%s.snap.*", n.config.NodeID),
	)
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("snapshot glob failed: %w", err)
	}

	// Filter out .tmp files
	var valid []string
	for _, m := range matches {
		if !strings.HasSuffix(m, ".tmp") {
			valid = append(valid, m)
		}
	}

	if len(valid) == 0 {
		return nil, nil // no snapshots — start fresh from WAL
	}

	// Sort by index (embedded in filename as 10-digit zero-padded number)
	sort.Strings(valid)
	latestPath := valid[len(valid)-1]

	data, err := os.ReadFile(latestPath)
	if err != nil {
		return nil, fmt.Errorf("snapshot read failed: %w", err)
	}

	var snap Snapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return nil, fmt.Errorf("snapshot unmarshal failed: %w", err)
	}

	fmt.Printf("[%s] loaded snapshot: index=%d, term=%d, entries=%d\n",
		n.config.NodeID, snap.Index, snap.Term, len(snap.StateMachine))

	return &snap, nil
}

// ApplySnapshot restores state machine from a snapshot.
// Called on startup if a snapshot exists.
func (n *Node) ApplySnapshot(snap *Snapshot) {
	n.mu.Lock()
	defer n.mu.Unlock()

	n.currentTerm = snap.Term
	n.commitIndex = snap.Index
	n.lastApplied = snap.Index

	// Restore state machine
	n.stateMachine = make(map[string]string, len(snap.StateMachine))
	for k, v := range snap.StateMachine {
		n.stateMachine[k] = v
	}

	// Trim log to entries after snapshot (these are still in WAL)
	var remaining []LogEntry
	for _, e := range n.log {
		if e.Index > snap.Index {
			remaining = append(remaining, e)
		}
	}
	n.log = remaining
}

// ------------------------------------------------------------------
// Snapshot transfer (InstallSnapshot RPC)
// ------------------------------------------------------------------

// InstallSnapshot handles the InstallSnapshot RPC from leader.
// Called when a follower is so far behind that log replication isn't enough.
// The leader sends its snapshot directly instead of individual log entries.
func (n *Node) InstallSnapshot(ctx interface{}, snap *Snapshot) error {
	n.mu.Lock()
	defer n.mu.Unlock()

	if snap.Index <= n.commitIndex {
		// We already have this data — ignore
		return nil
	}

	// Write snapshot to disk
	snapPath := filepath.Join(
		n.config.DataDir,
		fmt.Sprintf("%s.snap.%010d", n.config.NodeID, snap.Index),
	)
	data, _ := json.Marshal(snap)
	if err := os.WriteFile(snapPath, data, 0644); err != nil {
		return fmt.Errorf("InstallSnapshot write failed: %w", err)
	}

	// Apply to state machine
	n.currentTerm  = snap.Term
	n.commitIndex  = snap.Index
	n.lastApplied  = snap.Index
	n.stateMachine = snap.StateMachine
	n.log          = nil // clear log — snapshot covers all committed entries

	fmt.Printf("[%s] installed snapshot from leader: index=%d\n",
		n.config.NodeID, snap.Index)

	return nil
}

// ------------------------------------------------------------------
// Snapshot rotation
// ------------------------------------------------------------------

func (n *Node) pruneOldSnapshots(keepLast int) {
	pattern := filepath.Join(
		n.config.DataDir,
		fmt.Sprintf("%s.snap.*", n.config.NodeID),
	)
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return
	}

	var valid []string
	for _, m := range matches {
		if !strings.HasSuffix(m, ".tmp") {
			valid = append(valid, m)
		}
	}

	sort.Strings(valid)

	for i := 0; i < len(valid)-keepLast; i++ {
		os.Remove(valid[i])
		fmt.Printf("[%s] pruned old snapshot: %s\n", n.config.NodeID, valid[i])
	}
}
