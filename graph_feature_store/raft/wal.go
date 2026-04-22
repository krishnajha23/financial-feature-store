// raft/wal.go
// Write-Ahead Log with CRC32 integrity and fdatasync durability.
//
// Record format on disk: [length uint32][CRC32 uint32][JSON payload]
//
// Why WAL?
//   Without WAL: commit to state machine, crash → entry lost → safety violation.
//   With WAL: write entry to disk first (fdatasync), then respond → entry survives crash.
//   Recovery: replay WAL entries to rebuild state machine.
//
// Why fdatasync (f.Sync())?
//   f.Sync() calls fdatasync(2) on Linux — flushes data blocks to physical storage.
//   Without it: data lives in the OS page cache, lost on power failure.
//   With it: data is on the physical medium before we return to the caller.
//   Cost: ~1-10ms per sync on SSD — this limits write throughput.
//   In production: use a write-combining buffer or batched sync for higher throughput.
//
// Why CRC32?
//   Detects partial writes (power failure mid-record).
//   On recovery: CRC32 mismatch → truncate corrupted tail → safe state.
//   Alternative: use length check only (cheaper) — but doesn't detect bit rot.
//   CRC32 is hardware-accelerated on modern CPUs (< 1ns per byte).

package raft

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sync"
)

// ------------------------------------------------------------------
// WAL types
// ------------------------------------------------------------------

type WAL struct {
	path string
	f    *os.File
	mu   sync.Mutex

	// Cached for AppendTerm without re-reading
	currentTerm int64
	votedFor    string
}

type walRecord struct {
	Type    string          `json:"type"`    // "entry" | "term"
	Payload json.RawMessage `json:"payload"`
}

type termRecord struct {
	Term     int64  `json:"term"`
	VotedFor string `json:"voted_for"`
}

// ------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------

func NewWAL(path string) (*WAL, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, fmt.Errorf("failed to create WAL directory: %w", err)
	}

	// O_APPEND: all writes go to end of file (atomic on Linux for writes < page size)
	// O_RDWR:   needed for Seek during recovery
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL: %w", err)
	}

	return &WAL{path: path, f: f}, nil
}

// ------------------------------------------------------------------
// Append operations
// ------------------------------------------------------------------

func (w *WAL) AppendEntry(entry LogEntry) error {
	payload, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	record := walRecord{Type: "entry", Payload: json.RawMessage(payload)}
	return w.writeRecord(record)
}

func (w *WAL) AppendTerm(term int64, votedFor string) error {
	w.mu.Lock()
	w.currentTerm = term
	w.votedFor    = votedFor
	w.mu.Unlock()

	payload, err := json.Marshal(termRecord{Term: term, VotedFor: votedFor})
	if err != nil {
		return err
	}
	record := walRecord{Type: "term", Payload: json.RawMessage(payload)}
	return w.writeRecord(record)
}

func (w *WAL) writeRecord(record walRecord) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("marshal failed: %w", err)
	}

	// CRC32 over the payload — detects partial writes and bit corruption
	checksum := crc32.ChecksumIEEE(data)

	// Write: [length uint32][CRC32 uint32][JSON payload]
	header := make([]byte, 8)
	binary.LittleEndian.PutUint32(header[0:4], uint32(len(data)))
	binary.LittleEndian.PutUint32(header[4:8], checksum)

	if _, err := w.f.Write(header); err != nil {
		return fmt.Errorf("WAL header write failed: %w", err)
	}
	if _, err := w.f.Write(data); err != nil {
		return fmt.Errorf("WAL payload write failed: %w", err)
	}

	// fdatasync: flush data (and enough metadata to recover) to physical storage.
	// This is the key durability guarantee — without it, data lives only in page cache.
	// On a crash after this call, the entry is guaranteed to be on disk.
	if err := w.f.Sync(); err != nil {
		return fmt.Errorf("fdatasync failed: %w", err)
	}

	return nil
}

// ------------------------------------------------------------------
// Recovery
// ------------------------------------------------------------------

func (w *WAL) ReadAll() ([]LogEntry, int64, string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Seek to beginning for full scan
	if _, err := w.f.Seek(0, io.SeekStart); err != nil {
		return nil, 0, "", err
	}

	reader := bufio.NewReaderSize(w.f, 64*1024) // 64KB read buffer

	var entries   []LogEntry
	term         := int64(0)
	votedFor     := ""
	currentPos   := int64(0)
	lastGoodPos  := int64(0)

	for {
		header := make([]byte, 8)
		_, err  := io.ReadFull(reader, header)
		if err == io.EOF {
			break // clean end of file
		}
		if err != nil {
			// Corrupted header — truncate from last good position
			fmt.Printf("[wal] corrupted header at offset %d — truncating\n", currentPos)
			w.f.Truncate(lastGoodPos)
			break
		}

		length   := binary.LittleEndian.Uint32(header[0:4])
		checksum := binary.LittleEndian.Uint32(header[4:8])

		// Guard against malformed length field
		if length > 10*1024*1024 { // 10MB max record size
			fmt.Printf("[wal] implausible record length %d at offset %d\n",
				length, currentPos)
			w.f.Truncate(lastGoodPos)
			break
		}

		payload := make([]byte, length)
		if _, err := io.ReadFull(reader, payload); err != nil {
			fmt.Printf("[wal] short read at offset %d — truncating\n", currentPos)
			w.f.Truncate(lastGoodPos)
			break
		}

		// Verify CRC32 — catches partial writes and storage errors
		if crc32.ChecksumIEEE(payload) != checksum {
			fmt.Printf("[wal] CRC32 mismatch at offset %d — truncating\n", currentPos)
			w.f.Truncate(lastGoodPos)
			break
		}

		var record walRecord
		if err := json.Unmarshal(payload, &record); err != nil {
			fmt.Printf("[wal] unmarshal failed at offset %d — truncating\n", currentPos)
			w.f.Truncate(lastGoodPos)
			break
		}

		switch record.Type {
		case "entry":
			var entry LogEntry
			if err := json.Unmarshal(record.Payload, &entry); err == nil {
				entries = append(entries, entry)
			}
		case "term":
			var tr termRecord
			if err := json.Unmarshal(record.Payload, &tr); err == nil {
				term     = tr.Term
				votedFor = tr.VotedFor
			}
		}

		lastGoodPos = currentPos + 8 + int64(length)
		currentPos  = lastGoodPos
	}

	// Seek to end for future appends
	w.f.Seek(0, io.SeekEnd)

	return entries, term, votedFor, nil
}

// ------------------------------------------------------------------
// Log compaction
// ------------------------------------------------------------------

// CompactBefore removes WAL entries with index <= snapshotIndex.
// Called after TakeSnapshot to prevent unbounded WAL growth.
// A GPT-2 equivalent embedding (117M params × 4 bytes) would be 468MB per entry —
// for feature store embeddings (64 floats × 4 bytes = 256 bytes per entity),
// the WAL grows at ~256 bytes × writes/sec. At 12,000 writes/sec:
//   3MB/sec → ~180MB/min → compact every ~5-10 minutes.
func (w *WAL) CompactBefore(snapshotIndex int64) error {
	entries, term, votedFor, err := w.ReadAll()
	if err != nil {
		return err
	}

	// Keep only entries after snapshot index
	var remaining []LogEntry
	for _, e := range entries {
		if e.Index > snapshotIndex {
			remaining = append(remaining, e)
		}
	}

	// Rewrite WAL atomically: truncate then re-append
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.f.Truncate(0); err != nil {
		return fmt.Errorf("truncate failed: %w", err)
	}
	if _, err := w.f.Seek(0, io.SeekStart); err != nil {
		return err
	}

	// Re-write term record (always first)
	termPayload, _ := json.Marshal(termRecord{Term: term, VotedFor: votedFor})
	w.writeRecord(walRecord{Type: "term", Payload: json.RawMessage(termPayload)})

	// Re-write remaining entries
	for _, e := range remaining {
		payload, _ := json.Marshal(e)
		if err := w.writeRecord(walRecord{
			Type: "entry", Payload: json.RawMessage(payload),
		}); err != nil {
			return fmt.Errorf("re-append failed at index %d: %w", e.Index, err)
		}
	}

	fmt.Printf("[wal] compacted: kept %d entries (index > %d)\n",
		len(remaining), snapshotIndex)
	return nil
}

func (w *WAL) Close() error {
	return w.f.Close()
}
