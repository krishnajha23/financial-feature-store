// kafka/consumer.go
// Real-time EDGAR filing consumer.
//
// Flow:
//   Python EDGAR watcher detects new filings via RSS feed
//   → publishes EDGARFilingEvent to "edgar-filings" Kafka topic
//   → this consumer reads the event
//   → marks affected entity embeddings as stale in the Raft state machine
//   → Python GNN incremental retrainer picks up stale markers
//   → retrains on 2-hop subgraph → pushes fresh embeddings
//
// Why Kafka (vs direct RPC from Python watcher)?
//   Durability: Kafka retains messages even if the feature store is down.
//   Decoupling: the Python watcher doesn't need to know the feature store's address.
//   Replay: we can replay filing events to rebuild embeddings from scratch.
//   Fan-out: multiple consumers can receive the same filing event
//            (e.g., feature store invalidator + audit log + alert system).

package kafka

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	kafka "github.com/segmentio/kafka-go"
	feature_store "github.com/yourusername/graph-feature-store/feature_store"
)

// ------------------------------------------------------------------
// Event types
// ------------------------------------------------------------------

// EDGARFilingEvent is published when a new SEC filing is detected.
// The Python EDGAR watcher publishes these; this consumer receives them.
type EDGARFilingEvent struct {
	FilingType   string    `json:"filing_type"`    // "4", "DEF 14A", "10-K", "8-K"
	CIK          string    `json:"cik"`
	CompanyName  string    `json:"company_name"`
	AccessionNum string    `json:"accession_number"`
	FiledAt      time.Time `json:"filed_at"`
	// Embeddings for these entities are now stale — new filing changes relationships
	AffectedCIKs  []string `json:"affected_ciks"`
	AffectedExecs []string `json:"affected_execs"`
}

// ------------------------------------------------------------------
// Consumer
// ------------------------------------------------------------------

type EDGARConsumer struct {
	reader       *kafka.Reader
	featureStore *feature_store.FeatureStoreServer

	// Metrics
	processed   int64
	invalidated int64
	errors      int64
}

func NewEDGARConsumer(
	brokerAddr string,
	fs *feature_store.FeatureStoreServer,
) (*EDGARConsumer, error) {

	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers: []string{brokerAddr},
		Topic:   "edgar-filings",
		GroupID: "feature-store-invalidator",

		// Start from latest — we don't want to replay historical filings on startup
		// (embeddings for those are already in the store).
		// For a fresh cluster: set to kafka.FirstOffset to bootstrap from history.
		StartOffset: kafka.LastOffset,

		// Commit offsets every second — at-least-once delivery.
		// Duplicate events are idempotent (re-marking already-stale embeddings is safe).
		CommitInterval: time.Second,

		MinBytes: 1,
		MaxBytes: 10e6, // 10MB max message size
	})

	return &EDGARConsumer{
		reader:       reader,
		featureStore: fs,
	}, nil
}

// ------------------------------------------------------------------
// Run loop
// ------------------------------------------------------------------

func (c *EDGARConsumer) Run(ctx context.Context) {
	log.Printf("[kafka] EDGAR consumer started on topic edgar-filings")

	for {
		msg, err := c.reader.ReadMessage(ctx)
		if err != nil {
			if ctx.Err() != nil {
				log.Printf("[kafka] context cancelled — shutting down")
				return
			}
			log.Printf("[kafka] read error: %v", err)
			c.errors++
			time.Sleep(time.Second) // brief backoff on error
			continue
		}

		var event EDGARFilingEvent
		if err := json.Unmarshal(msg.Value, &event); err != nil {
			log.Printf("[kafka] failed to parse event at offset %d: %v",
				msg.Offset, err)
			c.errors++
			continue
		}

		c.processed++
		log.Printf("[kafka] filing: %s %s acc=%s affected=%d companies, %d execs",
			event.FilingType, event.CompanyName, event.AccessionNum,
			len(event.AffectedCIKs), len(event.AffectedExecs))

		if err := c.invalidateStaleEmbeddings(ctx, event); err != nil {
			log.Printf("[kafka] invalidation failed for %s: %v",
				event.AccessionNum, err)
			c.errors++
		}
	}
}

// ------------------------------------------------------------------
// Staleness invalidation
// ------------------------------------------------------------------

func (c *EDGARConsumer) invalidateStaleEmbeddings(
	ctx context.Context,
	event EDGARFilingEvent,
) error {
	// Staleness marker: JSON blob explaining why the embedding is stale.
	// Stored in Raft under key "stale:{namespace}:{entity_id}".
	// GetFeature handler checks for this key and sets is_stale=true.
	marker := fmt.Sprintf(
		`{"stale":true,"reason":"%s","accession":"%s","filed_at":"%s"}`,
		event.FilingType, event.AccessionNum,
		event.FiledAt.Format(time.RFC3339),
	)

	// Mark primary company
	if err := c.featureStore.MarkStale(ctx, "company_embeddings",
		event.CIK, marker); err != nil {
		return fmt.Errorf("mark primary company stale: %w", err)
	}
	c.invalidated++

	// Mark related companies (e.g., companies sharing board members with primary)
	for _, cik := range event.AffectedCIKs {
		if err := c.featureStore.MarkStale(ctx, "company_embeddings",
			cik, marker); err != nil {
			// Non-fatal: log and continue
			log.Printf("[kafka] failed to mark company %s stale: %v", cik, err)
			continue
		}
		c.invalidated++
	}

	// Mark affected executives
	// (executives who filed or are listed in the proxy statement)
	for _, exec := range event.AffectedExecs {
		if err := c.featureStore.MarkStale(ctx, "executive_embeddings",
			exec, marker); err != nil {
			log.Printf("[kafka] failed to mark exec %s stale: %v", exec, err)
			continue
		}
		c.invalidated++
	}

	log.Printf("[kafka] invalidated %d embeddings for filing %s",
		1+len(event.AffectedCIKs)+len(event.AffectedExecs),
		event.AccessionNum)

	return nil
}

// ------------------------------------------------------------------
// Stats and lifecycle
// ------------------------------------------------------------------

func (c *EDGARConsumer) Stats() map[string]int64 {
	return map[string]int64{
		"processed":   c.processed,
		"invalidated": c.invalidated,
		"errors":      c.errors,
	}
}

func (c *EDGARConsumer) Close() error {
	return c.reader.Close()
}

// ------------------------------------------------------------------
// Python Kafka producer (for reference — runs in Python EDGAR watcher)
// ------------------------------------------------------------------

// EDGARFilingProducer is the Go equivalent of the Python Kafka producer.
// The actual producer runs in Python (edgar_kafka_producer.py).
// This Go version is used in integration tests.
type EDGARFilingProducer struct {
	writer *kafka.Writer
}

func NewEDGARFilingProducer(brokerAddr string) *EDGARFilingProducer {
	return &EDGARFilingProducer{
		writer: &kafka.Writer{
			Addr:                   kafka.TCP(brokerAddr),
			Topic:                  "edgar-filings",
			Balancer:               &kafka.LeastBytes{},
			RequiredAcks:           kafka.RequireOne,
			AllowAutoTopicCreation: true,
		},
	}
}

func (p *EDGARFilingProducer) Publish(ctx context.Context,
	event EDGARFilingEvent) error {

	data, err := json.Marshal(event)
	if err != nil {
		return err
	}

	return p.writer.WriteMessages(ctx, kafka.Message{
		Key:   []byte(event.CIK),   // partition by CIK for ordering
		Value: data,
	})
}

func (p *EDGARFilingProducer) Close() error {
	return p.writer.Close()
}
