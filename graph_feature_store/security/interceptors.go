// security/firebase_auth.go + interceptors.go
// Two-layer auth:
//
// Layer 1 — mTLS (inter-node):
//   Raft peers authenticate via X.509 certificates signed by our CA.
//   Handled in tls.go + AuthInterceptor below.
//   No Firebase involved — this is server-to-server.
//
// Layer 2 — Firebase (external clients: Python trainer, Android app):
//   External callers (not Raft peers) present a Firebase ID token in metadata.
//   The Firebase auth interceptor verifies the token against Firebase Auth API.
//   On success: caller's uid and email are added to gRPC context.
//   Android app authenticates via Google Sign-In → gets Firebase ID token →
//   attaches as "Authorization: Bearer {token}" gRPC metadata header.
//
// Separation: Raft RPCs (AppendEntries, RequestVote) use mTLS only.
//             External RPCs (SetFeature, GetFeature) use Firebase token.
//             The interceptor chain runs Firebase check only on external methods.

package security

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"

	"github.com/prometheus/client_golang/prometheus"
)

// ------------------------------------------------------------------
// Prometheus metrics
// ------------------------------------------------------------------

var (
	rpcLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "grpc_server_latency_seconds",
			Help:    "gRPC server-side latency",
			Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0},
		},
		[]string{"method"},
	)
	rpcErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "grpc_server_errors_total",
			Help: "Total gRPC errors",
		},
		[]string{"method"},
	)
)

func init() {
	prometheus.MustRegister(rpcLatency, rpcErrors)
}

// ------------------------------------------------------------------
// mTLS AuthInterceptor (inter-node)
// ------------------------------------------------------------------

// AuthInterceptor extracts the caller's certificate CN and logs every RPC.
// Rejects calls with no peer certificate (can't happen with mTLS, but defensive).
func AuthInterceptor(nodeID string) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler) (interface{}, error) {

		p, ok := peer.FromContext(ctx)
		if !ok {
			return nil, status.Error(codes.Unauthenticated, "no peer info")
		}

		tlsInfo, ok := p.AuthInfo.(credentials.TLSInfo)
		if !ok {
			// External client — handled by FirebaseAuthInterceptor
			return handler(ctx, req)
		}

		if len(tlsInfo.State.PeerCertificates) == 0 {
			return nil, status.Error(codes.Unauthenticated, "no peer certificate")
		}

		callerCN := tlsInfo.State.PeerCertificates[0].Subject.CommonName
		log.Printf("[%s] RPC %s from peer CN=%s", nodeID, info.FullMethod, callerCN)

		return handler(ctx, req)
	}
}

// ------------------------------------------------------------------
// Firebase AuthInterceptor (external clients)
// ------------------------------------------------------------------

// FirebaseVerifier verifies Firebase ID tokens via Firebase Auth REST API.
// Caches verification results for 1 hour (tokens are valid for 1 hour).
type FirebaseVerifier struct {
	projectID string
	cache     map[string]cachedVerification
	cacheMu   sync.RWMutex
	httpClient *http.Client
}

type cachedVerification struct {
	uid       string
	email     string
	expiresAt time.Time
}

type firebaseKey string

const (
	FirebaseUIDKey   firebaseKey = "firebase_uid"
	FirebaseEmailKey firebaseKey = "firebase_email"
)

func NewFirebaseVerifier(projectID string) *FirebaseVerifier {
	return &FirebaseVerifier{
		projectID:  projectID,
		cache:      make(map[string]cachedVerification),
		httpClient: &http.Client{Timeout: 5 * time.Second},
	}
}

// Verify checks a Firebase ID token and returns (uid, email, error).
// Caches successful verifications for 55 minutes (tokens expire in 60 min).
func (v *FirebaseVerifier) Verify(token string) (uid, email string, err error) {
	// Check cache first
	v.cacheMu.RLock()
	if cached, ok := v.cache[token]; ok && time.Now().Before(cached.expiresAt) {
		v.cacheMu.RUnlock()
		return cached.uid, cached.email, nil
	}
	v.cacheMu.RUnlock()

	// Verify via Firebase Auth REST API
	url := fmt.Sprintf(
		"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=%s",
		v.projectID, // in practice, use FIREBASE_API_KEY env var
	)

	body := fmt.Sprintf(`{"idToken":"%s"}`, token)
	resp, err := v.httpClient.Post(url, "application/json",
		strings.NewReader(body))
	if err != nil {
		return "", "", fmt.Errorf("firebase verify request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("firebase verify: status %d", resp.StatusCode)
	}

	var result struct {
		Users []struct {
			LocalID string `json:"localId"`
			Email   string `json:"email"`
		} `json:"users"`
	}

	data, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(data, &result); err != nil {
		return "", "", fmt.Errorf("firebase verify parse: %w", err)
	}
	if len(result.Users) == 0 {
		return "", "", fmt.Errorf("firebase verify: no user found")
	}

	uid   = result.Users[0].LocalID
	email = result.Users[0].Email

	// Cache for 55 minutes
	v.cacheMu.Lock()
	v.cache[token] = cachedVerification{
		uid:       uid,
		email:     email,
		expiresAt: time.Now().Add(55 * time.Minute),
	}
	v.cacheMu.Unlock()

	return uid, email, nil
}

// FirebaseAuthInterceptor validates Firebase ID tokens from external clients.
// Skips validation for Raft-internal RPCs (those use mTLS).
func FirebaseAuthInterceptor(verifier *FirebaseVerifier) grpc.UnaryServerInterceptor {
	// Raft-internal RPC methods that don't need Firebase auth
	raftMethods := map[string]bool{
		"/feature_store.FeatureStore/AppendEntries": true,
		"/feature_store.FeatureStore/RequestVote":   true,
	}

	return func(ctx context.Context, req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler) (interface{}, error) {

		// Skip Firebase check for internal Raft RPCs (those use mTLS)
		if raftMethods[info.FullMethod] {
			return handler(ctx, req)
		}

		md, ok := metadata.FromIncomingContext(ctx)
		if !ok {
			return nil, status.Error(codes.Unauthenticated, "no metadata")
		}

		authHeaders := md.Get("authorization")
		if len(authHeaders) == 0 {
			return nil, status.Error(codes.Unauthenticated, "missing authorization header")
		}

		token := strings.TrimPrefix(authHeaders[0], "Bearer ")
		uid, email, err := verifier.Verify(token)
		if err != nil {
			return nil, status.Errorf(codes.Unauthenticated, "invalid token: %v", err)
		}

		// Inject uid/email into context for downstream handlers
		ctx = context.WithValue(ctx, FirebaseUIDKey,   uid)
		ctx = context.WithValue(ctx, FirebaseEmailKey, email)

		return handler(ctx, req)
	}
}

// ------------------------------------------------------------------
// Logging interceptor
// ------------------------------------------------------------------

func LoggingInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler) (interface{}, error) {

		t0       := time.Now()
		res, err := handler(ctx, req)
		lat      := time.Since(t0)

		if err != nil {
			log.Printf("RPC %s failed in %v: %v", info.FullMethod, lat, err)
		} else if lat > 10*time.Millisecond {
			// Log slow RPCs (> 10ms) — indicates potential Raft issues
			log.Printf("SLOW RPC %s: %v", info.FullMethod, lat)
		}

		return res, err
	}
}

// ------------------------------------------------------------------
// Prometheus metrics interceptor
// ------------------------------------------------------------------

func MetricsInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler) (interface{}, error) {

		t0       := time.Now()
		res, err := handler(ctx, req)
		lat      := time.Since(t0)

		rpcLatency.WithLabelValues(info.FullMethod).Observe(lat.Seconds())
		if err != nil {
			rpcErrors.WithLabelValues(info.FullMethod).Inc()
		}

		return res, err
	}
}

// ------------------------------------------------------------------
// Chain — apply all interceptors in order
// ------------------------------------------------------------------

// ChainInterceptors builds the interceptor chain for the gRPC server.
// Order: Metrics → Logging → mTLS Auth → Firebase Auth → handler
// Metrics first so we record latency even for auth failures.
func ChainInterceptors(nodeID string, fbVerifier *FirebaseVerifier) grpc.ServerOption {
	return grpc.ChainUnaryInterceptor(
		MetricsInterceptor(),
		LoggingInterceptor(),
		AuthInterceptor(nodeID),
		FirebaseAuthInterceptor(fbVerifier),
	)
}
