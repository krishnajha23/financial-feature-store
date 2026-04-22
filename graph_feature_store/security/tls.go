// security/tls.go
// mTLS setup for inter-node Raft communication.
//
// mTLS = mutual TLS: both client AND server present certificates.
// In a standard TLS connection, only the server authenticates itself.
// mTLS forces both sides to authenticate — prevents rogue nodes from
// joining the cluster or replaying Raft messages.
//
// Certificate structure:
//   ca.crt     — root CA (signs all node certs)
//   node{n}.crt — certificate for node n (CN=node{n})
//   node{n}.key — private key for node n
//
// The AuthInterceptor (interceptors.go) extracts the caller's CN from
// the peer certificate and logs every RPC. Unauthorized callers (no cert
// or wrong CA) are rejected at the TLS handshake layer.
//
// Generate certs: bash scripts/gen_certs.sh (uses openssl)

package security

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"

	"google.golang.org/grpc/credentials"
)

// TLSConfig holds paths to the node's cert material.
type TLSConfig struct {
	CertFile   string // path to this node's certificate
	KeyFile    string // path to this node's private key
	CAFile     string // path to root CA certificate
	ServerName string // expected CN of servers we connect to (for client verification)
}

// ServerTLSCredentials builds gRPC server credentials with mTLS.
// The server requires clients to present a certificate signed by our CA.
func ServerTLSCredentials(cfg TLSConfig) (credentials.TransportCredentials, error) {
	// Load this node's cert+key
	cert, err := tls.LoadX509KeyPair(cfg.CertFile, cfg.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("load node cert: %w", err)
	}

	// Load root CA for verifying client certificates
	caPEM, err := os.ReadFile(cfg.CAFile)
	if err != nil {
		return nil, fmt.Errorf("read CA cert: %w", err)
	}
	caPool := x509.NewCertPool()
	if !caPool.AppendCertsFromPEM(caPEM) {
		return nil, fmt.Errorf("failed to parse CA cert")
	}

	tlsCfg := &tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientAuth:   tls.RequireAndVerifyClientCert, // mutual: require client cert
		ClientCAs:    caPool,
		MinVersion:   tls.VersionTLS13, // TLS 1.3 only — 1.2 has known issues
	}

	return credentials.NewTLS(tlsCfg), nil
}

// ClientTLSCredentials builds gRPC client credentials for dialing peers.
// The client presents its own certificate and verifies the server's cert.
func ClientTLSCredentials(cfg TLSConfig) (credentials.TransportCredentials, error) {
	cert, err := tls.LoadX509KeyPair(cfg.CertFile, cfg.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("load node cert: %w", err)
	}

	caPEM, err := os.ReadFile(cfg.CAFile)
	if err != nil {
		return nil, fmt.Errorf("read CA cert: %w", err)
	}
	caPool := x509.NewCertPool()
	if !caPool.AppendCertsFromPEM(caPEM) {
		return nil, fmt.Errorf("failed to parse CA cert")
	}

	tlsCfg := &tls.Config{
		Certificates: []tls.Certificate{cert},
		RootCAs:      caPool,
		ServerName:   cfg.ServerName, // verify server's CN matches expected name
		MinVersion:   tls.VersionTLS13,
	}

	return credentials.NewTLS(tlsCfg), nil
}
