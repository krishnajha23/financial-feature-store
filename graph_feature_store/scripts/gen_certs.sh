#!/bin/bash
# scripts/gen_certs.sh
# Generate mTLS certificates for 5-node cluster.
# Creates: ca.crt, node{0-4}.crt, node{0-4}.key
#
# Certificate CN = node ID (node0, node1, ...).
# The AuthInterceptor extracts CN from peer certificate to identify callers.
#
# In production: use a proper CA (Vault PKI, AWS ACM PCA).
# For development/testing: these self-signed certs are fine.

set -e

CERTS_DIR="certs"
DAYS=3650  # 10 years for dev certs
mkdir -p "$CERTS_DIR"

echo "=== Generating root CA ==="
openssl genrsa -out "$CERTS_DIR/ca.key" 4096
openssl req -new -x509 -key "$CERTS_DIR/ca.key" \
    -out "$CERTS_DIR/ca.crt" -days "$DAYS" \
    -subj "/CN=raft-ca/O=graph-feature-store"

echo "=== Generating node certificates ==="
for i in 0 1 2 3 4; do
    NODE="node${i}"
    echo "  $NODE..."

    # Generate key
    openssl genrsa -out "$CERTS_DIR/${NODE}.key" 2048

    # Generate CSR
    openssl req -new -key "$CERTS_DIR/${NODE}.key" \
        -out "$CERTS_DIR/${NODE}.csr" \
        -subj "/CN=${NODE}/O=graph-feature-store"

    # Sign with CA
    # SAN (Subject Alternative Names) required for modern TLS validation
    openssl x509 -req \
        -in "$CERTS_DIR/${NODE}.csr" \
        -CA "$CERTS_DIR/ca.crt" \
        -CAkey "$CERTS_DIR/ca.key" \
        -CAcreateserial \
        -out "$CERTS_DIR/${NODE}.crt" \
        -days "$DAYS" \
        -extfile <(printf "subjectAltName=DNS:${NODE},DNS:localhost,IP:127.0.0.1")

    rm "$CERTS_DIR/${NODE}.csr"
    chmod 600 "$CERTS_DIR/${NODE}.key"
    echo "  Generated $CERTS_DIR/${NODE}.{crt,key}"
done

echo ""
echo "=== Certificate Summary ==="
for f in "$CERTS_DIR"/*.crt; do
    echo -n "  $f: "
    openssl x509 -noout -subject -in "$f" 2>/dev/null | sed 's/subject=//'
done

echo ""
echo "Done. Certificates in $CERTS_DIR/"
echo "Start cluster: bash scripts/run_cluster.sh"
