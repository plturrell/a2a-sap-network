#!/bin/bash

# Self-signed SSL Certificate Generation Script for A2A Network
# This script creates development SSL certificates for local testing

SSL_DIR="/Users/apple/projects/a2a/ssl"
DOMAIN="localhost"
COUNTRY="US"
STATE="CA"
CITY="San Francisco"
ORG="A2A Network"
ORG_UNIT="Development"

echo "Generating SSL certificates for A2A Network..."

# Create private key
openssl genrsa -out "${SSL_DIR}/a2a.key" 2048

# Create certificate signing request
openssl req -new -key "${SSL_DIR}/a2a.key" -out "${SSL_DIR}/a2a.csr" -subj "/C=${COUNTRY}/ST=${STATE}/L=${CITY}/O=${ORG}/OU=${ORG_UNIT}/CN=${DOMAIN}"

# Create self-signed certificate valid for 1 year
openssl x509 -req -days 365 -in "${SSL_DIR}/a2a.csr" -signkey "${SSL_DIR}/a2a.key" -out "${SSL_DIR}/a2a.crt"

# Create certificate with Subject Alternative Names for multiple domains
cat > "${SSL_DIR}/openssl.conf" << EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
C=${COUNTRY}
ST=${STATE}
L=${CITY}
O=${ORG}
OU=${ORG_UNIT}
CN=${DOMAIN}

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
DNS.3 = a2a-network
DNS.4 = *.a2a-network
DNS.5 = 127.0.0.1
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# Generate new certificate with SAN
openssl req -new -key "${SSL_DIR}/a2a.key" -out "${SSL_DIR}/a2a-san.csr" -config "${SSL_DIR}/openssl.conf"
openssl x509 -req -days 365 -in "${SSL_DIR}/a2a-san.csr" -signkey "${SSL_DIR}/a2a.key" -out "${SSL_DIR}/a2a-san.crt" -extensions v3_req -extfile "${SSL_DIR}/openssl.conf"

# Set proper permissions
chmod 600 "${SSL_DIR}/a2a.key"
chmod 644 "${SSL_DIR}/a2a.crt" "${SSL_DIR}/a2a-san.crt"

# Clean up
rm "${SSL_DIR}/a2a.csr" "${SSL_DIR}/a2a-san.csr"

echo "SSL certificates generated successfully:"
echo "  Certificate: ${SSL_DIR}/a2a.crt (basic)"
echo "  Certificate: ${SSL_DIR}/a2a-san.crt (with SAN)"
echo "  Private Key: ${SSL_DIR}/a2a.key"
echo ""
echo "To use with Docker, mount the ssl directory:"
echo "  volumes:"
echo "    - ./ssl:/etc/ssl/certs:ro"
echo ""
echo "Note: These are self-signed certificates for development only."
echo "For production, use certificates from a trusted CA."