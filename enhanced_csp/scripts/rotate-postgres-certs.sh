# File: scripts/rotate-postgres-certs.sh
#!/bin/bash
"""
#!/bin/bash
# Automated PostgreSQL certificate rotation

set -euo pipefail

CERT_DIR="certs/postgres"
BACKUP_DIR="certs/postgres.backup.$(date +%Y%m%d_%H%M%S)"

echo "🔄 Starting PostgreSQL certificate rotation..."

# Backup existing certificates
cp -r "$CERT_DIR" "$BACKUP_DIR"
echo "📦 Backed up existing certs to $BACKUP_DIR"

# Generate new certificates
./scripts/generate-certs.sh postgres

# Verify new certificates
openssl x509 -in "$CERT_DIR/server.crt" -noout -dates
openssl verify -CAfile "$CERT_DIR/ca.crt" "$CERT_DIR/server.crt"

echo "✅ New certificates generated and verified"
echo "📝 Next: Update Kubernetes secrets with kubectl commands"
"""
