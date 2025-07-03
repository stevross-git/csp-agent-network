# File: scripts/setup-redis-auth.sh
#!/bin/bash
"""
#!/bin/bash
# Setup Redis with authentication

set -euo pipefail

# Generate secure Redis password if not exists
if [ -z "${REDIS_PASSWORD:-}" ]; then
    export REDIS_PASSWORD=$(openssl rand -base64 32)
    echo "REDIS_PASSWORD=${REDIS_PASSWORD}" >> .env
fi

# Create Redis config directory
mkdir -p redis/conf redis/tls

# Generate Redis TLS certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout redis/tls/redis.key \
    -out redis/tls/redis.crt \
    -subj "/C=US/ST=State/L=City/O=UltimateAgent/CN=redis"

# Generate DH params for Redis
openssl dhparam -out redis/tls/redis.dh 2048

# Substitute password in config
envsubst < redis/redis.secure.conf > redis/conf/redis.conf

echo "✅ Redis authentication configured"
echo "📝 Test with: redis-cli -a \$REDIS_PASSWORD ping"
"""