# File: docs/SECURITY-CERT-ROTATION.md
"""
# Certificate Rotation Guide

## Overview
This guide covers rotating SSL/TLS certificates for Ultimate Agent components.

## Rotation Schedule
- **PostgreSQL Certificates**: Every 90 days
- **Redis Certificates**: Every 90 days
- **Nginx Certificates**: Every 90 days (or use Let's Encrypt auto-renewal)
- **JWT Signing Keys**: Every 30 days

## Automated Rotation Process

### 1. PostgreSQL Certificate Rotation

```bash
# Generate new certificates
./scripts/rotate-postgres-certs.sh

# Update Kubernetes secrets (zero downtime)
kubectl create secret generic postgres-certs-new \
  --from-file=server.crt=certs/postgres/server.crt \
  --from-file=server.key=certs/postgres/server.key \
  --from-file=ca.crt=certs/postgres/ca.crt

# Patch deployment to use new secret
kubectl patch deployment ultimate-agent-postgres \
  -p '{"spec":{"template":{"spec":{"volumes":[{"name":"certs","secret":{"secretName":"postgres-certs-new"}}]}}}}'

# Verify new certs are working
kubectl exec -it postgres-pod -- psql -c "SELECT ssl_is_used();"

# Clean up old secret after verification
kubectl delete secret postgres-certs-old
```

### 2. Redis Certificate Rotation

```bash
# Similar process for Redis
./scripts/rotate-redis-certs.sh
kubectl create secret generic redis-certs-new --from-file=redis/tls/
kubectl rollout restart deployment ultimate-agent-redis
```

### 3. JWT Key Rotation

```bash
# Generate new keys
export NEW_JWT_REST_SECRET=$(openssl rand -base64 64)
export NEW_JWT_WS_SECRET=$(openssl rand -base64 64)

# Update secrets (supports dual keys during rotation)
kubectl patch secret ultimate-agent-secrets \
  --type='json' \
  -p='[
    {"op": "add", "path": "/data/JWT_REST_SECRET_NEW", "value": "'$(echo -n $NEW_JWT_REST_SECRET | base64)'"},
    {"op": "add", "path": "/data/JWT_WS_SECRET_NEW", "value": "'$(echo -n $NEW_JWT_WS_SECRET | base64)'"}
  ]'

# Deploy with dual key support
kubectl set env deployment/ultimate-agent-backend \
  JWT_REST_SECRET_NEW=$NEW_JWT_REST_SECRET \
  JWT_WS_SECRET_NEW=$NEW_JWT_WS_SECRET

# After all tokens expire (based on your longest token lifetime)
# Remove old keys and promote new to primary
kubectl patch secret ultimate-agent-secrets \
  --type='json' \
  -p='[
    {"op": "remove", "path": "/data/JWT_REST_SECRET"},
    {"op": "move", "from": "/data/JWT_REST_SECRET_NEW", "path": "/data/JWT_REST_SECRET"}
  ]'
```

## Manual Rotation (Emergency)

If automated rotation fails:

1. **Generate new certificates locally**
2. **Base64 encode them**: `base64 -w0 < server.crt`
3. **Update secrets manually**: `kubectl edit secret <secret-name>`
4. **Force pod restart**: `kubectl delete pod <pod-name>`

## Monitoring Certificate Expiry

Add these Prometheus alerts:

```yaml
- alert: CertificateExpiringSoon
  expr: (probe_ssl_earliest_cert_expiry - time()) / 86400 < 30
  annotations:
    summary: "Certificate expiring in {{ $value }} days"
```
"""
