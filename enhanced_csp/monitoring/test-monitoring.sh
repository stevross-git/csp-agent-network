#!/bin/bash
# Test Monitoring Services
# ========================

echo "Testing Monitoring Services..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test function
test_endpoint() {
    local name=$1
    local url=$2
    local expected=$3
    
    echo -n "Testing $name... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
    
    if [ "$response" = "$expected" ]; then
        echo -e "${GREEN}✓ OK${NC} (HTTP $response)"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (HTTP $response, expected $expected)"
        return 1
    fi
}

# Test Docker network
echo -n "Checking Docker network... "
if docker network ls | grep -q scripts_csp-network; then
    echo -e "${GREEN}✓ EXISTS${NC}"
else
    echo -e "${RED}✗ MISSING${NC}"
    echo "Run: docker network create scripts_csp-network"
fi

echo ""

# Test services
test_endpoint "Jaeger UI" "http://localhost:16686" "200"
test_endpoint "OTEL Collector Metrics" "http://localhost:8888/metrics" "200"
test_endpoint "Prometheus" "http://localhost:9090/-/healthy" "200"
test_endpoint "Grafana" "http://localhost:3001/api/health" "200"

echo ""

# Check running containers
echo "Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(jaeger|otel|anomaly|prometheus|grafana)" || echo "No monitoring containers found"

echo ""
echo "Logs check (last 5 lines):"
echo ""

# Check Jaeger logs
if docker ps | grep -q csp_jaeger; then
    echo "Jaeger logs:"
    docker logs csp_jaeger 2>&1 | tail -5 | sed 's/^/  /'
    echo ""
fi

# Check OTEL logs
if docker ps | grep -q csp_otel_collector; then
    echo "OTEL Collector logs:"
    docker logs csp_otel_collector 2>&1 | tail -5 | sed 's/^/  /'
fi
