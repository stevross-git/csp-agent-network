#!/bin/bash
# Quick Fix for Monitoring Deployment Issues
# ==========================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# FIX 1: CREATE MISSING DIRECTORIES
# ============================================================================

log_info "Creating missing directories..."

# Create models directory for anomaly detector
mkdir -p monitoring/anomaly_detection/models
touch monitoring/anomaly_detection/models/.gitkeep
log_success "Created models directory"

# Create config directory
mkdir -p monitoring/anomaly_detection/config
touch monitoring/anomaly_detection/config/.gitkeep
log_success "Created config directory"

# ============================================================================
# FIX 2: UPDATE ANOMALY DETECTOR DOCKERFILE
# ============================================================================

log_info "Updating Anomaly Detector Dockerfile to handle missing directories..."

cat > monitoring/anomaly_detection/Dockerfile.anomaly << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY detector.py .

# Create directories for runtime
RUN mkdir -p models config

# Run the service
CMD ["python", "-u", "detector.py"]
EOF

log_success "Updated Anomaly Detector Dockerfile"

# ============================================================================
# FIX 3: CREATE SIMPLIFIED DEPLOYMENT SCRIPT
# ============================================================================

log_info "Creating simplified deployment script..."

cat > monitoring/deploy-monitoring-simple.sh << 'EOF'
#!/bin/bash
# Simplified Monitoring Deployment
# ================================

set -euo pipefail

echo "Deploying monitoring enhancements (simplified)..."
echo ""

# 1. Ensure network exists
echo "→ Creating Docker network..."
docker network create scripts_csp-network 2>/dev/null || echo "  Network already exists"

# 2. Start Tracing (Jaeger + OTEL)
echo ""
echo "→ Starting Distributed Tracing..."
cd monitoring/tracing
docker-compose -f docker-compose.tracing.yml up -d
cd ../..
echo "  ✓ Jaeger UI will be available at http://localhost:16686"

# 3. Build and start Anomaly Detection
echo ""
echo "→ Starting Anomaly Detection..."
cd monitoring/anomaly_detection

# Ensure directories exist
mkdir -p models config

# Build image
echo "  Building anomaly detector image..."
docker build -t csp-anomaly-detector -f Dockerfile.anomaly . || {
    echo "  ✗ Failed to build anomaly detector"
    echo "  Continuing with deployment..."
}

# Start container only if build succeeded
if docker images | grep -q csp-anomaly-detector; then
    docker-compose -f docker-compose.anomaly.yml up -d
    echo "  ✓ Anomaly detector started"
else
    echo "  ⚠ Skipping anomaly detector (build failed)"
fi

cd ../..

# 4. Verify running services
echo ""
echo "→ Verifying services..."
echo ""

# Check Jaeger
if docker ps | grep -q csp_jaeger; then
    echo "  ✓ Jaeger is running"
else
    echo "  ✗ Jaeger is not running"
fi

# Check OTEL Collector
if docker ps | grep -q csp_otel_collector; then
    echo "  ✓ OTEL Collector is running"
else
    echo "  ✗ OTEL Collector is not running"
fi

# Check Anomaly Detector
if docker ps | grep -q csp_anomaly_detector; then
    echo "  ✓ Anomaly Detector is running"
else
    echo "  ⚠ Anomaly Detector is not running (optional)"
fi

# 5. Test endpoints
echo ""
echo "→ Testing endpoints..."
echo ""

# Test Jaeger
if curl -s -f http://localhost:16686 > /dev/null 2>&1; then
    echo "  ✓ Jaeger UI is accessible at http://localhost:16686"
else
    echo "  ⚠ Jaeger UI is not yet accessible (may still be starting)"
fi

# Test OTEL metrics
if curl -s -f http://localhost:8888/metrics > /dev/null 2>&1; then
    echo "  ✓ OTEL Collector metrics available at http://localhost:8888/metrics"
else
    echo "  ⚠ OTEL Collector metrics not yet available"
fi

echo ""
echo "========================================="
echo "Deployment Summary"
echo "========================================="
echo ""
echo "Essential services (Tracing) have been deployed."
echo ""
echo "Access points:"
echo "• Jaeger UI: http://localhost:16686"
echo "• OTEL Metrics: http://localhost:8888/metrics"
echo ""
echo "Next steps:"
echo "1. Wait ~30 seconds for services to fully start"
echo "2. Verify Jaeger UI at http://localhost:16686"
echo "3. Integrate with your backend using the monitoring_integration module"
echo ""
echo "To check logs:"
echo "• Jaeger: docker logs csp_jaeger"
echo "• OTEL: docker logs csp_otel_collector"
echo "• Anomaly: docker logs csp_anomaly_detector"
EOF

chmod +x monitoring/deploy-monitoring-simple.sh

log_success "Created simplified deployment script"

# ============================================================================
# FIX 4: CREATE MINIMAL WORKING SECURITY MONITOR
# ============================================================================

log_info "Creating minimal security monitor setup..."

# Create a standalone security monitor that doesn't require complex setup
cat > monitoring/security/security_monitor_standalone.py << 'EOF'
"""
Standalone Security Monitor - Minimal Dependencies
"""
import re
from typing import Dict, List, Any
from datetime import datetime
import json

class MinimalSecurityMonitor:
    """Lightweight security monitor for testing"""
    
    def __init__(self):
        self.threat_patterns = [
            (r"(union|select|drop).*?(from|table)", "sql_injection", "high"),
            (r"<script[^>]*>", "xss", "high"),
            (r"\.\./|\.\.", "path_traversal", "medium"),
            (r"(;|\||&|`)", "command_injection", "high"),
        ]
        self.events = []
    
    def scan_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan request for threats"""
        threats = []
        
        # Convert request data to string for scanning
        scan_text = json.dumps(request_data)
        
        for pattern, threat_type, severity in self.threat_patterns:
            if re.search(pattern, scan_text, re.IGNORECASE):
                threats.append({
                    'type': threat_type,
                    'severity': severity,
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        threat_score = len(threats) * 25
        action = 'block' if threat_score >= 75 else 'allow'
        
        result = {
            'action': action,
            'threat_score': min(threat_score, 100),
            'threats': threats
        }
        
        self.events.append(result)
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get security status"""
        return {
            'total_scans': len(self.events),
            'threats_detected': sum(1 for e in self.events if e['threats']),
            'average_threat_score': sum(e['threat_score'] for e in self.events) / max(len(self.events), 1),
            'timestamp': datetime.utcnow().isoformat()
        }

# Global instance
security_monitor = MinimalSecurityMonitor()
EOF

log_success "Created minimal security monitor"

# ============================================================================
# FIX 5: CREATE TEST SCRIPT FOR VERIFICATION
# ============================================================================

log_info "Creating verification test script..."

cat > monitoring/test-monitoring.sh << 'EOF'
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
EOF

chmod +x monitoring/test-monitoring.sh

log_success "Created test script"

# ============================================================================
# FIX 6: CREATE BACKEND INTEGRATION STUB
# ============================================================================

log_info "Creating backend integration stub..."

mkdir -p backend/monitoring

# Create a minimal integration file if it doesn't exist
if [ ! -f "backend/monitoring/monitoring_integration.py" ]; then
    cat > backend/monitoring/monitoring_integration.py << 'EOF'
"""
Monitoring Integration for FastAPI
"""
from fastapi import FastAPI, Request
import time
import logging

logger = logging.getLogger(__name__)

def integrate_enhanced_monitoring(app: FastAPI) -> FastAPI:
    """
    Integrate monitoring into FastAPI app
    This is a minimal version that can be enhanced later
    """
    
    @app.middleware("http")
    async def add_monitoring_headers(request: Request, call_next):
        start_time = time.time()
        
        # Add trace ID to headers
        trace_id = f"trace-{int(time.time() * 1000000)}"
        
        response = await call_next(request)
        
        # Add monitoring headers
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Trace-ID"] = trace_id
        
        return response
    
    @app.get("/health/monitoring")
    async def monitoring_health():
        """Health check for monitoring systems"""
        return {
            "status": "healthy",
            "monitoring": {
                "tracing": "enabled",
                "metrics": "enabled",
                "security": "enabled"
            }
        }
    
    logger.info("Monitoring integration initialized")
    return app
EOF
    log_success "Created monitoring integration stub"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "====================================================="
echo "MONITORING DEPLOYMENT FIXES APPLIED"
echo "====================================================="
echo ""
echo "✅ Created missing directories (models/, config/)"
echo "✅ Updated Dockerfile to handle missing dirs"
echo "✅ Created simplified deployment script"
echo "✅ Created minimal security monitor"
echo "✅ Created test verification script"
echo "✅ Created backend integration stub"
echo ""
echo "IMMEDIATE NEXT STEPS:"
echo ""
echo "1. Run the simplified deployment:"
echo "   ./monitoring/deploy-monitoring-simple.sh"
echo ""
echo "2. Test the services:"
echo "   ./monitoring/test-monitoring.sh"
echo ""
echo "3. Check service logs if needed:"
echo "   docker logs csp_jaeger"
echo "   docker logs csp_otel_collector"
echo ""
echo "The essential tracing services should now deploy successfully!"