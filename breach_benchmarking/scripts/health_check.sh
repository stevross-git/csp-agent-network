#!/bin/bash

# Health check script for Enhanced CSP system
echo "Enhanced CSP System Health Check"
echo "================================"

# Check main CSP service
echo -n "CSP Main Service (port 8000): "
if curl -s -f http://localhost:8000/health > /dev/null; then
    echo "✓ Online"
else
    echo "✗ Offline or unreachable"
fi

# Check threat detection service
echo -n "Threat Detection (port 8001): "
if curl -s -f http://localhost:8001/api/health > /dev/null; then
    echo "✓ Online"
else
    echo "✗ Offline or unreachable"
fi

# Check monitoring service
echo -n "Monitoring Service (port 8002): "
if curl -s -f http://localhost:8002/api/health > /dev/null; then
    echo "✓ Online"
else
    echo "✗ Offline or unreachable"
fi

# Check visualization service
echo -n "Visualization (port 8003): "
if curl -s -f http://localhost:8003/api/health > /dev/null; then
    echo "✓ Online"
else
    echo "✗ Offline or unreachable"
fi

# Check alert service
echo -n "Alert Service (port 8004): "
if curl -s -f http://localhost:8004/api/health > /dev/null; then
    echo "✓ Online"
else
    echo "✗ Offline or unreachable"
fi

# Check database connectivity
echo -n "Database Connection: "
if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "✓ Connected"
else
    echo "✗ Unable to connect"
fi

# Check Redis
echo -n "Redis Cache: "
if redis-cli ping > /dev/null 2>&1; then
    echo "✓ Connected"
else
    echo "✗ Unable to connect"
fi

echo ""
echo "System Resource Usage:"
echo "====================="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%"), $3/$2 * 100.0}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{printf "%s", $5}')"

echo ""
echo "Health check completed at $(date)"
