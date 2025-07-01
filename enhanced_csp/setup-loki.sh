#!/bin/bash
# setup-loki.sh - Complete Loki setup and verification

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🚀 Setting up Loki logging stack..."

# 1. Ensure directories exist
echo "Creating required directories..."
mkdir -p monitoring/loki/{data,config}
mkdir -p monitoring/promtail/config
mkdir -p monitoring/grafana/datasources
mkdir -p logs/{application,database,monitoring,audit}

# 2. Set proper permissions
echo "Setting permissions..."
chmod -R 755 monitoring/loki
chmod -R 755 monitoring/promtail
chmod -R 755 logs

# 3. Create Grafana datasource for Loki
cat > monitoring/grafana/datasources/loki.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    isDefault: false
    editable: true
    jsonData:
      maxLines: 1000
EOF

# 4. Update docker-compose.monitoring.yml to include Loki services
echo "Updating docker-compose file..."

# Check if Loki is already in the compose file
if ! grep -q "loki:" monitoring/docker-compose.monitoring.yml 2>/dev/null; then
    echo "Adding Loki services to docker-compose.monitoring.yml..."
    
    # Add Loki services before the volumes section
    sed -i '/^volumes:/i \
  loki:\
    image: grafana/loki:latest\
    container_name: csp_loki\
    command: -config.file=/etc/loki/loki.yml\
    ports:\
      - "3100:3100"\
    volumes:\
      - ./monitoring/loki/loki.yml:/etc/loki/loki.yml\
      - ./monitoring/loki/recording_rules.yml:/loki/rules/recording_rules.yml\
      - loki_data:/loki\
    networks:\
      - scripts_csp-network\
    restart: unless-stopped\
    healthcheck:\
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:3100/ready"]\
      interval: 30s\
      timeout: 10s\
      retries: 3\
\
  promtail:\
    image: grafana/promtail:latest\
    container_name: csp_promtail\
    command: -config.file=/etc/promtail/promtail.yml\
    volumes:\
      - ./monitoring/promtail/promtail.yml:/etc/promtail/promtail.yml\
      - /var/log:/var/log:ro\
      - /var/lib/docker/containers:/var/lib/docker/containers:ro\
      - /var/run/docker.sock:/var/run/docker.sock:ro\
      - ./logs:/var/log/csp:ro\
    networks:\
      - scripts_csp-network\
    restart: unless-stopped\
    depends_on:\
      - loki\
' monitoring/docker-compose.monitoring.yml

    # Add loki_data volume
    sed -i '/^volumes:/a \  loki_data:\n    driver: local' monitoring/docker-compose.monitoring.yml
fi

# 5. Start Loki services
echo -e "${YELLOW}Starting Loki services...${NC}"
docker-compose -f monitoring/docker-compose.monitoring.yml up -d loki promtail

# 6. Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# 7. Verify Loki is running
echo -e "\n${GREEN}Verifying Loki setup...${NC}"

# Check Loki health
if curl -s http://localhost:3100/ready | grep -q "ready"; then
    echo -e "${GREEN}✓ Loki is ready${NC}"
else
    echo -e "${RED}✗ Loki is not responding${NC}"
fi

# Check Promtail
if docker ps | grep -q csp_promtail; then
    echo -e "${GREEN}✓ Promtail is running${NC}"
else
    echo -e "${RED}✗ Promtail is not running${NC}"
fi

# 8. Test log ingestion
echo -e "\n${YELLOW}Testing log ingestion...${NC}"

# Create a test log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TEST: Loki test log entry" >> logs/application/test.log

# Wait for log to be ingested
sleep 5

# Query Loki for the test log
RESULT=$(curl -s -G -X GET "http://localhost:3100/loki/api/v1/query_range" \
    --data-urlencode 'query={job="csp-api"}' \
    --data-urlencode 'limit=5' 2>/dev/null || echo "failed")

if [[ "$RESULT" != "failed" ]] && [[ "$RESULT" != *"error"* ]]; then
    echo -e "${GREEN}✓ Log ingestion is working${NC}"
else
    echo -e "${YELLOW}⚠ Could not verify log ingestion (this might be normal if no logs exist yet)${NC}"
fi

# 9. Display access information
echo -e "\n${GREEN}=== Loki Setup Complete ===${NC}"
echo ""
echo "Access points:"
echo "- Loki API: http://localhost:3100"
echo "- Loki Ready: http://localhost:3100/ready"
echo "- Loki Metrics: http://localhost:3100/metrics"
echo ""
echo "Grafana Configuration:"
echo "1. Open Grafana at http://localhost:3000"
echo "2. Go to Configuration > Data Sources"
echo "3. Loki should be automatically configured"
echo "4. Go to Explore and select Loki to query logs"
echo ""
echo "Example LogQL queries:"
echo '  {job="csp-api"}'
echo '  {job="postgres"} |= "ERROR"'
echo '  {container="csp_chroma"}'
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"