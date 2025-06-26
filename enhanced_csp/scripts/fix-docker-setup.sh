#!/bin/bash
# fix-docker-setup.sh
# Quick fix for the YAML syntax error and setup issues

echo "🔧 Quick Fix for Enhanced CSP Docker Setup"
echo "=========================================="

# Step 1: Backup the broken docker-compose.yml
if [ -f "docker-compose.yml" ]; then
    echo "📋 Backing up broken docker-compose.yml..."
    mv docker-compose.yml docker-compose.yml.broken.$(date +%Y%m%d_%H%M%S)
fi

# Step 2: Create a clean docker-compose.yml
echo "📝 Creating clean docker-compose.yml..."
cat > docker-compose.yml << 'EOF'

EOF

echo "✅ Clean docker-compose.yml created"

# Step 3: Test the YAML syntax
echo "🔍 Testing YAML syntax..."
if docker-compose config >/dev/null 2>&1; then
    echo "✅ YAML syntax is valid"
else
    echo "❌ YAML syntax still has issues"
    docker-compose config
    exit 1
fi

# Step 4: Stop any running containers and clean up
echo "🧹 Cleaning up any existing containers..."
docker-compose down -v 2>/dev/null || echo "No containers to stop"

# Step 5: Start the core databases first
echo "🚀 Starting core databases..."
docker-compose up -d postgres postgres_ai_models postgres_vector redis

# Wait for databases to start
echo "⏳ Waiting for databases to initialize..."
sleep 30

# Step 6: Check database health
echo "🔍 Checking database health..."
echo -n "PostgreSQL (Main): "
if docker-compose exec postgres pg_isready -U csp_user -d csp_visual_designer >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready - Let's check logs"
    docker-compose logs postgres | tail -10
fi

echo -n "PostgreSQL (AI Models): "
if docker-compose exec postgres_ai_models pg_isready -U ai_models_user -d ai_models_db >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready"
fi

echo -n "PostgreSQL (Vector): "
if docker-compose exec postgres_vector pg_isready -U vector_user -d vector_db >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready"
fi

echo -n "Redis: "
if docker-compose exec redis redis-cli ping >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready"
fi

# Step 7: Start vector databases
echo "🧠 Starting vector databases..."
docker-compose up -d chroma qdrant weaviate

# Step 8: Start admin tools
echo "🛠️ Starting admin tools..."
docker-compose up -d pgadmin redis-insight

echo ""
echo "✅ Quick fix completed!"
echo ""
echo "📊 Access Points:"
echo "- PostgreSQL: localhost:5432"
echo "- AI Models DB: localhost:5433"
echo "- Vector DB: localhost:5434"
echo "- Redis: localhost:6379"
echo "- Chroma: localhost:8200"
echo "- Qdrant: localhost:6333"
echo "- Weaviate: localhost:8080"
echo "- pgAdmin: http://localhost:5050"
echo "- Redis Insight: http://localhost:8001"
echo ""
echo "🔍 To check status: docker-compose ps"
echo "📜 To view logs: docker-compose logs [service_name]"
EOF