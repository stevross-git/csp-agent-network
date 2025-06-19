#!/bin/bash
# CSP Visual Designer Backend - Quick Start Guide
# ==============================================
# This script will get your backend up and running in minutes!

set -e

echo "🚀 CSP Visual Designer Backend - Quick Start"
echo "============================================"
echo ""

# Step 1: Create project structure
echo "📁 Creating project structure..."
mkdir -p csp-visual-designer/{backend,cli,tests,deploy,config,monitoring,k8s,static,logs,data}
cd csp-visual-designer

# Step 2: Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 3: Create requirements.txt (abbreviated for quick start)
cat > requirements.txt << 'EOF'
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2

# Database
sqlalchemy==2.0.23
alembic==1.12.1
asyncpg==0.29.0
aiosqlite==0.19.0

# Caching & Sessions
redis==5.0.1
aioredis==2.0.1

# Authentication
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
bcrypt==4.0.1

# HTTP & WebSocket
httpx==0.25.1
websockets==12.0

# AI Integration
openai==1.3.5
anthropic==0.7.7

# Monitoring
prometheus-client==0.19.0
psutil==5.9.6

# Configuration
python-dotenv==1.0.0
pydantic-settings==2.0.3
pyyaml==6.0.1

# CLI Tools
click==8.1.7
rich==13.7.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.1
EOF

# Step 4: Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Step 5: Create environment configuration
echo "⚙️ Creating environment configuration..."
cat > .env << 'EOF'
# Application
ENVIRONMENT=development
DEBUG=true
APP_NAME=CSP Visual Designer API
SECRET_KEY=dev-secret-key-change-in-production

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=csp_visual_designer
DB_USER=csp_user
DB_PASSWORD=csp_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# AI Services (add your API keys)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
EOF

# Step 6: Create minimal Docker Compose for development
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: csp_visual_designer
      POSTGRES_USER: csp_user
      POSTGRES_PASSWORD: csp_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
EOF

# Step 7: Create basic alembic configuration
cat > alembic.ini << 'EOF'
[alembic]
script_location = alembic
sqlalchemy.url = postgresql://csp_user:csp_password@localhost:5432/csp_visual_designer

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
EOF

# Step 8: Create placeholder files for the artifacts we created
echo "📝 Creating backend structure..."

# Create the main FastAPI app file
mkdir -p backend
cat > backend/__init__.py << 'EOF'
"""CSP Visual Designer Backend"""
__version__ = "2.0.0"
EOF

cat > backend/main.py << 'EOF'
"""
CSP Visual Designer Backend - Minimal FastAPI Application
========================================================
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CSP Visual Designer API",
    description="Advanced AI-Powered CSP Process Designer Backend",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "CSP Visual Designer API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "csp-visual-designer-api"}

# Add your artifact code here!
# Copy the database models, API endpoints, auth system, etc.
# from the artifacts created in our conversation.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Step 9: Create a simple test
mkdir -p tests
cat > tests/test_basic.py << 'EOF'
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "CSP Visual Designer API" in response.json()["message"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
EOF

# Step 10: Create CLI entry point
mkdir -p cli
cat > cli/manage.py << 'EOF'
#!/usr/bin/env python3
"""
CSP Visual Designer Management CLI
"""

import click

@click.group()
def cli():
    """CSP Visual Designer Management CLI"""
    pass

@cli.command()
def start():
    """Start the development server"""
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

@cli.command()
def test():
    """Run tests"""
    import subprocess
    subprocess.run(["python", "-m", "pytest", "tests/", "-v"])

if __name__ == "__main__":
    cli()
EOF

chmod +x cli/manage.py

# Step 11: Start databases
echo "🗄️ Starting databases..."
docker-compose up -d postgres redis

# Wait for databases to be ready
echo "⏳ Waiting for databases to be ready..."
sleep 10

# Step 12: Test the basic setup
echo "🧪 Running basic tests..."
python -m pytest tests/ -v

# Step 13: Start the development server in background for testing
echo "🚀 Starting development server..."
python backend/main.py &
SERVER_PID=$!

# Wait a moment for server to start
sleep 3

# Test the API
echo "✅ Testing API endpoints..."
curl -f http://localhost:8000/ || echo "❌ API test failed"
curl -f http://localhost:8000/health || echo "❌ Health check failed"

# Stop the test server
kill $SERVER_PID 2>/dev/null || true

echo ""
echo "🎉 Quick Start Setup Complete!"
echo "==============================="
echo ""
echo "✅ Project structure created"
echo "✅ Virtual environment set up"
echo "✅ Dependencies installed"
echo "✅ Databases running"
echo "✅ Basic tests passing"
echo ""
echo "🎯 Next Steps:"
echo "1. Copy the database models from the artifacts to backend/models/"
echo "2. Copy the API endpoints from the artifacts to backend/api/"
echo "3. Copy the authentication system from the artifacts to backend/auth/"
echo "4. Copy the component registry from the artifacts to backend/components/"
echo "5. Copy the execution engine from the artifacts to backend/execution/"
echo "6. Copy the WebSocket manager from the artifacts to backend/realtime/"
echo "7. Copy the AI integration from the artifacts to backend/ai/"
echo "8. Copy the monitoring system from the artifacts to backend/monitoring/"
echo ""
echo "🚀 To start development:"
echo "  source venv/bin/activate"
echo "  python cli/manage.py start"
echo ""
echo "📚 To view API docs:"
echo "  http://localhost:8000/docs"
echo ""
echo "🔧 To manage the system:"
echo "  python cli/manage.py --help"
echo ""
echo "💡 All the artifacts created in our conversation contain the complete"
echo "   implementation. Copy them into the appropriate directories to get"
echo "   the full-featured backend running!"
echo ""
echo "Happy coding! 🚀"

# ============================================================================
# ARTIFACT IMPLEMENTATION CHECKLIST
# ============================================================================

cat > IMPLEMENTATION_CHECKLIST.md << 'EOF'
# CSP Visual Designer Backend - Implementation Checklist

## 📋 Step-by-Step Implementation Guide

### Phase 1: Core Infrastructure ✅
- [x] Project structure created
- [x] Virtual environment set up
- [x] Dependencies installed
- [ ] **TODO**: Copy `database_models.py` to `backend/models/`
- [ ] **TODO**: Copy `api_schemas.py` to `backend/schemas/`
- [ ] **TODO**: Copy `connection.py` to `backend/database/`
- [ ] **TODO**: Copy `settings.py` to `backend/config/`

### Phase 2: Authentication & Security
- [ ] **TODO**: Copy `auth_system.py` to `backend/auth/`
- [ ] **TODO**: Set up JWT secret keys in `.env`
- [ ] **TODO**: Configure rate limiting
- [ ] **TODO**: Test user registration and login

### Phase 3: API Endpoints
- [ ] **TODO**: Copy `designs.py` to `backend/api/endpoints/`
- [ ] **TODO**: Update `main.py` with all artifacts
- [ ] **TODO**: Test CRUD operations
- [ ] **TODO**: Verify authentication on protected endpoints

### Phase 4: Component System
- [ ] **TODO**: Copy `registry.py` to `backend/components/`
- [ ] **TODO**: Test component listing and creation
- [ ] **TODO**: Verify component validation

### Phase 5: Execution Engine
- [ ] **TODO**: Copy `execution_engine.py` to `backend/execution/`
- [ ] **TODO**: Test design execution
- [ ] **TODO**: Verify performance monitoring

### Phase 6: Real-time Features
- [ ] **TODO**: Copy `websocket_manager.py` to `backend/realtime/`
- [ ] **TODO**: Test WebSocket connections
- [ ] **TODO**: Verify real-time collaboration

### Phase 7: AI Integration
- [ ] **TODO**: Copy `ai_integration.py` to `backend/ai/`
- [ ] **TODO**: Configure AI provider API keys
- [ ] **TODO**: Test AI components
- [ ] **TODO**: Monitor usage and costs

### Phase 8: Monitoring & Performance
- [ ] **TODO**: Copy `performance.py` to `backend/monitoring/`
- [ ] **TODO**: Set up Prometheus metrics
- [ ] **TODO**: Configure alerting
- [ ] **TODO**: Test health checks

### Phase 9: Testing
- [ ] **TODO**: Copy test files to `tests/` directory
- [ ] **TODO**: Run full test suite
- [ ] **TODO**: Set up CI/CD pipeline
- [ ] **TODO**: Configure load testing

### Phase 10: Deployment
- [ ] **TODO**: Copy Docker configurations
- [ ] **TODO**: Set up production environment
- [ ] **TODO**: Configure monitoring dashboards
- [ ] **TODO**: Set up backup procedures

## 🔧 Quick Commands

```bash
# Start development
source venv/bin/activate
python cli/manage.py start

# Run tests
python -m pytest tests/ -v

# Database operations
python cli/manage.py db migrate
python cli/manage.py users create-admin

# Check system status
python cli/manage.py server status
python cli/manage.py monitor performance
```

## 📚 Documentation Links

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

## 🆘 Need Help?

1. Check the artifacts created in our conversation
2. Review the complete implementation guide
3. Run the health checks and monitoring
4. Check the logs for any errors

All the code you need is in the artifacts! 🚀
EOF

echo ""
echo "📋 Implementation checklist created: IMPLEMENTATION_CHECKLIST.md"
echo ""
echo "🎯 You now have everything you need to build a production-ready"
echo "   CSP Visual Designer backend! All the artifacts contain the"
echo "   complete, working implementation."