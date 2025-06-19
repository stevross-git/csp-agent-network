# File: tests/conftest.py
"""
Test Configuration and Fixtures
==============================
Pytest configuration and shared fixtures for the CSP Visual Designer backend
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from uuid import uuid4
import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient
import redis.asyncio as redis

# Import application components
from backend.main import app
from backend.models.database_models import Base, User, Design, DesignNode, DesignConnection
from backend.database.connection import get_db_session, get_redis_client
from backend.auth.auth_system import AuthenticationService, PasswordManager
from backend.config.settings import configure_for_testing
from backend.components.registry import component_registry


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Configure for testing
configure_for_testing()

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False}
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    TestSessionLocal = sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with TestSessionLocal() as session:
        yield session

@pytest_asyncio.fixture
async def test_redis():
    """Create test Redis client"""
    # Use fakeredis for testing
    try:
        import fakeredis.aioredis
        redis_client = fakeredis.aioredis.FakeRedis()
        yield redis_client
        await redis_client.close()
    except ImportError:
        # Fallback to real Redis if fakeredis not available
        redis_client = redis.from_url("redis://localhost:6379/15")  # Use test DB
        await redis_client.flushdb()
        yield redis_client
        await redis_client.flushdb()
        await redis_client.close()

# =============================================================================
# APPLICATION FIXTURES
# =============================================================================

@pytest_asyncio.fixture
async def test_client(db_session: AsyncSession, test_redis):
    """Create test client with overridden dependencies"""
    
    async def override_get_db_session():
        yield db_session
    
    async def override_get_redis_client():
        return test_redis
    
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = override_get_redis_client
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    # Cleanup
    app.dependency_overrides.clear()

@pytest.fixture
def sync_client():
    """Synchronous test client for simple tests"""
    return TestClient(app)

# =============================================================================
# USER AND AUTHENTICATION FIXTURES
# =============================================================================

@pytest_asyncio.fixture
async def password_manager():
    """Password manager instance"""
    return PasswordManager()

@pytest_asyncio.fixture
async def auth_service(test_redis):
    """Authentication service instance"""
    return AuthenticationService(test_redis)

@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession, password_manager: PasswordManager):
    """Create test user"""
    hashed_password = password_manager.hash_password("testpassword123")
    
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=hashed_password,
        full_name="Test User",
        is_active=True
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user

@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession, password_manager: PasswordManager):
    """Create admin user"""
    hashed_password = password_manager.hash_password("adminpassword123")
    
    user = User(
        username="admin",
        email="admin@example.com",
        hashed_password=hashed_password,
        full_name="Admin User",
        is_active=True,
        is_admin=True
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user

@pytest_asyncio.fixture
async def user_token(test_client: AsyncClient, test_user: User):
    """Get authentication token for test user"""
    response = await test_client.post(
        "/api/auth/login",
        json={
            "username": test_user.username,
            "password": "testpassword123"
        }
    )
    
    assert response.status_code == 200
    token_data = response.json()
    return token_data["access_token"]

@pytest_asyncio.fixture
async def admin_token(test_client: AsyncClient, admin_user: User):
    """Get authentication token for admin user"""
    response = await test_client.post(
        "/api/auth/login",
        json={
            "username": admin_user.username,
            "password": "adminpassword123"
        }
    )
    
    assert response.status_code == 200
    token_data = response.json()
    return token_data["access_token"]

@pytest_asyncio.fixture
async def auth_headers(user_token: str):
    """Authentication headers for requests"""
    return {"Authorization": f"Bearer {user_token}"}

@pytest_asyncio.fixture
async def admin_headers(admin_token: str):
    """Admin authentication headers for requests"""
    return {"Authorization": f"Bearer {admin_token}"}

# =============================================================================
# DESIGN FIXTURES
# =============================================================================

@pytest_asyncio.fixture
async def test_design(db_session: AsyncSession, test_user: User):
    """Create test design"""
    design = Design(
        name="Test Design",
        description="A test design for testing",
        version="1.0.0",
        created_by=test_user.id,
        canvas_settings={
            "width": 1200,
            "height": 800,
            "zoom": 1.0
        }
    )
    
    db_session.add(design)
    await db_session.commit()
    await db_session.refresh(design)
    
    return design

@pytest_asyncio.fixture
async def test_design_with_nodes(db_session: AsyncSession, test_design: Design):
    """Create test design with nodes and connections"""
    # Create nodes
    node1 = DesignNode(
        design_id=test_design.id,
        node_id="node_1",
        component_type="ai_agent",
        position_x=100,
        position_y=100,
        properties={"model": "gpt-3.5-turbo", "temperature": 0.7}
    )
    
    node2 = DesignNode(
        design_id=test_design.id,
        node_id="node_2",
        component_type="data_processor",
        position_x=300,
        position_y=100,
        properties={"operation": "transform"}
    )
    
    db_session.add_all([node1, node2])
    await db_session.commit()
    
    # Create connection
    connection = DesignConnection(
        design_id=test_design.id,
        connection_id="conn_1",
        from_node_id="node_1",
        to_node_id="node_2",
        connection_type="data_flow"
    )
    
    db_session.add(connection)
    await db_session.commit()
    
    return test_design

# =============================================================================
# COMPONENT FIXTURES
# =============================================================================

@pytest_asyncio.fixture
async def mock_ai_service():
    """Mock AI service for testing"""
    class MockAIService:
        async def generate_response(self, messages, **kwargs):
            return {
                "content": "Mock AI response",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                },
                "model": "mock-model",
                "finish_reason": "stop"
            }
        
        async def generate_streaming_response(self, messages, **kwargs):
            for chunk in ["Mock ", "AI ", "response"]:
                yield chunk
    
    return MockAIService()

# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def temp_file():
    """Create temporary file"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('{"test": "data"}')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

@pytest.fixture
def sample_design_data():
    """Sample design data for testing"""
    return {
        "name": "Sample Design",
        "description": "A sample design for testing",
        "version": "1.0.0",
        "canvas_settings": {
            "width": 1200,
            "height": 800,
            "zoom": 1.0,
            "grid_enabled": True
        }
    }

@pytest.fixture
def sample_node_data():
    """Sample node data for testing"""
    return {
        "node_id": "test_node_1",
        "component_type": "ai_agent",
        "position": {"x": 100, "y": 200},
        "size": {"width": 150, "height": 100},
        "properties": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "visual_style": {
            "color": "#4CAF50",
            "border_color": "#2E7D32"
        }
    }

@pytest.fixture
def sample_connection_data():
    """Sample connection data for testing"""
    return {
        "connection_id": "test_conn_1",
        "from_node_id": "node_1",
        "to_node_id": "node_2",
        "from_port": "output",
        "to_port": "input",
        "connection_type": "data_flow",
        "properties": {
            "data_type": "text",
            "buffer_size": 1000
        },
        "visual_style": {
            "color": "#2196F3",
            "width": 2
        }
    }

# =============================================================================
# EVENT LOOP CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# =============================================================================
# PARAMETRIZED FIXTURES
# =============================================================================

@pytest.fixture(params=["ai_agent", "data_processor", "input_validator"])
def component_type(request):
    """Parametrized component types for testing"""
    return request.param

@pytest.fixture(params=[200, 400, 401, 403, 404, 500])
def http_status_code(request):
    """Parametrized HTTP status codes for testing"""
    return request.param

# =============================================================================
# PERFORMANCE FIXTURES
# =============================================================================

@pytest.fixture
def performance_timer():
    """Performance timing utility"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest_asyncio.fixture(autouse=True)
async def cleanup_database(db_session: AsyncSession):
    """Cleanup database after each test"""
    yield
    
    # Rollback any pending transactions
    try:
        await db_session.rollback()
    except Exception:
        pass

@pytest_asyncio.fixture(autouse=True)
async def cleanup_redis(test_redis):
    """Cleanup Redis after each test"""
    yield
    
    try:
        await test_redis.flushdb()
    except Exception:
        pass

# =============================================================================
# EXAMPLE TEST FILES
# =============================================================================

# File: tests/test_auth.py
"""
Authentication Tests
===================
"""

import pytest
from fastapi import status

class TestAuthentication:
    """Test authentication functionality"""
    
    async def test_user_registration(self, test_client):
        """Test user registration"""
        registration_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "SecurePassword123",
            "full_name": "New User"
        }
        
        response = await test_client.post("/api/auth/register", json=registration_data)
        assert response.status_code == status.HTTP_200_OK
        
        user_data = response.json()
        assert user_data["username"] == "newuser"
        assert user_data["email"] == "newuser@example.com"
    
    async def test_user_login(self, test_client, test_user):
        """Test user login"""
        login_data = {
            "username": test_user.username,
            "password": "testpassword123"
        }
        
        response = await test_client.post("/api/auth/login", json=login_data)
        assert response.status_code == status.HTTP_200_OK
        
        token_data = response.json()
        assert "access_token" in token_data
        assert "refresh_token" in token_data
        assert token_data["token_type"] == "bearer"
    
    async def test_invalid_login(self, test_client, test_user):
        """Test login with invalid credentials"""
        login_data = {
            "username": test_user.username,
            "password": "wrongpassword"
        }
        
        response = await test_client.post("/api/auth/login", json=login_data)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

# File: tests/test_designs.py
"""
Design Management Tests
======================
"""

import pytest
from fastapi import status

class TestDesigns:
    """Test design management functionality"""
    
    async def test_create_design(self, test_client, auth_headers, sample_design_data):
        """Test design creation"""
        response = await test_client.post(
            "/api/designs",
            json=sample_design_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        
        design = response.json()
        assert design["name"] == sample_design_data["name"]
        assert design["description"] == sample_design_data["description"]
    
    async def test_get_design(self, test_client, auth_headers, test_design):
        """Test getting a design"""
        response = await test_client.get(
            f"/api/designs/{test_design.id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        design = response.json()
        assert design["id"] == str(test_design.id)
        assert design["name"] == test_design.name
    
    async def test_update_design(self, test_client, auth_headers, test_design):
        """Test updating a design"""
        update_data = {
            "name": "Updated Design Name",
            "description": "Updated description"
        }
        
        response = await test_client.put(
            f"/api/designs/{test_design.id}",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        design = response.json()
        assert design["name"] == update_data["name"]
        assert design["description"] == update_data["description"]
    
    async def test_delete_design(self, test_client, auth_headers, test_design):
        """Test deleting a design"""
        response = await test_client.delete(
            f"/api/designs/{test_design.id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    async def test_add_node_to_design(self, test_client, auth_headers, test_design, sample_node_data):
        """Test adding a node to a design"""
        response = await test_client.post(
            f"/api/designs/{test_design.id}/nodes",
            json=sample_node_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        
        node = response.json()
        assert node["node_id"] == sample_node_data["node_id"]
        assert node["component_type"] == sample_node_data["component_type"]

# File: tests/test_components.py
"""
Component Registry Tests
=======================
"""

import pytest

class TestComponents:
    """Test component registry functionality"""
    
    async def test_list_components(self, test_client):
        """Test listing available components"""
        response = await test_client.get("/api/components")
        assert response.status_code == status.HTTP_200_OK
        
        components = response.json()
        assert isinstance(components, dict)
        assert len(components) > 0
    
    async def test_get_component_info(self, test_client):
        """Test getting component information"""
        response = await test_client.get("/api/components/ai_agent")
        assert response.status_code == status.HTTP_200_OK
        
        component = response.json()
        assert component["component_type"] == "ai_agent"
        assert "input_ports" in component
        assert "output_ports" in component

# File: tests/test_performance.py
"""
Performance Tests
================
"""

import pytest
import asyncio

class TestPerformance:
    """Test system performance"""
    
    async def test_concurrent_requests(self, test_client, auth_headers, performance_timer):
        """Test handling concurrent requests"""
        performance_timer.start()
        
        tasks = []
        for i in range(10):
            task = test_client.get("/api/components", headers=auth_headers)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        performance_timer.stop()
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
        
        # Should complete in reasonable time
        assert performance_timer.elapsed < 5.0
    
    @pytest.mark.parametrize("num_designs", [1, 5, 10])
    async def test_bulk_design_creation(self, test_client, auth_headers, num_designs):
        """Test creating multiple designs"""
        tasks = []
        
        for i in range(num_designs):
            design_data = {
                "name": f"Bulk Design {i}",
                "description": f"Bulk design number {i}"
            }
            
            task = test_client.post("/api/designs", json=design_data, headers=auth_headers)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_201_CREATED

# File: tests/load/locustfile.py
"""
Load Testing with Locust
========================
"""

from locust import HttpUser, task, between

class CSPUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login before starting tasks"""
        response = self.client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "testpassword123"
        })
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}
    
    @task(3)
    def list_components(self):
        """List available components"""
        self.client.get("/api/components", headers=self.headers)
    
    @task(2)
    def create_design(self):
        """Create a new design"""
        design_data = {
            "name": f"Load Test Design {self.environment.runner.user_count}",
            "description": "Design created during load test"
        }
        
        response = self.client.post("/api/designs", json=design_data, headers=self.headers)
        
        if response.status_code == 201:
            design_id = response.json()["id"]
            # Clean up - delete the design
            self.client.delete(f"/api/designs/{design_id}", headers=self.headers)
    
    @task(1)
    def health_check(self):
        """Check system health"""
        self.client.get("/health")
    
    @task(1)
    def websocket_connection(self):
        """Test WebSocket connection"""
        # Note: Locust doesn't natively support WebSocket,
        # but this can be extended with custom WebSocket client