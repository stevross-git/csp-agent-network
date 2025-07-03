"""
WebSocket security tests
"""

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from backend.realtime.secure_websocket import SecureWebSocketManager
import jwt

@pytest.fixture
def app_with_secure_ws():
    """FastAPI app with secure WebSocket"""
    app = FastAPI()
    
    ws_manager = SecureWebSocketManager(
        allowed_origins=["http://localhost:3000", "https://example.com"],
        jwt_secret="test-secret",
        max_message_size=1024
    )
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        client_id = await ws_manager.connect(websocket)
        if not client_id:
            return
        
        try:
            while True:
                data = await websocket.receive_text()
                await ws_manager.handle_message(client_id, data)
        except:
            await ws_manager.disconnect(client_id)
    
    return app

def test_websocket_rejects_invalid_origin(app_with_secure_ws):
    """Test WebSocket rejects connections from invalid origins"""
    client = TestClient(app_with_secure_ws)
    
    # Invalid origin
    with pytest.raises(Exception):
        with client.websocket_connect(
            "/ws",
            headers={"Origin": "http://evil.com"}
        ) as websocket:
            # Should not reach here
            pass

def test_websocket_accepts_valid_origin(app_with_secure_ws):
    """Test WebSocket accepts valid origins"""
    client = TestClient(app_with_secure_ws)
    
    with client.websocket_connect(
        "/ws",
        headers={"Origin": "http://localhost:3000"}
    ) as websocket:
        # Send auth message
        websocket.send_json({
            "type": "ping"
        })
        
        response = websocket.receive_json()
        assert response["type"] == "pong"

def test_websocket_message_size_limit(app_with_secure_ws):
    """Test WebSocket enforces message size limits"""
    client = TestClient(app_with_secure_ws)
    
    with client.websocket_connect(
        "/ws",
        headers={"Origin": "http://localhost:3000"}
    ) as websocket:
        # Send oversized message
        large_message = "x" * 2048  # Over 1KB limit
        websocket.send_text(large_message)
        
        # Should disconnect
        with pytest.raises(Exception):
            websocket.receive_json()

def test_websocket_requires_auth_for_messages(app_with_secure_ws):
    """Test WebSocket requires authentication for sensitive operations"""
    client = TestClient(app_with_secure_ws)
    
    with client.websocket_connect(
        "/ws",
        headers={"Origin": "http://localhost:3000"}
    ) as websocket:
        # Try to send message without auth
        websocket.send_json({
            "type": "message",
            "data": {"content": "test"}
        })
        
        response = websocket.receive_json()
        assert response["type"] == "error"
        assert "Authentication required" in response["data"]["error"]