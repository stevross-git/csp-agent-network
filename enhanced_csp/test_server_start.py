#!/usr/bin/env python3
"""
Simple test to verify the server can start without errors
"""
import sys
import os
sys.path.insert(0, '.')

def test_server_import():
    """Test that the server can be imported"""
    try:
        # Import the main module
        from backend.main import app
        print("✅ FastAPI app created successfully")
        
        # Check if CORS is configured
        cors_found = False
        for middleware in app.user_middleware:
            if 'CORSMiddleware' in str(middleware):
                cors_found = True
                break
        
        if cors_found:
            print("✅ CORS middleware is configured")
        else:
            print("⚠️ CORS middleware not found")
        
        print("✅ Server import test passed")
        return True
        
    except Exception as e:
        print(f"❌ Server import test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_server_import()
    sys.exit(0 if success else 1)
