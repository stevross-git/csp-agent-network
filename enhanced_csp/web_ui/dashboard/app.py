from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging

def create_dashboard_app():
    """Create a simple dashboard sub-application"""
    from fastapi import FastAPI
    
    dashboard = FastAPI(title="CSP Dashboard", version="2.0.0")
    
    @dashboard.get("/")
    async def dashboard_home():
        """Dashboard home redirects to main dashboard page"""
        return {
            "message": "CSP Dashboard", 
            "status": "running",
            "redirect": "/dashboard"
        }
    
    @dashboard.get("/health")
    async def dashboard_health():
        """Dashboard health check"""
        return {"status": "healthy", "service": "csp-dashboard"}
    
    return dashboard
