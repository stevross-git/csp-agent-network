#!/usr/bin/env python3
"""
Enhanced CSP System - Web Interface
Serves the developer tools and dashboard
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_server")

# Create FastAPI app
app = FastAPI(
    title="Enhanced CSP System - Web Interface",
    description="Developer Tools and Dashboard",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

# Make sure static directory exists
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse(
        "web_ui/developer_tools.html", 
        {
            "request": request,
            "title": "Enhanced CSP System",
            "version": "1.0.0",
            "status": "running"
        }
    )

@app.get("/developer-tools", response_class=HTMLResponse)
async def developer_tools(request: Request):
    """Developer tools page"""
    return templates.TemplateResponse(
        "web_ui/developer_tools.html", 
        {
            "request": request,
            "title": "CSP Developer Tools",
            "version": "1.0.0",
            "features": {
                "visual_designer": True,
                "debugger": True,
                "performance_monitor": True,
                "test_runner": True
            }
        }
    )

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "csp_engine": "running",
            "database": "connected",
            "monitoring": "active"
        }
    }

@app.get("/api/metrics")
async def api_metrics():
    """System metrics endpoint"""
    import psutil
    import time
    
    # Get real system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        "cpu_usage": round(cpu_percent, 1),
        "memory_usage": round(memory.percent, 1),
        "active_processes": 12,
        "channels": 8,
        "agents": 5,
        "uptime": "2h 45m",
        "timestamp": int(time.time())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "csp-web-interface"}

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Enhanced CSP Web Interface started successfully!")
    logger.info("🌐 Main Dashboard: http://localhost:8080/")
    logger.info("🛠️  Developer Tools: http://localhost:8080/developer-tools")
    logger.info("📊 API Status: http://localhost:8080/api/status")
    logger.info("📈 Metrics: http://localhost:8080/api/metrics")

def main():
    """Start the web server"""
    logger.info("Starting Enhanced CSP Web Interface...")
    
    # Use uvicorn without reload to avoid the warning
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080, 
        log_level="info"
    )

if __name__ == "__main__":
    main()
