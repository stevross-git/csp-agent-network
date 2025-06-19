from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging

def create_dashboard_app():
    dashboard = FastAPI(title="CSP Dashboard", version="2.0.0")
    
    # Setup templates
    templates_dir = Path("templates")
    if templates_dir.exists():
        templates = Jinja2Templates(directory=str(templates_dir))
    else:
        templates = None
    
    @dashboard.get("/", response_class=HTMLResponse)
    async def dashboard_home(request: Request):
        if templates:
            return templates.TemplateResponse(
                "web_ui/developer_tools.html", 
                {"request": request, "title": "Enhanced CSP System", "version": "2.0.0", "status": "running"}
            )
        else:
            return HTMLResponse("<h1>🚀 CSP Dashboard</h1><p>Dashboard is running!</p>")
    
    logging.getLogger(__name__).info("✅ Dashboard created - loads developer_tools.html")
    return dashboard
