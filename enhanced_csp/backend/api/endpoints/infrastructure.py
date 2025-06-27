# backend/api/endpoints/infrastructure.py
"""Infrastructure management endpoints"""

from fastapi import APIRouter
from typing import Dict, Any, List
from datetime import datetime
import uuid
import subprocess

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    from backend.ai.ai_coordination_engine import coordination_engine
    AI_COORDINATION_AVAILABLE = True
except Exception:
    coordination_engine = None
    AI_COORDINATION_AVAILABLE = False

router = APIRouter(prefix="/api/infrastructure", tags=["infrastructure"])

# ---------------------------------------------------------------------------
# Mock infrastructure data
# ---------------------------------------------------------------------------

infrastructure_services = [
    {"name": "Web Server", "status": "running", "uptime": "1d 0h", "port": 8000},
    {"name": "Database", "status": "running", "uptime": "1d 0h", "port": 5432},
    {"name": "Redis Cache", "status": "running", "uptime": "1d 0h", "port": 6379},
]

infrastructure_alerts: List[Dict[str, Any]] = []
infrastructure_backups: List[Dict[str, Any]] = []
maintenance_mode = False

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def generate_infrastructure_metrics() -> Dict[str, Any]:
    """Generate sample infrastructure metrics."""
    return {
        "cpu": {
            "current": psutil.cpu_percent() if PSUTIL_AVAILABLE else 0,
            "max": 100,
            "unit": "%",
        },
        "memory": {
            "current": psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0,
            "max": 100,
            "unit": "%",
        },
        "disk": {
            "current": psutil.disk_usage('/') .percent if PSUTIL_AVAILABLE else 0,
            "max": 100,
            "unit": "%",
        },
        "uptime": {
            "current": 99.9,
            "max": 100,
            "unit": "%",
        },
    }

def get_infrastructure_status() -> Dict[str, Any]:
    return {
        "message": "Maintenance mode" if maintenance_mode else "All systems operational",
        "timestamp": datetime.utcnow().isoformat(),
        "health": "maintenance" if maintenance_mode else "healthy",
        "maintenance_mode": maintenance_mode,
        "services": {svc["name"]: svc["status"] for svc in infrastructure_services},
    }

async def get_agents_data() -> Dict[str, Any]:
    if AI_COORDINATION_AVAILABLE and coordination_engine:
        try:
            status = await coordination_engine.get_system_status()
            return {
                "registered_agents": status.get("registered_agents", 0),
                "active_sessions": status.get("active_sessions", 0),
                "coordination_sessions": status.get("coordination_sessions", 0),
            }
        except Exception:
            return {"registered_agents": 0}
    return {"registered_agents": 0}

def get_node_server_info() -> Dict[str, Any]:
    info = {}
    try:
        version = subprocess.check_output(["node", "-v"], text=True).strip()
        info["version"] = version
    except Exception:
        info["version"] = None
    processes = []
    if PSUTIL_AVAILABLE:
        for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            name = proc.info.get("name") or ""
            if "node" in name.lower():
                processes.append({
                    "pid": proc.info["pid"],
                    "cmdline": " ".join(proc.info.get("cmdline", [])),
                })
    info["processes"] = processes
    return info

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
async def api_infrastructure_status():
    return get_infrastructure_status()

@router.get("/metrics")
async def api_infrastructure_metrics():
    return generate_infrastructure_metrics()

@router.get("/services")
async def api_infrastructure_services():
    return infrastructure_services

@router.get("/alerts")
async def api_infrastructure_alerts():
    return infrastructure_alerts

@router.post("/backup")
async def api_create_backup(payload: Dict[str, Any]):
    backup = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "size": payload.get("size", 0),
        "name": payload.get("name", "backup"),
    }
    infrastructure_backups.insert(0, backup)
    return backup

@router.get("/backups")
async def api_list_backups():
    return infrastructure_backups

@router.post("/backup/{backup_id}/restore")
async def api_restore_backup(backup_id: str):
    return {"id": backup_id, "status": "restored"}

@router.post("/services/{service}/restart")
async def api_restart_service(service: str):
    for svc in infrastructure_services:
        if svc["name"] == service:
            svc["status"] = "running"
    return {"service": service, "status": "restarted"}

@router.post("/services/{service}/stop")
async def api_stop_service(service: str):
    for svc in infrastructure_services:
        if svc["name"] == service:
            svc["status"] = "stopped"
    return {"service": service, "status": "stopped"}

@router.post("/services/{service}/start")
async def api_start_service(service: str):
    for svc in infrastructure_services:
        if svc["name"] == service:
            svc["status"] = "running"
    return {"service": service, "status": "started"}

@router.post("/maintenance")
async def api_toggle_maintenance(payload: Dict[str, Any]):
    global maintenance_mode
    maintenance_mode = bool(payload.get("enabled", False))
    return {"maintenance_mode": maintenance_mode}

@router.post("/emergency-shutdown")
async def api_emergency_shutdown():
    return {"status": "shutdown_initiated"}

@router.get("/logs/export")
async def api_logs_export():
    return {"url": "/static/logs/latest.log", "filename": "logs.txt"}

@router.get("/node-info")
async def api_node_info():
    return get_node_server_info()

@router.get("/agents")
async def api_agent_info():
    return await get_agents_data()
