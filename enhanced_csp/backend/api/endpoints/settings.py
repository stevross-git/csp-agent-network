from fastapi import APIRouter, Depends
from typing import Dict, Any, List

from backend.config.settings import settings, reload_configuration

router = APIRouter(prefix="/api/settings", tags=["settings"])

@router.get("/", response_model=Dict[str, Any])
async def read_settings():
    return {"settings": settings.model_dump()}

@router.put("/", response_model=Dict[str, str])
async def update_settings(payload: Dict[str, List[Dict[str, Any]]]):
    for item in payload.get("settings", []):
        if hasattr(settings, item["key"]):
            setattr(settings, item["key"], item["value"])
    reload_configuration()
    return {"message": "updated"}
