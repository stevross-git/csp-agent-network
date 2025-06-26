# backend/api/ai_models.py
"""
AI Models API Endpoints
=======================
RESTful API for managing AI models with real database integration
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from backend.database.ai_models_db import (
    ai_models_db, AIModel, ModelStatus, ModelType, ModelUsageLog,
    initialize_ai_models_db
)

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class ModelCreateRequest(BaseModel):
    """Request schema for creating a new AI model"""
    name: str = Field(..., min_length=1, max_length=255)
    model_type: ModelType
    provider: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., min_length=1, max_length=50)
    endpoint_url: Optional[str] = None
    api_key_name: Optional[str] = None
    max_tokens: Optional[int] = Field(None, gt=0)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    context_window: Optional[int] = Field(None, gt=0)
    cost_per_1k_tokens: Optional[float] = Field(None, ge=0.0)
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class ModelUpdateRequest(BaseModel):
    """Request schema for updating an AI model"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[ModelStatus] = None
    endpoint_url: Optional[str] = None
    api_key_name: Optional[str] = None
    max_tokens: Optional[int] = Field(None, gt=0)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    context_window: Optional[int] = Field(None, gt=0)
    cost_per_1k_tokens: Optional[float] = Field(None, ge=0.0)
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    limitations: Optional[List[str]] = None
    tags: Optional[List[str]] = None

class ModelResponse(BaseModel):
    """Response schema for AI model data"""
    id: str
    name: str
    model_type: str
    provider: str
    version: str
    status: str
    endpoint_url: Optional[str]
    
    # Performance metrics
    requests_per_hour: int
    total_requests: int
    average_response_time: float
    success_rate: float
    error_count: int
    
    # Configuration
    max_tokens: Optional[int]
    temperature: Optional[float]
    context_window: Optional[int]
    cost_per_1k_tokens: Optional[float]
    
    # Metadata
    description: Optional[str]
    capabilities: List[str]
    limitations: List[str]
    tags: List[str]
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime]

class UsageLogRequest(BaseModel):
    """Request schema for logging model usage"""
    model_id: str
    request_type: str
    response_time: float = Field(..., gt=0)
    tokens_used: Optional[int] = Field(None, gt=0)
    success: bool = True
    error_message: Optional[str] = None
    user_id: Optional[str] = None

class ModelStatsResponse(BaseModel):
    """Response schema for model statistics"""
    total_models: int
    active_models: int
    total_requests: int
    requests_last_hour: int
    average_response_time: float
    average_success_rate: float

# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(prefix="/api/ai-models", tags=["AI Models"])

# Dependency to ensure database is initialized
async def ensure_db_initialized():
    """Ensure the AI models database is initialized"""
    if not ai_models_db._initialized:
        await initialize_ai_models_db()
    return ai_models_db

# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/", response_model=List[ModelResponse])
async def get_all_models(
    status: Optional[ModelStatus] = Query(None, description="Filter by model status"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    model_type: Optional[ModelType] = Query(None, description="Filter by model type"),
    db = Depends(ensure_db_initialized)
):
    """Get all AI models with optional filtering"""
    try:
        if status:
            models = await db.get_models_by_status(status)
        elif provider:
            models = await db.get_models_by_provider(provider)
        else:
            models = await db.get_all_models()
        
        # Filter by model_type if specified
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        # Convert to response format
        return [
            ModelResponse(
                id=model.id,
                name=model.name,
                model_type=model.model_type.value,
                provider=model.provider,
                version=model.version,
                status=model.status.value,
                endpoint_url=model.endpoint_url,
                requests_per_hour=model.requests_per_hour,
                total_requests=model.total_requests,
                average_response_time=model.average_response_time,
                success_rate=model.success_rate,
                error_count=model.error_count,
                max_tokens=model.max_tokens,
                temperature=model.temperature,
                context_window=model.context_window,
                cost_per_1k_tokens=model.cost_per_1k_tokens,
                description=model.description,
                capabilities=model.capabilities,
                limitations=model.limitations,
                tags=model.tags,
                created_at=model.created_at,
                updated_at=model.updated_at,
                last_used_at=model.last_used_at
            )
            for model in models
        ]
        
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")

@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    db = Depends(ensure_db_initialized)
):
    """Get a specific AI model by ID"""
    try:
        model = await db.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        
        return ModelResponse(
            id=model.id,
            name=model.name,
            model_type=model.model_type.value,
            provider=model.provider,
            version=model.version,
            status=model.status.value,
            endpoint_url=model.endpoint_url,
            requests_per_hour=model.requests_per_hour,
            total_requests=model.total_requests,
            average_response_time=model.average_response_time,
            success_rate=model.success_rate,
            error_count=model.error_count,
            max_tokens=model.max_tokens,
            temperature=model.temperature,
            context_window=model.context_window,
            cost_per_1k_tokens=model.cost_per_1k_tokens,
            description=model.description,
            capabilities=model.capabilities,
            limitations=model.limitations,
            tags=model.tags,
            created_at=model.created_at,
            updated_at=model.updated_at,
            last_used_at=model.last_used_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model: {str(e)}")

@router.post("/", response_model=Dict[str, str])
async def create_model(
    model_data: ModelCreateRequest,
    db = Depends(ensure_db_initialized)
):
    """Create a new AI model"""
    try:
        import uuid
        
        # Create AIModel object
        model = AIModel(
            id=str(uuid.uuid4()),
            name=model_data.name,
            model_type=model_data.model_type,
            provider=model_data.provider,
            version=model_data.version,
            status=ModelStatus.ACTIVE,  # Default to active
            endpoint_url=model_data.endpoint_url,
            api_key_name=model_data.api_key_name,
            max_tokens=model_data.max_tokens,
            temperature=model_data.temperature,
            context_window=model_data.context_window,
            cost_per_1k_tokens=model_data.cost_per_1k_tokens,
            description=model_data.description,
            capabilities=model_data.capabilities,
            limitations=model_data.limitations,
            tags=model_data.tags
        )
        
        model_id = await db.create_model(model)
        
        return {
            "id": model_id,
            "message": f"Model '{model_data.name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")

@router.put("/{model_id}", response_model=Dict[str, str])
async def update_model(
    model_id: str,
    updates: ModelUpdateRequest,
    db = Depends(ensure_db_initialized)
):
    """Update an existing AI model"""
    try:
        # Convert Pydantic model to dict, excluding None values
        update_dict = {k: v for k, v in updates.dict().items() if v is not None}
        
        if not update_dict:
            raise HTTPException(status_code=400, detail="No valid updates provided")
        
        success = await db.update_model(model_id, update_dict)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        
        return {
            "id": model_id,
            "message": f"Model updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")

@router.delete("/{model_id}", response_model=Dict[str, str])
async def delete_model(
    model_id: str,
    db = Depends(ensure_db_initialized)
):
    """Delete an AI model"""
    try:
        success = await db.delete_model(model_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        
        return {
            "id": model_id,
            "message": f"Model deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

# ============================================================================
# MODEL OPERATIONS ENDPOINTS
# ============================================================================

@router.post("/{model_id}/activate", response_model=Dict[str, str])
async def activate_model(
    model_id: str,
    db = Depends(ensure_db_initialized)
):
    """Activate an AI model"""
    try:
        success = await db.update_model(model_id, {"status": ModelStatus.ACTIVE})
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        
        return {
            "id": model_id,
            "message": f"Model activated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {str(e)}")

@router.post("/{model_id}/pause", response_model=Dict[str, str])
async def pause_model(
    model_id: str,
    db = Depends(ensure_db_initialized)
):
    """Pause an AI model"""
    try:
        success = await db.update_model(model_id, {"status": ModelStatus.PAUSED})
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        
        return {
            "id": model_id,
            "message": f"Model paused successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause model: {str(e)}")

# ============================================================================
# USAGE LOGGING ENDPOINTS
# ============================================================================

@router.post("/usage-log", response_model=Dict[str, str])
async def log_model_usage(
    usage_data: UsageLogRequest,
    db = Depends(ensure_db_initialized)
):
    """Log usage of an AI model"""
    try:
        import uuid
        from datetime import timezone
        
        usage_log = ModelUsageLog(
            id=str(uuid.uuid4()),
            model_id=usage_data.model_id,
            timestamp=datetime.now(timezone.utc),
            request_type=usage_data.request_type,
            response_time=usage_data.response_time,
            tokens_used=usage_data.tokens_used,
            success=usage_data.success,
            error_message=usage_data.error_message,
            user_id=usage_data.user_id
        )
        
        log_id = await db.log_usage(usage_log)
        
        return {
            "id": log_id,
            "message": "Usage logged successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to log usage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log usage: {str(e)}")

# ============================================================================
# STATISTICS ENDPOINTS
# ============================================================================

@router.get("/stats/overview", response_model=ModelStatsResponse)
async def get_model_stats(
    db = Depends(ensure_db_initialized)
):
    """Get overall AI model statistics"""
    try:
        stats = await db.get_model_stats()
        
        return ModelStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get model stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")

@router.get("/providers", response_model=List[str])
async def get_providers(
    db = Depends(ensure_db_initialized)
):
    """Get list of all model providers"""
    try:
        models = await db.get_all_models()
        providers = list(set(model.provider for model in models))
        return sorted(providers)
        
    except Exception as e:
        logger.error(f"Failed to get providers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve providers: {str(e)}")

@router.get("/types", response_model=List[str])
async def get_model_types(
    db = Depends(ensure_db_initialized)
):
    """Get list of all model types"""
    try:
        return [model_type.value for model_type in ModelType]
        
    except Exception as e:
        logger.error(f"Failed to get model types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model types: {str(e)}")

# ============================================================================
# BULK OPERATIONS
# ============================================================================

@router.post("/bulk/activate", response_model=Dict[str, Any])
async def bulk_activate_models(
    model_ids: List[str] = Body(..., description="List of model IDs to activate"),
    db = Depends(ensure_db_initialized)
):
    """Activate multiple models at once"""
    try:
        results = {"success": [], "failed": []}
        
        for model_id in model_ids:
            try:
                success = await db.update_model(model_id, {"status": ModelStatus.ACTIVE})
                if success:
                    results["success"].append(model_id)
                else:
                    results["failed"].append({"id": model_id, "error": "Model not found"})
            except Exception as e:
                results["failed"].append({"id": model_id, "error": str(e)})
        
        return {
            "message": f"Activated {len(results['success'])} models",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed bulk activation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to activate models: {str(e)}")

@router.get("/export", response_model=Dict[str, Any])
async def export_models_data(
    db = Depends(ensure_db_initialized)
):
    """Export all models data for migration to SaaS"""
    try:
        export_data = await db.export_to_dict()
        
        return {
            "message": "Models data exported successfully",
            "data": export_data
        }
        
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")

# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health", response_model=Dict[str, str])
async def health_check(
    db = Depends(ensure_db_initialized)
):
    """Health check for AI models database"""
    try:
        # Simple query to test database connectivity
        models_count = len(await db.get_all_models())
        
        return {
            "status": "healthy",
            "database": "connected",
            "models_count": str(models_count),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Database health check failed: {str(e)}")

# Export the router
ai_models_router = router