# backend/database/ai_models_db.py
"""
AI Models Local Database System
==============================
SQLite-based local database for AI models with SaaS migration capability
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio
import aiosqlite
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

class ModelStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    LOADING = "loading"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class ModelType(str, Enum):
    LLM = "llm"
    IMAGE_GENERATION = "image_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    MULTIMODAL = "multimodal"
    CODE_GENERATION = "code_generation"
    EMBEDDING = "embedding"
    FINE_TUNED = "fine_tuned"

@dataclass
class AIModel:
    """AI Model data structure"""
    id: str
    name: str
    model_type: ModelType
    provider: str
    version: str
    status: ModelStatus
    endpoint_url: Optional[str] = None
    api_key_name: Optional[str] = None
    
    # Performance metrics
    requests_per_hour: int = 0
    total_requests: int = 0
    average_response_time: float = 0.0
    success_rate: float = 100.0
    error_count: int = 0
    
    # Configuration
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    context_window: Optional[int] = None
    cost_per_1k_tokens: Optional[float] = None
    
    # Metadata
    description: Optional[str] = None
    capabilities: List[str] = None
    limitations: List[str] = None
    tags: List[str] = None
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    last_used_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.limitations is None:
            self.limitations = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)

@dataclass
class ModelUsageLog:
    """Model usage logging"""
    id: str
    model_id: str
    timestamp: datetime
    request_type: str
    response_time: float
    tokens_used: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    cost: Optional[float] = None

@dataclass
class ModelMetrics:
    """Aggregated model metrics"""
    model_id: str
    date: datetime
    total_requests: int
    successful_requests: int
    average_response_time: float
    total_tokens_used: int
    total_cost: float
    peak_requests_per_hour: int

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class AIModelsDatabase:
    """Local SQLite database for AI models with async support"""
    
    def __init__(self, db_path: str = "data/ai_models.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._initialized = False
    
    async def initialize(self):
        """Initialize database with tables"""
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS ai_models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    endpoint_url TEXT,
                    api_key_name TEXT,
                    
                    -- Performance metrics
                    requests_per_hour INTEGER DEFAULT 0,
                    total_requests INTEGER DEFAULT 0,
                    average_response_time REAL DEFAULT 0.0,
                    success_rate REAL DEFAULT 100.0,
                    error_count INTEGER DEFAULT 0,
                    
                    -- Configuration
                    max_tokens INTEGER,
                    temperature REAL,
                    context_window INTEGER,
                    cost_per_1k_tokens REAL,
                    
                    -- Metadata
                    description TEXT,
                    capabilities TEXT, -- JSON array
                    limitations TEXT,  -- JSON array
                    tags TEXT,         -- JSON array
                    
                    -- Timestamps
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_usage_logs (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    request_type TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    tokens_used INTEGER,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    user_id TEXT,
                    cost REAL,
                    FOREIGN KEY (model_id) REFERENCES ai_models (id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    total_requests INTEGER NOT NULL,
                    successful_requests INTEGER NOT NULL,
                    average_response_time REAL NOT NULL,
                    total_tokens_used INTEGER NOT NULL,
                    total_cost REAL NOT NULL,
                    peak_requests_per_hour INTEGER NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES ai_models (id),
                    UNIQUE (model_id, date)
                )
            """)
            
            # Create indexes for performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_models_status ON ai_models (status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_models_provider ON ai_models (provider)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_models_type ON ai_models (model_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_usage_model_timestamp ON model_usage_logs (model_id, timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_metrics_model_date ON model_metrics (model_id, date)")
            
            await db.commit()
        
        await self._create_default_models()
        self._initialized = True
        logger.info(f"✅ AI Models database initialized at {self.db_path}")
    
    async def _create_default_models(self):
        """Create default AI models if none exist"""
        existing_models = await self.get_all_models()
        if existing_models:
            return
        
        default_models = [
            AIModel(
                id=str(uuid.uuid4()),
                name="GPT-4 Turbo",
                model_type=ModelType.LLM,
                provider="OpenAI",
                version="2024-01-25",
                status=ModelStatus.ACTIVE,
                endpoint_url="https://api.openai.com/v1/chat/completions",
                api_key_name="OPENAI_API_KEY",
                max_tokens=4096,
                temperature=0.7,
                context_window=128000,
                cost_per_1k_tokens=0.01,
                description="Latest GPT-4 Turbo model with enhanced capabilities",
                capabilities=["text_generation", "code_generation", "analysis", "reasoning"],
                tags=["openai", "gpt4", "llm", "conversational"]
            ),
            AIModel(
                id=str(uuid.uuid4()),
                name="Claude-3 Sonnet",
                model_type=ModelType.LLM,
                provider="Anthropic",
                version="3.0",
                status=ModelStatus.ACTIVE,
                endpoint_url="https://api.anthropic.com/v1/messages",
                api_key_name="ANTHROPIC_API_KEY",
                max_tokens=4096,
                temperature=0.7,
                context_window=200000,
                cost_per_1k_tokens=0.003,
                description="Claude-3 Sonnet for balanced performance and capability",
                capabilities=["text_generation", "analysis", "reasoning", "code_review"],
                tags=["anthropic", "claude", "llm", "safety"]
            ),
            AIModel(
                id=str(uuid.uuid4()),
                name="DALL-E 3",
                model_type=ModelType.IMAGE_GENERATION,
                provider="OpenAI",
                version="3.0",
                status=ModelStatus.ACTIVE,
                endpoint_url="https://api.openai.com/v1/images/generations",
                api_key_name="OPENAI_API_KEY",
                cost_per_1k_tokens=0.04,
                description="Advanced image generation model",
                capabilities=["image_generation", "creative_art", "photorealistic"],
                tags=["openai", "dalle", "image_generation", "creative"]
            ),
            AIModel(
                id=str(uuid.uuid4()),
                name="Whisper",
                model_type=ModelType.SPEECH_TO_TEXT,
                provider="OpenAI",
                version="v2",
                status=ModelStatus.PAUSED,
                endpoint_url="https://api.openai.com/v1/audio/transcriptions",
                api_key_name="OPENAI_API_KEY",
                cost_per_1k_tokens=0.006,
                description="Speech recognition and transcription",
                capabilities=["speech_to_text", "transcription", "multilingual"],
                tags=["openai", "whisper", "speech", "transcription"]
            ),
            AIModel(
                id=str(uuid.uuid4()),
                name="CodeLlama",
                model_type=ModelType.CODE_GENERATION,
                provider="Meta",
                version="34B",
                status=ModelStatus.LOADING,
                max_tokens=2048,
                temperature=0.1,
                context_window=16384,
                cost_per_1k_tokens=0.0015,
                description="Specialized code generation model",
                capabilities=["code_generation", "code_completion", "debugging"],
                tags=["meta", "code", "programming", "llama"]
            ),
            AIModel(
                id=str(uuid.uuid4()),
                name="Gemini Pro",
                model_type=ModelType.MULTIMODAL,
                provider="Google",
                version="1.0",
                status=ModelStatus.ACTIVE,
                endpoint_url="https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
                api_key_name="GOOGLE_API_KEY",
                max_tokens=8192,
                temperature=0.9,
                context_window=32768,
                cost_per_1k_tokens=0.00025,
                description="Multimodal AI for text and image understanding",
                capabilities=["text_generation", "image_understanding", "multimodal"],
                tags=["google", "gemini", "multimodal", "vision"]
            )
        ]
        
        for model in default_models:
            await self.create_model(model)
        
        logger.info(f"✅ Created {len(default_models)} default AI models")
    
    # ========================================================================
    # MODEL CRUD OPERATIONS
    # ========================================================================
    
    async def create_model(self, model: AIModel) -> str:
        """Create a new AI model"""
        if not self._initialized:
            await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO ai_models (
                    id, name, model_type, provider, version, status,
                    endpoint_url, api_key_name, requests_per_hour, total_requests,
                    average_response_time, success_rate, error_count,
                    max_tokens, temperature, context_window, cost_per_1k_tokens,
                    description, capabilities, limitations, tags,
                    created_at, updated_at, last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.id, model.name, model.model_type.value, model.provider, model.version, model.status.value,
                model.endpoint_url, model.api_key_name, model.requests_per_hour, model.total_requests,
                model.average_response_time, model.success_rate, model.error_count,
                model.max_tokens, model.temperature, model.context_window, model.cost_per_1k_tokens,
                model.description, json.dumps(model.capabilities), json.dumps(model.limitations), json.dumps(model.tags),
                model.created_at.isoformat(), model.updated_at.isoformat(),
                model.last_used_at.isoformat() if model.last_used_at else None
            ))
            await db.commit()
        
        logger.info(f"✅ Created AI model: {model.name} ({model.id})")
        return model.id
    
    async def get_model(self, model_id: str) -> Optional[AIModel]:
        """Get a single AI model by ID"""
        if not self._initialized:
            await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM ai_models WHERE id = ?", (model_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_model(row)
                return None
    
    async def get_all_models(self) -> List[AIModel]:
        """Get all AI models"""
        if not self._initialized:
            await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM ai_models ORDER BY created_at DESC") as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_model(row) for row in rows]
    
    async def get_models_by_status(self, status: ModelStatus) -> List[AIModel]:
        """Get models by status"""
        if not self._initialized:
            await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM ai_models WHERE status = ? ORDER BY name", (status.value,)) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_model(row) for row in rows]
    
    async def get_models_by_provider(self, provider: str) -> List[AIModel]:
        """Get models by provider"""
        if not self._initialized:
            await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM ai_models WHERE provider = ? ORDER BY name", (provider,)) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_model(row) for row in rows]
    
    async def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update an AI model"""
        if not self._initialized:
            await self.initialize()
        
        # Build dynamic UPDATE query
        set_clauses = []
        values = []
        
        for key, value in updates.items():
            if key in ['capabilities', 'limitations', 'tags'] and isinstance(value, list):
                set_clauses.append(f"{key} = ?")
                values.append(json.dumps(value))
            elif key in ['status'] and isinstance(value, ModelStatus):
                set_clauses.append(f"{key} = ?")
                values.append(value.value)
            elif key in ['model_type'] and isinstance(value, ModelType):
                set_clauses.append(f"{key} = ?")
                values.append(value.value)
            else:
                set_clauses.append(f"{key} = ?")
                values.append(value)
        
        # Always update the updated_at timestamp
        set_clauses.append("updated_at = ?")
        values.append(datetime.now(timezone.utc).isoformat())
        values.append(model_id)
        
        query = f"UPDATE ai_models SET {', '.join(set_clauses)} WHERE id = ?"
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, values)
            await db.commit()
            success = cursor.rowcount > 0
        
        if success:
            logger.info(f"✅ Updated AI model: {model_id}")
        else:
            logger.warning(f"⚠️ Model not found for update: {model_id}")
        
        return success
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete an AI model"""
        if not self._initialized:
            await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Delete related usage logs first
            await db.execute("DELETE FROM model_usage_logs WHERE model_id = ?", (model_id,))
            await db.execute("DELETE FROM model_metrics WHERE model_id = ?", (model_id,))
            
            # Delete the model
            cursor = await db.execute("DELETE FROM ai_models WHERE id = ?", (model_id,))
            await db.commit()
            success = cursor.rowcount > 0
        
        if success:
            logger.info(f"✅ Deleted AI model: {model_id}")
        else:
            logger.warning(f"⚠️ Model not found for deletion: {model_id}")
        
        return success
    
    # ========================================================================
    # USAGE LOGGING
    # ========================================================================
    
    async def log_usage(self, usage_log: ModelUsageLog) -> str:
        """Log model usage"""
        if not self._initialized:
            await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO model_usage_logs (
                    id, model_id, timestamp, request_type, response_time,
                    tokens_used, success, error_message, user_id, cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                usage_log.id, usage_log.model_id, usage_log.timestamp.isoformat(),
                usage_log.request_type, usage_log.response_time, usage_log.tokens_used,
                usage_log.success, usage_log.error_message, usage_log.user_id, usage_log.cost
            ))
            await db.commit()
        
        # Update model statistics
        await self._update_model_stats_from_usage(usage_log)
        
        return usage_log.id
    
    async def _update_model_stats_from_usage(self, usage_log: ModelUsageLog):
        """Update model statistics based on usage log"""
        model = await self.get_model(usage_log.model_id)
        if not model:
            return
        
        # Calculate new averages
        total_requests = model.total_requests + 1
        new_avg_response_time = (
            (model.average_response_time * model.total_requests + usage_log.response_time) / total_requests
        )
        
        error_count = model.error_count + (0 if usage_log.success else 1)
        success_rate = ((total_requests - error_count) / total_requests) * 100
        
        updates = {
            'total_requests': total_requests,
            'average_response_time': new_avg_response_time,
            'success_rate': success_rate,
            'error_count': error_count,
            'last_used_at': usage_log.timestamp.isoformat()
        }
        
        await self.update_model(usage_log.model_id, updates)
    
    # ========================================================================
    # METRICS AND ANALYTICS
    # ========================================================================
    
    async def get_model_stats(self) -> Dict[str, Any]:
        """Get overall model statistics"""
        if not self._initialized:
            await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Get basic counts
            async with db.execute("SELECT COUNT(*) FROM ai_models") as cursor:
                total_models = (await cursor.fetchone())[0]
            
            async with db.execute("SELECT COUNT(*) FROM ai_models WHERE status = 'active'") as cursor:
                active_models = (await cursor.fetchone())[0]
            
            # Get total requests and average response time
            async with db.execute("""
                SELECT 
                    SUM(total_requests) as total_requests,
                    AVG(average_response_time) as avg_response_time,
                    AVG(success_rate) as avg_success_rate
                FROM ai_models WHERE total_requests > 0
            """) as cursor:
                stats = await cursor.fetchone()
                total_requests = stats[0] or 0
                avg_response_time = stats[1] or 0
                avg_success_rate = stats[2] or 100
            
            # Get requests in last hour
            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            async with db.execute("""
                SELECT COUNT(*) FROM model_usage_logs 
                WHERE timestamp > ?
            """, (one_hour_ago.isoformat(),)) as cursor:
                requests_last_hour = (await cursor.fetchone())[0]
        
        return {
            "total_models": total_models,
            "active_models": active_models,
            "total_requests": total_requests,
            "requests_last_hour": requests_last_hour,
            "average_response_time": round(avg_response_time, 2),
            "average_success_rate": round(avg_success_rate, 1)
        }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _row_to_model(self, row) -> AIModel:
        """Convert database row to AIModel object"""
        return AIModel(
            id=row['id'],
            name=row['name'],
            model_type=ModelType(row['model_type']),
            provider=row['provider'],
            version=row['version'],
            status=ModelStatus(row['status']),
            endpoint_url=row['endpoint_url'],
            api_key_name=row['api_key_name'],
            requests_per_hour=row['requests_per_hour'],
            total_requests=row['total_requests'],
            average_response_time=row['average_response_time'],
            success_rate=row['success_rate'],
            error_count=row['error_count'],
            max_tokens=row['max_tokens'],
            temperature=row['temperature'],
            context_window=row['context_window'],
            cost_per_1k_tokens=row['cost_per_1k_tokens'],
            description=row['description'],
            capabilities=json.loads(row['capabilities']) if row['capabilities'] else [],
            limitations=json.loads(row['limitations']) if row['limitations'] else [],
            tags=json.loads(row['tags']) if row['tags'] else [],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            last_used_at=datetime.fromisoformat(row['last_used_at']) if row['last_used_at'] else None
        )
    
    async def export_to_dict(self) -> Dict[str, Any]:
        """Export all data for SaaS migration"""
        models = await self.get_all_models()
        return {
            "models": [asdict(model) for model in models],
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0"
        }

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Global database instance
ai_models_db = AIModelsDatabase()

# Async initialization function
async def initialize_ai_models_db():
    """Initialize the global AI models database"""
    await ai_models_db.initialize()
    return ai_models_db

# Sync initialization for backwards compatibility
def get_ai_models_db() -> AIModelsDatabase:
    """Get the global AI models database instance"""
    return ai_models_db