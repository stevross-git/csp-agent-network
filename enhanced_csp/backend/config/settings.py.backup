# File: backend/config/settings.py
"""
Configuration Management System
==============================
Centralized configuration management with environment-based overrides
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from collections import defaultdict

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
import yaml

logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT ENUM
# ============================================================================

class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

class DatabaseConfig(BaseSettings):
    """Database configuration"""
    
    # PostgreSQL settings
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="csp_visual_designer")
    username: str = Field(default="csp_user")
    password: str = Field(default="csp_password")
    
    # Connection pool settings
    pool_size: int = Field(default=20)
    max_overflow: int = Field(default=30)
    pool_timeout: int = Field(default=30)
    pool_recycle: int = Field(default=3600)
    
    # Additional settings
    echo_sql: bool = Field(default=False)
    ssl_mode: str = Field(default="prefer")
    
    model_config = {
        "env_prefix": "DB_",
        "case_sensitive": False
    }
    
    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def sync_url(self) -> str:
        """Get synchronous database URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class RedisConfig(BaseSettings):
    """Redis configuration"""
    
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    database: int = Field(default=0)
    password: Optional[str] = Field(default=None)
    
    # Connection settings
    max_connections: int = Field(default=20)
    timeout: int = Field(default=5)
    retry_on_timeout: bool = Field(default=True)
    
    model_config = {
        "env_prefix": "REDIS_",
        "case_sensitive": False
    }
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        auth_part = f":{self.password}@" if self.password else ""
        return f"redis://{auth_part}{self.host}:{self.port}/{self.database}"

class AIConfig(BaseSettings):
    """AI service configuration"""
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None)
    openai_org_id: Optional[str] = Field(default=None)
    openai_base_url: Optional[str] = Field(default=None)
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = Field(default=None)
    
    # Default model settings
    default_model: str = Field(default="gpt-3.5-turbo")
    default_temperature: float = Field(default=0.7)
    default_max_tokens: int = Field(default=1000)
    
    # Rate limiting
    max_requests_per_minute: int = Field(default=60)
    max_tokens_per_minute: int = Field(default=10000)
    
    # Cost control
    max_daily_cost: float = Field(default=100.0)
    cost_alert_threshold: float = Field(default=80.0)
    
    model_config = {
        "env_prefix": "AI_",
        "case_sensitive": False
    }

class SecurityConfig(BaseSettings):
    """Security configuration"""
    
    # JWT settings
    secret_key: str = Field(default="change-this-secret-key")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=1440)  # 24 hours
    refresh_token_expire_days: int = Field(default=30)
    
    # Password settings
    password_min_length: int = Field(default=8)
    password_require_uppercase: bool = Field(default=True)
    password_require_lowercase: bool = Field(default=True)
    password_require_numbers: bool = Field(default=True)
    password_require_special: bool = Field(default=False)
    
    # Rate limiting
    max_login_attempts: int = Field(default=5)
    lockout_duration_minutes: int = Field(default=15)
    
    # CORS settings
    allowed_origins: List[str] = Field(default=["*"])
    allowed_methods: List[str] = Field(default=["*"])
    allowed_headers: List[str] = Field(default=["*"])
    
    model_config = {
        "case_sensitive": False
    }
    
    @field_validator('allowed_origins', 'allowed_methods', 'allowed_headers', mode='before')
    @classmethod
    def parse_list_from_string(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v

class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration"""
    
    # Metrics collection
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    collection_interval: int = Field(default=15)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    enable_json_logging: bool = Field(default=False)
    
    # Tracing
    enable_tracing: bool = Field(default=False)
    jaeger_endpoint: Optional[str] = Field(default=None)
    
    # Alerts
    enable_alerts: bool = Field(default=True)
    alert_webhook_url: Optional[str] = Field(default=None)
    slack_webhook_url: Optional[str] = Field(default=None)
    
    # Health checks
    health_check_interval: int = Field(default=30)
    
    model_config = {
        "case_sensitive": False
    }
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

class PerformanceConfig(BaseSettings):
    """Performance and optimization configuration"""
    
    # WebSocket settings
    max_websocket_connections: int = Field(default=1000)
    websocket_timeout: int = Field(default=300)
    
    # Execution engine
    max_parallel_executions: int = Field(default=50)
    execution_timeout: int = Field(default=3600)
    max_execution_memory_mb: int = Field(default=2048)
    
    # Caching
    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600)
    max_cache_size_mb: int = Field(default=512)
    
    # Background tasks
    max_background_tasks: int = Field(default=10)
    task_queue_size: int = Field(default=1000)
    
    model_config = {
        "case_sensitive": False
    }

class APIConfig(BaseSettings):
    """API configuration"""
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    
    # Request settings
    max_request_size: int = Field(default=16 * 1024 * 1024)  # 16MB
    request_timeout: int = Field(default=30)
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True)
    rate_limit_requests_per_minute: int = Field(default=100)
    
    # Documentation
    enable_docs: bool = Field(default=True)
    docs_url: str = Field(default="/docs")
    redoc_url: str = Field(default="/redoc")
    
    model_config = {
        "env_prefix": "API_",
        "case_sensitive": False
    }

# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

class Settings(BaseSettings):
    """Main application settings"""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    
    # Application metadata
    app_name: str = Field(default="CSP Visual Designer API")
    version: str = Field(default="2.0.0")
    description: str = Field(default="Advanced AI-Powered CSP Process Designer Backend")
    
    # Feature flags
    enable_ai: bool = Field(default=True)
    enable_websockets: bool = Field(default=True)
    enable_authentication: bool = Field(default=True)
    enable_file_upload: bool = Field(default=True)
    # Monitoring feature flag
    MONITORING_ENABLED: bool = Field(default=False)
    
    # Configuration sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    @field_validator('environment', mode='before')
    @classmethod
    def validate_environment(cls, v):
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                return Environment.DEVELOPMENT
        return v
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

class ConfigLoader:
    """Configuration loader with support for multiple sources"""
    
    def __init__(self):
        self._settings: Optional[Settings] = None
        self._config_sources = []
    
    def load_from_env(self) -> Settings:
        """Load configuration from environment variables"""
        if self._settings is None:
            self._settings = Settings()
            self._config_sources.append("environment")
        return self._settings
    
    def load_from_file(self, config_path: Union[str, Path]) -> Settings:
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return self.load_from_env()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                    return self.load_from_env()
            
            # Merge with environment variables (env vars take precedence)
            env_settings = self.load_from_env()
            
            # Update with file-based config
            for key, value in config_data.items():
                if hasattr(env_settings, key):
                    # Handle nested configurations
                    if isinstance(value, dict) and hasattr(getattr(env_settings, key), '__dict__'):
                        nested_config = getattr(env_settings, key)
                        for nested_key, nested_value in value.items():
                            if hasattr(nested_config, nested_key):
                                setattr(nested_config, nested_key, nested_value)
                    else:
                        setattr(env_settings, key, value)
            
            self._config_sources.append(f"file:{config_path}")
            logger.info(f"Configuration loaded from: {config_path}")
            
            return env_settings
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            return self.load_from_env()
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about loaded configuration"""
        if self._settings is None:
            return {"error": "No configuration loaded"}
        
        # Mask sensitive information
        config_dict = self._settings.model_dump()
        self._mask_sensitive_data(config_dict)
        
        return {
            "environment": self._settings.environment.value,
            "debug": self._settings.debug,
            "sources": self._config_sources,
            "feature_flags": {
                "ai": self._settings.enable_ai,
                "websockets": self._settings.enable_websockets,
                "authentication": self._settings.enable_authentication,
                "file_upload": self._settings.enable_file_upload,
            },
            "config": config_dict
        }
    
    def _mask_sensitive_data(self, config_dict: Dict[str, Any]):
        """Mask sensitive configuration data"""
        sensitive_keys = [
            'password', 'secret_key', 'api_key', 'token',
            'private_key', 'webhook_url'
        ]
        
        def mask_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                        obj[key] = "***MASKED***" if value else None
                    elif isinstance(value, (dict, list)):
                        mask_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    mask_recursive(item)
        
        mask_recursive(config_dict)

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_configuration(settings: Settings) -> Dict[str, List[str]]:
    """Validate configuration and return any issues"""
    issues = defaultdict(list)
    
    # Database validation
    if not settings.database.host:
        issues['database'].append("Database host is required")
    
    if not settings.database.database:
        issues['database'].append("Database name is required")
    
    # Security validation
    if settings.enable_authentication and not settings.security.secret_key:
        issues['security'].append("Secret key is required when authentication is enabled")
    
    if settings.security.secret_key and len(settings.security.secret_key) < 32:
        issues['security'].append("Secret key should be at least 32 characters long")
    
    # AI validation
    if settings.enable_ai:
        if not settings.ai.openai_api_key and not settings.ai.anthropic_api_key:
            issues['ai'].append("At least one AI provider API key is required when AI is enabled")
    
    # Production-specific validations
    if settings.is_production():
        if settings.debug:
            issues['production'].append("Debug mode should be disabled in production")
        
        if "*" in settings.security.allowed_origins:
            issues['production'].append("Wildcard CORS origins should not be used in production")
        
        if settings.monitoring.log_level == "DEBUG":
            issues['production'].append("Debug logging should not be enabled in production")
    
    return dict(issues)

# ============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================================

# Global configuration loader
config_loader = ConfigLoader()

# Load configuration based on environment
def load_configuration() -> Settings:
    """Load configuration from appropriate sources"""
    config_file_env = os.getenv("CONFIG_FILE")
    
    if config_file_env:
        # Load from specified config file
        settings = config_loader.load_from_file(config_file_env)
    else:
        # Try to load from default locations
        default_config_paths = [
            "config.yml",
            "config.yaml",
            "config.json",
            f"config/{os.getenv('ENVIRONMENT', 'development')}.yml",
            f"config/{os.getenv('ENVIRONMENT', 'development')}.yaml"
        ]
        
        settings = None
        for config_path in default_config_paths:
            if Path(config_path).exists():
                settings = config_loader.load_from_file(config_path)
                break
        
        # Fallback to environment variables only
        if settings is None:
            settings = config_loader.load_from_env()
    
    # Validate configuration
    validation_issues = validate_configuration(settings)
    if validation_issues:
        logger.warning("Configuration validation issues found:")
        for section, issues in validation_issues.items():
            for issue in issues:
                logger.warning(f"  {section}: {issue}")
    
    logger.info(f"Configuration loaded successfully (environment: {settings.environment.value})")
    return settings

# Global settings instance
settings = load_configuration()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings

def reload_configuration():
    """Reload configuration from sources"""
    global settings
    settings = load_configuration()
    logger.info("Configuration reloaded")

def get_database_url() -> str:
    """Get database URL"""
    return settings.database.url

def get_redis_url() -> str:
    """Get Redis URL"""
    return settings.redis.url

def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled"""
    feature_flags = {
        "ai": settings.enable_ai,
        "websockets": settings.enable_websockets,
        "authentication": settings.enable_authentication,
        "file_upload": settings.enable_file_upload,
        "metrics": settings.monitoring.enable_metrics,
        "alerts": settings.monitoring.enable_alerts,
        "caching": settings.performance.enable_caching,
        "rate_limiting": settings.api.enable_rate_limiting,
        "docs": settings.api.enable_docs
    }
    
    return feature_flags.get(feature, False)

def get_log_config() -> Dict[str, Any]:
    """Get logging configuration"""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": settings.monitoring.log_format
            },
            "json": {
                "format": "%(message)s",
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
            } if settings.monitoring.enable_json_logging else {
                "format": settings.monitoring.log_format
            }
        },
        "handlers": {
            "default": {
                "formatter": "json" if settings.monitoring.enable_json_logging else "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout"
            }
        },
        "root": {
            "level": settings.monitoring.log_level,
            "handlers": ["default"]
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["default"],
                "propagate": False
            },
            "sqlalchemy.engine": {
                "level": "INFO" if settings.database.echo_sql else "WARNING",
                "handlers": ["default"],
                "propagate": False
            }
        }
    }

# Environment-specific configuration helpers
def configure_for_testing():
    """Configure application for testing"""
    global settings
    settings.environment = Environment.TESTING
    settings.debug = True
    settings.database.database = "csp_test_db"
    settings.redis.database = 1
    settings.monitoring.enable_metrics = False
    settings.monitoring.enable_alerts = False
    
    logger.info("Configuration updated for testing environment")

def configure_for_production():
    """Configure application for production"""
    global settings
    settings.environment = Environment.PRODUCTION
    settings.debug = False
    settings.monitoring.log_level = "INFO"
    settings.security.allowed_origins = [
        origin for origin in settings.security.allowed_origins 
        if origin != "*"
    ]
    
    logger.info("Configuration updated for production environment")

# Configuration export for debugging
def export_configuration(mask_sensitive: bool = True) -> Dict[str, Any]:
    """Export current configuration"""
    config_info = config_loader.get_config_info()
    
    if not mask_sensitive:
        # Return unmasked configuration (for debugging only)
        config_info["config"] = settings.model_dump()
    
    return config_info