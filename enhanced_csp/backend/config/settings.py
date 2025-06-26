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

from pydantic import BaseSettings, Field, validator
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
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="csp_visual_designer", env="DB_NAME")
    username: str = Field(default="csp_user", env="DB_USER")
    password: str = Field(default="csp_password", env="DB_PASSWORD")
    
    # Connection pool settings
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    
    # Additional settings
    echo_sql: bool = Field(default=False, env="DB_ECHO_SQL")
    ssl_mode: str = Field(default="prefer", env="DB_SSL_MODE")
    
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
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    database: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Connection settings
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    timeout: int = Field(default=5, env="REDIS_TIMEOUT")
    retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        auth_part = f":{self.password}@" if self.password else ""
        return f"redis://{auth_part}{self.host}:{self.port}/{self.database}"

class AIConfig(BaseSettings):
    """AI service configuration"""
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(default=None, env="OPENAI_ORG_ID")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Default model settings
    default_model: str = Field(default="gpt-3.5-turbo", env="AI_DEFAULT_MODEL")
    default_temperature: float = Field(default=0.7, env="AI_DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=1000, env="AI_DEFAULT_MAX_TOKENS")
    
    # Rate limiting
    max_requests_per_minute: int = Field(default=60, env="AI_MAX_REQUESTS_PER_MINUTE")
    max_tokens_per_minute: int = Field(default=10000, env="AI_MAX_TOKENS_PER_MINUTE")
    
    # Cost control
    max_daily_cost: float = Field(default=100.0, env="AI_MAX_DAILY_COST")
    cost_alert_threshold: float = Field(default=80.0, env="AI_COST_ALERT_THRESHOLD")

class SecurityConfig(BaseSettings):
    """Security configuration"""
    
    # JWT settings
    secret_key: str = Field(env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=1440, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")  # 24 hours
    refresh_token_expire_days: int = Field(default=30, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Password settings
    password_min_length: int = Field(default=8, env="PASSWORD_MIN_LENGTH")
    password_require_uppercase: bool = Field(default=True, env="PASSWORD_REQUIRE_UPPERCASE")
    password_require_lowercase: bool = Field(default=True, env="PASSWORD_REQUIRE_LOWERCASE")
    password_require_numbers: bool = Field(default=True, env="PASSWORD_REQUIRE_NUMBERS")
    password_require_special: bool = Field(default=False, env="PASSWORD_REQUIRE_SPECIAL")
    
    # Rate limiting
    max_login_attempts: int = Field(default=5, env="MAX_LOGIN_ATTEMPTS")
    lockout_duration_minutes: int = Field(default=15, env="LOCKOUT_DURATION_MINUTES")
    
    # CORS settings
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    allowed_methods: List[str] = Field(default=["*"], env="ALLOWED_METHODS")
    allowed_headers: List[str] = Field(default=["*"], env="ALLOWED_HEADERS")
    
    @validator('allowed_origins', 'allowed_methods', 'allowed_headers', pre=True)
    def parse_list_from_string(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v

class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration"""
    
    # Metrics collection
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    collection_interval: int = Field(default=15, env="METRICS_COLLECTION_INTERVAL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    enable_json_logging: bool = Field(default=False, env="ENABLE_JSON_LOGGING")
    
    # Tracing
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    
    # Alerts
    enable_alerts: bool = Field(default=True, env="ENABLE_ALERTS")
    alert_webhook_url: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    
    # Health checks
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

class PerformanceConfig(BaseSettings):
    """Performance and optimization configuration"""
    
    # WebSocket settings
    max_websocket_connections: int = Field(default=1000, env="MAX_WEBSOCKET_CONNECTIONS")
    websocket_timeout: int = Field(default=300, env="WEBSOCKET_TIMEOUT")
    
    # Execution engine
    max_parallel_executions: int = Field(default=50, env="MAX_PARALLEL_EXECUTIONS")
    execution_timeout: int = Field(default=3600, env="EXECUTION_TIMEOUT")
    max_execution_memory_mb: int = Field(default=2048, env="MAX_EXECUTION_MEMORY_MB")
    
    # Caching
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    max_cache_size_mb: int = Field(default=512, env="MAX_CACHE_SIZE_MB")
    
    # Background tasks
    max_background_tasks: int = Field(default=10, env="MAX_BACKGROUND_TASKS")
    task_queue_size: int = Field(default=1000, env="TASK_QUEUE_SIZE")

class APIConfig(BaseSettings):
    """API configuration"""
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    
    # Request settings
    max_request_size: int = Field(default=16 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 16MB
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    rate_limit_requests_per_minute: int = Field(default=100, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    
    # Documentation
    enable_docs: bool = Field(default=True, env="ENABLE_DOCS")
    docs_url: str = Field(default="/docs", env="DOCS_URL")
    redoc_url: str = Field(default="/redoc", env="REDOC_URL")

# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

class Settings(BaseSettings):
    """Main application settings"""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Application metadata
    app_name: str = Field(default="CSP Visual Designer API", env="APP_NAME")
    version: str = Field(default="2.0.0", env="APP_VERSION")
    description: str = Field(default="Advanced AI-Powered CSP Process Designer Backend", env="APP_DESCRIPTION")
    
    # Feature flags
    enable_ai: bool = Field(default=True, env="ENABLE_AI")
    enable_websockets: bool = Field(default=True, env="ENABLE_WEBSOCKETS")
    enable_authentication: bool = Field(default=True, env="ENABLE_AUTHENTICATION")
    enable_file_upload: bool = Field(default=True, env="ENABLE_FILE_UPLOAD")
    
    # Configuration sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"  # For nested configs like DATABASE__HOST
        case_sensitive = False
    
    @validator('environment', pre=True)
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
        config_dict = self._settings.dict()
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
        config_info["config"] = settings.dict()
    
    return config_info
    