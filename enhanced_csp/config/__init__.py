# enhanced_csp/config/__init__.py
"""
Configuration management for Enhanced CSP
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class NetworkOptimizationConfig:
    """Network optimization configuration"""
    enabled: bool = True
    compression: Dict[str, Any] = field(default_factory=lambda: {
        'default_algorithm': 'lz4',
        'min_size_bytes': 256,
        'max_decompress_mb': 100
    })
    batching: Dict[str, Any] = field(default_factory=lambda: {
        'max_size': 100,
        'max_bytes': 1048576,
        'max_wait_ms': 50,
        'queue_size': 10000
    })
    connection_pool: Dict[str, Any] = field(default_factory=lambda: {
        'min': 10,
        'max': 100,
        'keepalive_timeout': 300,
        'http2': True
    })
    adaptive: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'interval_seconds': 10
    })

@dataclass
class CSPSettings:
    """Main CSP configuration settings"""
    # Core settings
    project_name: str = "Enhanced CSP"
    version: str = "1.0.0"
    debug: bool = False
    
    # Network settings
    host: str = "0.0.0.0"
    port: int = 8080
    
    # Agent settings
    agent_default_timeout: int = 300
    agent_max_retries: int = 3
    
    # Channel settings
    channel_default_pattern: str = "neural_mesh"
    channel_max_message_size: int = 16 * 1024 * 1024  # 16MB
    
    # Network optimization
    network_optimization: NetworkOptimizationConfig = field(default_factory=NetworkOptimizationConfig)
    
    # Storage settings
    data_dir: Path = Path("./data")
    log_dir: Path = Path("./logs")
    
    # Security settings
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_delta: int = 3600
    
    # External services
    redis_url: Optional[str] = None
    database_url: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CSPSettings':
        """Create settings from dictionary"""
        # Handle nested network optimization config
        if 'network_optimization' in data and isinstance(data['network_optimization'], dict):
            data['network_optimization'] = NetworkOptimizationConfig(**data['network_optimization'])
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> 'CSPSettings':
        """Create settings from environment variables"""
        settings = cls()
        
        # Override with environment variables
        env_mappings = {
            'CSP_DEBUG': ('debug', lambda x: x.lower() == 'true'),
            'CSP_HOST': ('host', str),
            'CSP_PORT': ('port', int),
            'CSP_JWT_SECRET_KEY': ('jwt_secret_key', str),
            'CSP_REDIS_URL': ('redis_url', str),
            'CSP_DATABASE_URL': ('database_url', str),
            'CSP_NETWORK_OPT_ENABLED': ('network_optimization.enabled', lambda x: x.lower() == 'true'),
            'CSP_COMPRESSION_ALGORITHM': ('network_optimization.compression.default_algorithm', str),
        }
        
        for env_key, (attr_path, converter) in env_mappings.items():
            if env_key in os.environ:
                value = converter(os.environ[env_key])
                
                # Handle nested attributes
                if '.' in attr_path:
                    parts = attr_path.split('.')
                    obj = settings
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)
                else:
                    setattr(settings, attr_path, value)
        
        return settings

class ConfigLoader:
    """Load and merge configuration from multiple sources"""
    
    def __init__(self):
        self._config_cache = {}
        
    def load_defaults(self) -> Dict[str, Any]:
        """Load default network optimization configuration"""
        try:
            # Try using importlib.resources (Python 3.9+)
            import importlib.resources as resources
            config_text = resources.files(__package__).joinpath(
                "network_optimization_defaults.yaml"
            ).read_text()
            return yaml.safe_load(config_text)
        except (ImportError, AttributeError):
            # Fallback for older Python
            config_path = Path(__file__).parent / "network_optimization_defaults.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning("Default configuration not found")
                return {}
    
    def load_config(self, 
                   config_file: Optional[str] = None,
                   env_prefix: str = "CSP_") -> Dict[str, Any]:
        """Load configuration with precedence: env > file > defaults"""
        
        # Start with defaults
        config = self.load_defaults()
        
        # Merge file config if provided
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_file}")
                    
                # Deep merge
                config = self._deep_merge(config, file_config)
        
        # Override with environment variables
        config = self._apply_env_overrides(config, env_prefix)
        
        return config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _apply_env_overrides(self, config: Dict, prefix: str) -> Dict:
        """Apply environment variable overrides"""
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert CSP_NETWORK_OPTIMIZATION_ENABLED to network.optimization.enabled
                config_path = key[len(prefix):].lower().split('_')
                
                # Navigate to the right place in config
                current = config
                for part in config_path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value (with type conversion)
                final_key = config_path[-1]
                if value.lower() in ('true', 'false'):
                    current[final_key] = value.lower() == 'true'
                elif value.isdigit():
                    current[final_key] = int(value)
                else:
                    current[final_key] = value
                    
        return config

# Create global instances
config_loader = ConfigLoader()
load_config = config_loader.load_config
load_defaults = config_loader.load_defaults

# Create settings instance
settings = CSPSettings.from_env()

# Export key items
__all__ = [
    'CSPSettings',
    'NetworkOptimizationConfig', 
    'settings',
    'config_loader',
    'load_config',
    'load_defaults'
]