# enhanced_csp/config/__init__.py
"""
Configuration management for Enhanced CSP
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

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
        # Example env vars:
        # CSP_NETWORK_OPTIMIZATION_ENABLED=true
        # CSP_COMPRESSION_ALGORITHM=zstd
        
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

# Global config loader instance
config_loader = ConfigLoader()
load_config = config_loader.load_config
load_defaults = config_loader.load_defaults