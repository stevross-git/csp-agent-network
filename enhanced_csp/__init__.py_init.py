# enhanced_csp/__init__.py
"""Enhanced CSP System - Advanced Computing Systems Platform."""

__version__ = "1.0.0"
__author__ = "Enhanced CSP Team"
__description__ = "Advanced Computing Systems Platform with AI, Quantum, and Distributed Computing"

# Make this a proper Python package
from pathlib import Path

# Package information
__package_root__ = Path(__file__).parent
__package_name__ = "enhanced_csp"

# Version info
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

def get_version() -> str:
    """Get the current version string."""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"

# Export version
__all__ = ["__version__", "get_version", "VERSION_INFO"]