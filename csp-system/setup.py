#!/usr/bin/env python3
"""
CSP System Setup
================
Advanced AI-to-AI Communication Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
long_description = (Path(__file__).parent / "README.md").read_text()

# Read requirements
requirements = [
    "asyncio>=3.4.3",
    "aiohttp>=3.8.0",
    "numpy>=1.24.0",
    "networkx>=3.1.0",
    "pydantic>=1.10.0",
    "pyyaml>=6.0",
    "fastapi>=0.95.0",
    "click>=8.1.0",
    "rich>=13.3.0",
    "websockets>=11.0",
    "redis>=4.5.0",
    "prometheus-client>=0.16.0",
    "kubernetes>=26.0.0",
    "docker>=6.0.0",
    "cryptography>=40.0.0",
    "psutil>=5.9.0",
    "uvicorn>=0.22.0",
    "jinja2>=3.1.0",
    "matplotlib>=3.7.0",
    "plotly>=5.14.0",
    "flask>=2.3.0",
    "flask-cors>=4.0.0",
    "flask-socketio>=5.3.0",
]

dev_requirements = [
    "pytest>=7.3.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.3.0",
    "flake8>=6.0.0",
    "mypy>=1.3.0",
    "isort>=5.12.0",
    "bandit>=1.7.0",
    "pre-commit>=3.3.0",
]

setup(
    name="csp-system",
    version="1.0.0",
    description="Revolutionary AI-to-AI Communication Platform using CSP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CSP System Team",
    author_email="team@csp-system.org",
    url="https://github.com/csp-system/csp-system",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "csp_system": [
            "web_ui/templates/*.html",
            "web_ui/static/css/*.css",
            "web_ui/static/js/*.js",
            "config/templates/*.yaml",
            "deployment/k8s/*.yaml",
        ],
    },
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "full": requirements + dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "csp=cli.csp_cli:main",
            "csp-server=runtime.server:main",
            "csp-dashboard=web_ui.dashboard.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    keywords=[
        "ai", "communication", "csp", "agents", "distributed-systems", 
        "process-algebra", "quantum-computing", "formal-verification"
    ],
    project_urls={
        "Documentation": "https://docs.csp-system.org",
        "Bug Tracker": "https://github.com/csp-system/csp-system/issues",
        "Source Code": "https://github.com/csp-system/csp-system",
    },
)