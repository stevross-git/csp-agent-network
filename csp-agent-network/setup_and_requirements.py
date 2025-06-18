# setup.py
"""
CSP System - Advanced AI Communication Platform
Setup configuration for package installation and distribution
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version
version = "1.0.0"
if os.path.exists("VERSION"):
    with open("VERSION", "r") as fh:
        version = fh.read().strip()

setup(
    name="csp-system",
    version=version,
    author="CSP Development Team",
    author_email="team@csp-system.org",
    description="Revolutionary CSP System for Advanced AI-to-AI Communication",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csp-system/csp-system",
    project_urls={
        "Bug Tracker": "https://github.com/csp-system/csp-system/issues",
        "Documentation": "https://docs.csp-system.org",
        "Source Code": "https://github.com/csp-system/csp-system",
    },
    packages=find_packages(),
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
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.16.0",
            "grafana-api>=1.0.3",
            "jaeger-client>=4.8.0",
        ],
        "cloud": [
            "boto3>=1.26.0",
            "google-cloud>=0.34.0",
            "azure-mgmt>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "csp=cli.csp_implementation_guide:main",
            "csp-server=runtime.server:main",
            "csp-dashboard=web_ui.dashboard.app:main",
            "csp-designer=dev_tools.visual_designer.designer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.html", "*.css", "*.js", "*.md"],
    },
    zip_safe=False,
)

# requirements.txt
"""
Core Dependencies for CSP System
===============================
"""

# Core async and networking
asyncio-mqtt>=0.13.0
aiohttp>=3.8.0
aiofiles>=23.0.0
uvloop>=0.17.0

# Scientific computing and AI
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
plotly>=5.14.0
networkx>=3.1.0

# AI and ML libraries
openai>=0.27.0
anthropic>=0.2.0
transformers>=4.28.0
torch>=2.0.0
tensorflow>=2.12.0

# Data processing
pydantic>=1.10.0
pyyaml>=6.0
toml>=0.10.2
jsonschema>=4.17.0

# Web and API
fastapi>=0.95.0
uvicorn>=0.21.0
websockets>=11.0.0
dash>=2.10.0
flask>=2.3.0
jinja2>=3.1.0

# Database and storage
sqlalchemy>=2.0.0
alembic>=1.10.0
redis>=4.5.0
pymongo>=4.3.0

# Monitoring and observability
prometheus-client>=0.16.0
opencensus>=0.11.0
structlog>=23.1.0

# Deployment and orchestration
docker>=6.1.0
kubernetes>=26.1.0
helm>=0.1.0

# Development tools
click>=8.1.0
rich>=13.3.0
typer>=0.9.0

# Security and encryption
cryptography>=40.0.0
pyjwt>=2.6.0
bcrypt>=4.0.0

# Testing (for development)
pytest>=7.3.0
pytest-asyncio>=0.21.0
pytest-mock>=3.10.0

# Optional cloud providers
boto3>=1.26.0
google-cloud-core>=2.3.0
azure-core>=1.26.0

# requirements-dev.txt
"""
Development Dependencies for CSP System
======================================
"""

# Include all production requirements
-r requirements.txt

# Testing framework
pytest>=7.3.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-benchmark>=4.0.0
pytest-xdist>=3.2.0

# Code quality
black>=23.3.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.3.0
pylint>=2.17.0
bandit>=1.7.0

# Pre-commit hooks
pre-commit>=3.3.0

# Documentation
sphinx>=6.2.0
sphinx-rtd-theme>=1.2.0
myst-parser>=1.0.0
sphinx-autodoc-typehints>=1.23.0

# Development tools
ipython>=8.13.0
jupyter>=1.0.0
notebook>=6.5.0

# Debugging and profiling
pdb++>=0.10.3
memory-profiler>=0.60.0
line-profiler>=4.0.0

# API documentation
swagger-ui-bundle>=0.1.2
redoc>=2.0.0

# Database tools
alembic>=1.11.0
sqlalchemy-utils>=0.41.0

# Monitoring development
grafana-api>=1.0.3
prometheus-api-client>=0.5.3

# Performance testing
locust>=2.15.0
artillery>=1.0.0

# pyproject.toml
"""
Modern Python project configuration
"""

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "csp-system"
version = "1.0.0"
description = "Revolutionary CSP System for Advanced AI-to-AI Communication"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "CSP Development Team", email = "team@csp-system.org"}
]
maintainers = [
    {name = "CSP Development Team", email = "team@csp-system.org"}
]
keywords = [
    "csp", "ai", "communication", "distributed-systems", 
    "process-algebra", "quantum-computing", "formal-verification"
]
classifiers = [
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
]
requires-python = ">=3.8"
dependencies = [
    "asyncio-mqtt>=0.13.0",
    "aiohttp>=3.8.0",
    "numpy>=1.24.0",
    "networkx>=3.1.0",
    "pydantic>=1.10.0",
    "pyyaml>=6.0",
    "fastapi>=0.95.0",
    "click>=8.1.0",
    "rich>=13.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.3.0",
    "black>=23.3.0",
    "flake8>=6.0.0",
    "mypy>=1.3.0",
]
docs = [
    "sphinx>=6.2.0",
    "sphinx-rtd-theme>=1.2.0",
]
cloud = [
    "boto3>=1.26.0",
    "google-cloud>=0.34.0",
    "azure-mgmt>=4.0.0",
]

[project.urls]
Homepage = "https://github.com/csp-system/csp-system"
Documentation = "https://docs.csp-system.org"
Repository = "https://github.com/csp-system/csp-system"
"Bug Tracker" = "https://github.com/csp-system/csp-system/issues"

[project.scripts]
csp = "cli.csp_implementation_guide:main"
csp-server = "runtime.server:main"
csp-dashboard = "web_ui.dashboard.app:main"

[tool.setuptools.packages.find]
exclude = ["tests*", "docs*", "examples*"]

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["csp_system"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

[tool.coverage.run]
source = ["csp_system"]
omit = ["*/tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]

# Makefile
"""
Build automation for CSP System
"""

.PHONY: help install install-dev test test-cov lint format clean build deploy docs

help:
	@echo "CSP System - Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run linting (flake8, mypy)"
	@echo "  format        Format code (black, isort)"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build package"
	@echo "  deploy        Deploy to production"
	@echo "  docs          Build documentation"
	@echo "  showcase      Run system showcase"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=csp_system --cov-report=html --cov-report=term

lint:
	flake8 csp_system/ tests/
	mypy csp_system/
	bandit -r csp_system/

format:
	black csp_system/ tests/
	isort csp_system/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

deploy: build
	python cli/csp_implementation_guide.py deploy config/templates/production.yaml

docs:
	cd docs && make html

showcase:
	python cli/csp_implementation_guide.py showcase

# Start development environment
dev-start:
	docker-compose -f docker-compose.dev.yml up -d

# Stop development environment
dev-stop:
	docker-compose -f docker-compose.dev.yml down

# Run all quality checks
check-all: lint test-cov
	@echo "All quality checks passed!"

# VERSION
1.0.0

# .gitignore
"""
Git ignore rules for CSP System
"""

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# CSP System specific
data/
logs/
config/local/
config/secrets/
*.db
*.sqlite

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Docker
.dockerignore

# Kubernetes
*.kubeconfig

# Terraform
*.tfstate
*.tfstate.*
.terraform/

# Secrets
secrets.yaml
.secrets/
*.key
*.pem
*.crt
