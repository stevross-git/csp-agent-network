# Enhanced CSP Backend Overview

This document provides a high level description of the backend contained in the
`enhanced_csp` package. The backend powers the API and monitoring services for
the CSP Visual Designer.

## Architecture

The backend is built with **FastAPI** and uses **SQLAlchemy** with PostgreSQL
for persistence. Redis is employed for caching and temporary storage. Key
modules include:

- **api** – REST API routers exposing over 70 endpoints
- **auth** – Azure AD and local authentication with JWT rotation and RBAC
- **database** – engine management, connection helpers and model definitions
- **models** – SQLAlchemy ORM models for designs, executions and components
- **execution** – asynchronous execution engine for designs
- **monitoring** – performance monitoring and metrics collection system
- **components** – registry of pluggable component types
- **services** – supporting utilities such as JWT helpers and cache monitoring
- **realtime** – WebSocket and real-time communication helpers

Entry point `main.py` wires these modules together and configures middleware,
CORS and background tasks.

## Data Persistence

PostgreSQL stores application data through the models in
`backend/models/database_models.py`. The `create_tables()` routine
automatically creates the necessary tables as well as the `monitoring` schema
which stores system metrics. Redis is used by `CacheManager` for caching and by
the monitoring subsystem for temporary metric storage.

### Monitoring Metrics

The `MonitoringMetric` model represents generic metrics stored under the
`monitoring.metrics` table. `PerformanceMonitor` periodically collects CPU,
memory, disk usage and request statistics, pushing them to Redis and inserting
rows into this table.

## Developer Workflow

- Install dependencies from `requirements-lock.txt`
- Run tests with `pytest -vv`
- Start the application using `uvicorn backend.main:app`

For a comprehensive API analysis refer to `backend/backend_api_review.md`.
