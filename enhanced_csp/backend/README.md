# Enhanced CSP Backend

This directory contains the FastAPI backend that powers the CSP Visual Designer and monitoring services. It exposes over 70 API endpoints and includes a background execution engine and metrics collection tools.

## Requirements
- Python 3.8+
- PostgreSQL for persistent storage
- Redis for caching and monitoring queues

## Installation
Install Python packages using the locked requirements file:
```bash
pip install -r requirements-lock.txt
```

A running PostgreSQL instance is expected. Configure the connection using the `DATABASE_URL` environment variable. The optional network schema can be configured with `NETWORK_DATABASE_URL`.

## Running the Server
Start the API server in development mode with:
```bash
uvicorn enhanced_csp.backend.main:app --reload
```
This will automatically create the required tables by invoking `create_tables()` at startup.

## Tests
Execute the backend test suite with:
```bash
pytest -vv
```

## Additional Documentation
- [Backend architecture overview](../../docs/backend_overview.md)
- [Database and schema details](../../docs/database_schema_overview.md)
- [API review report](backend_api_review.md)
