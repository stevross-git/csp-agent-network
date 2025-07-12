# Database and Schema Overview

This repository uses **PostgreSQL** as the main data store. Two closely related database schemas are employed:

1. **Application Database** – models defined with SQLAlchemy under `enhanced_csp/backend`.
2. **Network Schema** – an advanced schema for networking metrics located in `enhanced_csp/network`.

## Application Database

The application backend relies on asynchronous SQLAlchemy. Connections are managed via `backend/database/connection.py`. Table definitions live in `backend/models/database_models.py` and are automatically created at startup by the `create_tables()` function.

Key tables include:

- `users` and `local_users` – authentication records.
- `designs`, `design_nodes`, `design_connections` – Visual Designer entities.
- `execution_sessions` and `execution_metrics` – execution tracking.
- `component_types` – registry of available component implementations.
- `audit_logs`, `system_config`, `notifications`, and `licenses` – system management tables.

A separate `monitoring` schema stores generic metrics via the `MonitoringMetric` model.

## Network Schema

The networking subsystem uses a dedicated `network` schema. The full SQL definition is contained in [`enhanced_csp/network-database-schema.txt`](../enhanced_csp/network-database-schema.txt).  A setup script [`enhanced_csp/setup-network-database.sh`](../enhanced_csp/setup-network-database.sh) creates the schema and helper functions.

Important tables and views:

- **Core tables** – `nodes`, `mesh_topologies`, `mesh_links` describe the mesh topology.
- **Routing** – `routing_entries` with associated `routing_metrics` maintain BATMAN-style routes.
- **Connection pooling** – `connection_pools` and `connections` manage socket pools.
- **Optimization and compression** – `optimization_params`, `compression_stats`.
- **Batching metrics** – `batch_configs`, `batch_metrics`.
- **Telemetry** – `metrics`, `events`, and `performance_snapshots` with views such as `realtime_status`.
- **Service discovery** – `dns_cache` and `service_registry`.
- **History** – `topology_optimizations` for tracking optimization runs.

Triggers in the schema update timestamps and maintain best routes. Monthly partitions are created for the `metrics` table.

## Initialization

To create the application tables and populate default component types run:

```bash
python -m enhanced_csp.backend.main  # during startup
```

For the network schema execute:

```bash
bash enhanced_csp/setup-network-database.sh
```

Both processes expect a PostgreSQL instance configured via environment variables (e.g., `DATABASE_URL` or `NETWORK_DATABASE_URL`).

## Directory Layout

- `enhanced_csp/backend/database` – engine configuration and utilities.
- `enhanced_csp/backend/models` – ORM models for the API.
- `enhanced_csp/network/database_models.py` – ORM models mirroring the network schema.
- `enhanced_csp/network-database-schema.txt` – declarative SQL schema.

These files collectively define and manage the database layer for the project.
