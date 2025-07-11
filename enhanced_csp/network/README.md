# Enhanced CSP Network

This module implements the peer‑to‑peer networking layer used by the Enhanced CSP platform. It provides mesh routing, peer discovery, connection pooling, and metrics collection. The network stack can operate independently or alongside the FastAPI backend.

## Requirements
- Python 3.8+
- Optional: `aiohttp` for the metrics endpoint and dashboard

## Installation
Install the Python dependencies from the root of the repository:

```bash
pip install -r requirements-lock.txt
```

If you plan to use the network database schema, ensure PostgreSQL is available and set the `NETWORK_DATABASE_URL` environment variable. The schema can be initialized with `enhanced_csp/setup-network-database.sh`.

## Running a Network Node
Start a local node using the provided entry script:

```bash
python -m enhanced_csp.network.main --genesis
```

Additional options are available via `--help`. The node supports peer discovery, NAT traversal, and can expose a simple HTTP dashboard when `aiohttp` is installed.

## Tests
Execute the network tests only:

```bash
pytest enhanced_csp/network/tests -vv
```

## Directory Overview
```
network/
├── core/          # Node primitives and type definitions
├── dns/           # Lightweight DNS overlay for service discovery
├── p2p/           # DHT, NAT traversal and transport modules
├── mesh/          # Mesh topology and routing algorithms
├── routing/       # Adaptive and multipath routing engines
├── examples/      # Example applications using the network stack
├── dashboard/     # Optional real‑time monitoring dashboard
├── database_models.py  # SQLAlchemy models for the network schema
└── tests/         # Unit and integration tests
```

Further details on the database schemas can be found in [../docs/database_schema_overview.md](../../docs/database_schema_overview.md).
