# csp-agent-network

[![Tests](https://github.com/stevross-git/enhanced_csp/actions/workflows/build-test.yml/badge.svg)](https://github.com/stevross-git/enhanced_csp/actions/workflows/build-test.yml)
[![Docker Build](https://github.com/stevross-git/enhanced_csp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/stevross-git/enhanced_csp/actions/workflows/docker-publish.yml)
[![Prod Deploy](https://github.com/stevross-git/enhanced_csp/actions/workflows/deploy-production.yml/badge.svg)](https://github.com/stevross-git/enhanced_csp/actions/workflows/deploy-production.yml)

## Migration Map
Old Symbol | New Module
--- | ---
AdvancedAICommChannel | `enhanced_csp.ai_comm.AdvancedAICommChannel`
BaseAgent | `enhanced_csp.agents.BaseAgent`
DataCleanerAgent | `enhanced_csp.agents.DataCleanerAgent`
PlannerAgent | `enhanced_csp.agents.PlannerAgent`
CSPLogStore | `enhanced_csp.api.CSPLogStore`
ChromaVectorStore | `enhanced_csp.memory.ChromaVectorStore`
create_csp_message | `enhanced_csp.protocols.create_csp_message`

## Quick Start
```python
from enhanced_csp.ai_comm import AdvancedAICommChannel, AdvancedCommPattern
from enhanced_csp.agents import BaseAgent

channel = AdvancedAICommChannel('demo', AdvancedCommPattern.NEURAL_MESH)
agent = BaseAgent('agent1')
channel.participants[agent.agent_id] = agent
```

### Quick Demo

```
python examples/quick_demo.py --config examples/config/peoplesai.yaml
```

## Running Tests

Install requirements and run pytest:

```bash
pip install -r requirements-lock.txt
pytest -vv
```

## Deploying with Docker Compose

```bash
docker compose -f deploy/docker/docker-compose.yaml up -d
```

## Helm Installation

```bash
helm install enhanced-csp deploy/helm/enhanced-csp
```

## Secrets and Variables

Add the following secrets in **Settings → Secrets** for CI/CD workflows:

| Name | Purpose |
| ---- | ------- |
| `AWS_ACCESS_KEY_ID` | AWS access key for automation |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_REGION` | Default AWS region |
| `TF_BACKEND_BUCKET` | S3 bucket for Terraform state |
| `TF_DYNAMODB_TABLE` | DynamoDB table for state locking |
| `GITHUB_TOKEN` | Automatically provided GitHub token |

## Monitoring Database

Performance metrics collected by the backend are persisted in the `monitoring`
schema. The database initialization routine automatically creates this schema
when invoking `create_tables()` during startup.

For a deeper look at the backend architecture see
[`docs/backend_overview.md`](docs/backend_overview.md).
The mapping of frontend services to backend routes is documented in
[`docs/frontend_backend_api_mapping.md`](docs/frontend_backend_api_mapping.md).

Detailed setup instructions for the FastAPI backend live in
[`enhanced_csp/backend/README.md`](enhanced_csp/backend/README.md).
Frontend configuration and development instructions are in [`enhanced_csp/frontend/README.md`](enhanced_csp/frontend/README.md).

Documentation on the database schemas is available in [docs/database_schema_overview.md](docs/database_schema_overview.md).
The peer-to-peer networking layer is documented in
[`enhanced_csp/network/README.md`](enhanced_csp/network/README.md).
