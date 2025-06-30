# csp-agent-network

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

## STUN/TURN Configuration
Before running a cluster you may specify external STUN/TURN servers so that
peers behind NATs can reach each other:

```bash
export P2P_STUN_SERVERS="stun:stun.l.google.com:19302"
export P2P_TURN_SERVERS="turn:turn.example.com:3478"
```

These environment variables are consumed by the Docker compose and Helm charts.
