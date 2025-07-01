# Network API Documentation

## Overview

The Enhanced CSP Backend now includes P2P networking capabilities for distributed execution and real-time collaboration across network nodes.

## Configuration

Set these environment variables to enable network features:

```bash
NETWORK_ENABLED=true
NETWORK_PORT=30300
BOOTSTRAP_NODES=seed1.web4ai:30300,seed2.web4ai:30300
ENABLE_QUANTUM=false
ENABLE_BLOCKCHAIN=false
```

## API Endpoints

### GET /api/network/status
Get current network status and node information.

**Response:**
```json
{
  "node_id": "QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "status": "online",
  "peers": 5,
  "designs": 10,
  "executions": 3,
  "capabilities": {
    "execution": true,
    "storage": true,
    "relay": true,
    "collaboration": true
  },
  "network_info": {
    "nat_type": "full_cone",
    "external_address": "1.2.3.4:30300",
    "topology_type": "dynamic_partial"
  }
}
```

### GET /api/network/nodes
List all known network nodes.

### GET /api/network/designs/{design_id}/nodes
Find nodes that have a specific design.

### POST /api/network/execute
Execute a design on the distributed network.

### POST /api/network/broadcast
Broadcast a message to all network nodes (admin only).

## WebSocket Integration

Design updates are automatically synchronized across all nodes in the network.

### Network Events

- `network:peer_joined` - A new peer joined the network
- `network:peer_left` - A peer left the network
- `network:design_sync` - Design synchronized from network
- `network:execution_update` - Execution update from remote node
