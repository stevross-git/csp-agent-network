import type { NetworkNode, NetworkMetrics, NetworkEvent } from '../types'

// Mock data generators for real-time simulation
class MockRealtimeAPI {
  private nodeCounter = 0
  private eventCounter = 0
  private baseNodes: NetworkNode[] = []

  constructor() {
    this.initializeBaseNodes()
  }

  private initializeBaseNodes() {
    this.baseNodes = [
      {
        id: 'node-1',
        name: 'Central Hub',
        type: 'hub',
        status: 'active',
        ipAddress: '192.168.1.1',
        port: 8080,
        connections: 8,
        lastSeen: new Date().toISOString(),
        uptime: 99.9,
        version: '1.2.3',
        location: {
          latitude: 40.7128,
          longitude: -74.0060,
          country: 'USA',
          city: 'New York'
        },
        metrics: {
          cpu: 45 + Math.random() * 20,
          memory: 60 + Math.random() * 15,
          disk: 30 + Math.random() * 10,
          bandwidth: { in: 100 + Math.random() * 50, out: 80 + Math.random() * 40 },
          latency: 5 + Math.random() * 10,
          packetLoss: Math.random() * 2,
          temperature: 45 + Math.random() * 15
        }
      },
      {
        id: 'node-2',
        name: 'Edge Server A',
        type: 'edge',
        status: 'active',
        ipAddress: '192.168.1.2',
        port: 8080,
        connections: 4,
        lastSeen: new Date().toISOString(),
        uptime: 98.5,
        version: '1.2.3',
        metrics: {
          cpu: 30 + Math.random() * 25,
          memory: 45 + Math.random() * 20,
          disk: 25 + Math.random() * 15,
          bandwidth: { in: 60 + Math.random() * 30, out: 50 + Math.random() * 25 },
          latency: 8 + Math.random() * 15,
          packetLoss: Math.random() * 3,
          temperature: 40 + Math.random() * 20
        }
      },
      {
        id: 'node-3',
        name: 'Edge Server B',
        type: 'edge',
        status: 'warning',
        ipAddress: '192.168.1.3',
        port: 8080,
        connections: 3,
        lastSeen: new Date().toISOString(),
        uptime: 85.2,
        version: '1.2.2',
        metrics: {
          cpu: 75 + Math.random() * 15,
          memory: 80 + Math.random() * 10,
          disk: 90 + Math.random() * 5,
          bandwidth: { in: 40 + Math.random() * 20, out: 35 + Math.random() * 15 },
          latency: 25 + Math.random() * 20,
          packetLoss: 1 + Math.random() * 4,
          temperature: 65 + Math.random() * 15
        }
      },
      {
        id: 'node-4',
        name: 'Peer Node Alpha',
        type: 'peer',
        status: 'active',
        ipAddress: '192.168.1.4',
        port: 8080,
        connections: 2,
        lastSeen: new Date().toISOString(),
        uptime: 97.8,
        version: '1.2.3',
        metrics: {
          cpu: 20 + Math.random() * 30,
          memory: 35 + Math.random() * 25,
          disk: 20 + Math.random() * 20,
          bandwidth: { in: 30 + Math.random() * 20, out: 25 + Math.random() * 15 },
          latency: 12 + Math.random() * 18,
          packetLoss: Math.random() * 2,
          temperature: 35 + Math.random() * 20
        }
      },
      {
        id: 'node-5',
        name: 'Storage Node',
        type: 'storage',
        status: 'inactive',
        ipAddress: '192.168.1.5',
        port: 8080,
        connections: 0,
        lastSeen: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
        uptime: 0,
        version: '1.1.9',
        metrics: {
          cpu: 0,
          memory: 0,
          disk: 95,
          bandwidth: { in: 0, out: 0 },
          latency: 0,
          packetLoss: 100,
          temperature: 25
        }
      }
    ]
  }

  // Simulate real-time node updates
  async getNodes(): Promise<{ data: NetworkNode[] }> {
    // Add some randomization to simulate real-time changes
    const updatedNodes = this.baseNodes.map(node => ({
      ...node,
      lastSeen: node.status === 'active' ? new Date().toISOString() : node.lastSeen,
      connections: node.status === 'active' 
        ? Math.max(0, node.connections + Math.floor((Math.random() - 0.5) * 2))
        : node.connections,
      metrics: node.metrics ? {
        ...node.metrics,
        cpu: Math.max(0, Math.min(100, node.metrics.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.max(0, Math.min(100, node.metrics.memory + (Math.random() - 0.5) * 8)),
        bandwidth: {
          in: Math.max(0, node.metrics.bandwidth.in + (Math.random() - 0.5) * 20),
          out: Math.max(0, node.metrics.bandwidth.out + (Math.random() - 0.5) * 15)
        },
        latency: Math.max(0, node.metrics.latency + (Math.random() - 0.5) * 5),
        packetLoss: Math.max(0, Math.min(100, node.metrics.packetLoss + (Math.random() - 0.5) * 1))
      } : undefined
    }))

    // Occasionally change node status
    if (Math.random() < 0.1) {
      const randomNode = updatedNodes[Math.floor(Math.random() * updatedNodes.length)]
      if (randomNode.status === 'active' && Math.random() < 0.3) {
        randomNode.status = 'warning'
      } else if (randomNode.status === 'warning' && Math.random() < 0.7) {
        randomNode.status = 'active'
      }
    }

    this.baseNodes = updatedNodes
    return { data: updatedNodes }
  }

  async getMetrics(timeRange: string = '24h'): Promise<{ data: NetworkMetrics }> {
    const activeNodes = this.baseNodes.filter(n => n.status === 'active').length
    const totalNodes = this.baseNodes.length
    const totalConnections = this.baseNodes.reduce((sum, node) => sum + node.connections, 0)
    
    const avgCpu = this.baseNodes
      .filter(n => n.metrics)
      .reduce((sum, node) => sum + (node.metrics?.cpu || 0), 0) / activeNodes

    const avgMemory = this.baseNodes
      .filter(n => n.metrics)
      .reduce((sum, node) => sum + (node.metrics?.memory || 0), 0) / activeNodes

    const totalBandwidthIn = this.baseNodes
      .filter(n => n.metrics)
      .reduce((sum, node) => sum + (node.metrics?.bandwidth.in || 0), 0)

    const totalBandwidthOut = this.baseNodes
      .filter(n => n.metrics)
      .reduce((sum, node) => sum + (node.metrics?.bandwidth.out || 0), 0)

    const avgLatency = this.baseNodes
      .filter(n => n.metrics && n.status === 'active')
      .reduce((sum, node) => sum + (node.metrics?.latency || 0), 0) / activeNodes

    return {
      data: {
        timestamp: new Date().toISOString(),
        totalNodes,
        activeNodes,
        totalConnections,
        totalThroughput: (totalBandwidthIn + totalBandwidthOut) / 1000, // Convert to GB/s
        averageLatency: avgLatency,
        networkUptime: (activeNodes / totalNodes) * 100,
        dataTransferred: {
          in: totalBandwidthIn,
          out: totalBandwidthOut
        },
        performance: {
          cpu: avgCpu,
          memory: avgMemory,
          disk: 45 + Math.random() * 20
        }
      }
    }
  }

  async getEvents(limit: number = 50): Promise<{ data: NetworkEvent[] }> {
    const eventTypes = ['connection', 'performance', 'security', 'system', 'error', 'maintenance']
    const severities = ['info', 'warning', 'error', 'success']
    
    const events: NetworkEvent[] = []
    
    // Generate some real-time events
    for (let i = 0; i < Math.min(limit, 10); i++) {
      const type = eventTypes[Math.floor(Math.random() * eventTypes.length)]
      const severity = severities[Math.floor(Math.random() * severities.length)]
      const node = this.baseNodes[Math.floor(Math.random() * this.baseNodes.length)]
      
      events.push({
        id: `event-${this.eventCounter++}`,
        type: type as any,
        severity: severity as any,
        message: this.generateEventMessage(type, node.name),
        nodeId: node.id,
        timestamp: new Date(Date.now() - Math.random() * 3600 * 1000).toISOString(),
        resolved: Math.random() > 0.7,
        metadata: {
          nodeType: node.type,
          connections: node.connections
        }
      })
    }
    
    return { data: events.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()) }
  }

  private generateEventMessage(type: string, nodeName: string): string {
    const messages = {
      connection: [
        `New peer connected to ${nodeName}`,
        `Connection established with ${nodeName}`,
        `Peer disconnected from ${nodeName}`,
        `Connection timeout on ${nodeName}`
      ],
      performance: [
        `Performance improved on ${nodeName}`,
        `High CPU usage detected on ${nodeName}`,
        `Memory usage spike on ${nodeName}`,
        `Bandwidth utilization increased on ${nodeName}`
      ],
      security: [
        `Security scan completed on ${nodeName}`,
        `Suspicious activity detected on ${nodeName}`,
        `Authentication attempt on ${nodeName}`,
        `Firewall rule updated for ${nodeName}`
      ],
      system: [
        `System update completed on ${nodeName}`,
        `Backup process started on ${nodeName}`,
        `Configuration change applied to ${nodeName}`,
        `Service restart on ${nodeName}`
      ],
      error: [
        `Error occurred on ${nodeName}`,
        `Connection failed to ${nodeName}`,
        `Service unavailable on ${nodeName}`,
        `Timeout error on ${nodeName}`
      ],
      maintenance: [
        `Maintenance window started for ${nodeName}`,
        `Scheduled update on ${nodeName}`,
        `Maintenance completed on ${nodeName}`,
        `System check performed on ${nodeName}`
      ]
    }
    
    const typeMessages = messages[type as keyof typeof messages] || messages.system
    return typeMessages[Math.floor(Math.random() * typeMessages.length)]
  }

  async getSystemHealth(): Promise<{ data: any }> {
    return {
      data: {
        status: 'healthy',
        uptime: 99.5 + Math.random() * 0.5,
        lastCheck: new Date().toISOString(),
        services: {
          api: 'healthy',
          database: 'healthy',
          websocket: 'healthy',
          monitoring: 'healthy'
        }
      }
    }
  }
}

export const mockRealtimeAPI = new MockRealtimeAPI()