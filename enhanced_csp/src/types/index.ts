// Network Node Types
export interface NetworkNode {
  id: string
  name: string
  status: 'active' | 'degraded' | 'failed' | 'maintenance'
  region: string
  asg?: string // Auto Scaling Group
  cpu: number // percentage
  memory: number // percentage
  uptime: number // seconds
  lastUpdated: string // ISO timestamp
  ipAddress: string
  type: 'master' | 'worker' | 'edge' | 'gateway'
  metrics: NodeMetrics
}

export interface NodeMetrics {
  throughput: number // Mbps
  latency: number // ms
  packetLoss: number // percentage
  connections: number
  errorRate: number // errors per second
}

// Topology Types
export interface TopologyData {
  nodes: TopologyNode[]
  links: TopologyLink[]
  lastUpdated: string
}

export interface TopologyNode {
  id: string
  label: string
  type: 'master' | 'worker' | 'edge' | 'gateway'
  status: 'active' | 'degraded' | 'failed' | 'maintenance'
  x?: number
  y?: number
}

export interface TopologyLink {
  id: string
  source: string
  target: string
  bandwidth: number // Mbps
  latency: number // ms
  utilization: number // percentage
  status: 'healthy' | 'congested' | 'failed'
}

// Event Types
export interface NetworkEvent {
  id: string
  timestamp: string
  type: 'info' | 'warning' | 'error' | 'critical'
  category: 'node' | 'link' | 'security' | 'performance' | 'system'
  nodeId?: string
  title: string
  message: string
  metadata?: Record<string, any>
}

// Metrics Types
export interface MetricPoint {
  timestamp: number
  value: number
}

export interface MetricSeries {
  name: string
  data: MetricPoint[]
  unit: string
}

export interface PrometheusMetric {
  metric: {
    __name__: string
    [key: string]: string
  }
  values: Array<[number, string]>
}

// Dashboard KPI Types
export interface DashboardKPI {
  activeNodes: number
  failedNodes: number
  avgLatency: number
  packetLoss: number
  throughput: number
  totalNodes: number
  degradedNodes: number
}

// Auth Types
export interface User {
  id: string
  email: string
  name: string
  role: 'admin' | 'operator' | 'viewer'
}

export interface AuthState {
  user: User | null
  token: string | null
  refreshToken: string | null
  isAuthenticated: boolean
}

export interface LoginCredentials {
  email: string
  password: string
}

export interface AuthResponse {
  user: User
  token: string
  refreshToken: string
  expiresIn: number
}

// Settings Types
export interface AppSettings {
  theme: 'light' | 'dark' | 'system'
  refreshInterval: number // seconds
  metricUnits: {
    throughput: 'Mbps' | 'Gbps'
    storage: 'GB' | 'TB'
  }
  notifications: {
    sound: boolean
    desktop: boolean
    criticalOnly: boolean
  }
  language: string
  timeZone: string
}

// Filter Types
export interface NodeFilters {
  status?: NetworkNode['status'][]
  region?: string[]
  asg?: string[]
  type?: NetworkNode['type'][]
  search?: string
}

export interface EventFilters {
  type?: NetworkEvent['type'][]
  category?: NetworkEvent['category'][]
  nodeId?: string
  startDate?: Date
  endDate?: Date
}

// Chart Types
export interface ChartConfig {
  timeRange: '5m' | '1h' | '24h' | 'custom'
  customRange?: {
    start: Date
    end: Date
  }
  metrics: string[]
  aggregation: 'avg' | 'max' | 'min' | 'sum'
  groupBy?: 'node' | 'region' | 'type'
}

// WebSocket Message Types
export interface WSMessage {
  type: 'metric' | 'event' | 'node_update' | 'topology_update'
  payload: any
  timestamp: string
}

// API Response Types
export interface ApiResponse<T> {
  data: T
  status: 'success' | 'error'
  message?: string
  timestamp: string
}

export interface PaginatedResponse<T> {
  data: T[]
  pagination: {
    page: number
    pageSize: number
    total: number
    totalPages: number
  }
}

// Error Types
export interface ApiError {
  code: string
  message: string
  details?: Record<string, any>
  timestamp: string
}