// Icon types
export type { IconType, IconProps, LucideProps } from './icon'

// User and Authentication Types
export interface User {
  id: string
  username: string
  email?: string
  role: 'admin' | 'user' | 'viewer'
  createdAt: string
  lastLogin?: string
}

export interface AuthResponse {
  user: User
  token: string
  expiresAt: string
}

// Network Node Types
export interface NetworkNode {
  id: string
  name: string
  type: 'hub' | 'edge' | 'peer' | 'storage'
  status: 'active' | 'inactive' | 'warning' | 'error'
  ipAddress: string
  port: number
  connections: number
  lastSeen: string
  uptime: number
  version: string
  location?: {
    latitude: number
    longitude: number
    country: string
    city: string
  }
  hardware?: {
    cpu: string
    memory: number
    storage: number
  }
  metrics?: NodeMetrics
}

export interface NodeMetrics {
  cpu: number
  memory: number
  disk: number
  bandwidth: {
    in: number
    out: number
  }
  latency: number
  packetLoss: number
  temperature?: number
}

// Network Metrics Types
export interface NetworkMetrics {
  timestamp: string
  totalNodes: number
  activeNodes: number
  totalConnections: number
  totalThroughput: number
  averageLatency: number
  networkUptime: number
  dataTransferred: {
    in: number
    out: number
  }
  performance: {
    cpu: number
    memory: number
    disk: number
  }
}

export interface MetricsDataPoint {
  timestamp: string
  throughput: number
  latency: number
  cpu: number
  memory: number
  connections: number
}

// Network Event Types
export interface NetworkEvent {
  id: string
  type: 'connection' | 'performance' | 'security' | 'system' | 'error' | 'maintenance'
  severity: 'info' | 'warning' | 'error' | 'success'
  message: string
  details?: string
  nodeId?: string
  timestamp: string
  resolved?: boolean
  resolvedAt?: string
  metadata?: Record<string, any>
}

// Network Topology Types
export interface TopologyNode {
  id: string
  label: string
  type: string
  status: string
  position: {
    x: number
    y: number
  }
  data: NetworkNode
}

export interface TopologyEdge {
  id: string
  source: string
  target: string
  type: 'default' | 'smoothstep' | 'step' | 'straight'
  animated?: boolean
  style?: {
    stroke: string
    strokeWidth: number
  }
  data?: {
    bandwidth: number
    latency: number
    status: string
  }
}

export interface NetworkTopology {
  nodes: TopologyNode[]
  edges: TopologyEdge[]
  layout: 'hierarchical' | 'force' | 'circular' | 'grid'
  lastUpdated: string
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean
  data: T
  message?: string
  timestamp: string
}

export interface PaginatedResponse<T> {
  data: T[]
  pagination: {
    page: number
    limit: number
    total: number
    pages: number
  }
}

export interface ErrorResponse {
  success: false
  error: {
    code: string
    message: string
    details?: any
  }
  timestamp: string
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: string
  payload: any
  timestamp: string
  id?: string
}

export interface SubscriptionMessage extends WebSocketMessage {
  type: 'subscription'
  payload: {
    event: string
    action: 'subscribe' | 'unsubscribe'
  }
}

// Chart and Visualization Types
export interface ChartDataPoint {
  name: string
  value: number
  timestamp?: string
  [key: string]: any
}

export interface ChartConfig {
  title: string
  type: 'line' | 'bar' | 'area' | 'pie' | 'scatter'
  xAxis: string
  yAxis: string
  color?: string
  height?: number
}

// Component Props Types
export interface BaseComponentProps {
  className?: string
  children?: React.ReactNode
}

export interface LoadingProps extends BaseComponentProps {
  size?: 'sm' | 'md' | 'lg'
  text?: string
}

export interface ErrorProps extends BaseComponentProps {
  error: Error | string
  retry?: () => void
}

// Form Types
export interface FormField {
  name: string
  label: string
  type: 'text' | 'email' | 'password' | 'number' | 'select' | 'checkbox' | 'textarea'
  required?: boolean
  placeholder?: string
  options?: { label: string; value: string | number }[]
  validation?: {
    min?: number
    max?: number
    pattern?: string
    message?: string
  }
}

// Settings Types
export interface DashboardSettings {
  theme: 'light' | 'dark'
  refreshInterval: number
  notifications: boolean
  defaultView: string
  chartAnimations: boolean
  compactMode: boolean
}

// Utility Types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P]
}

export type Nullable<T> = T | null

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>

export type TimeRange = '1h' | '6h' | '24h' | '7d' | '30d'

export type SortOrder = 'asc' | 'desc'

export interface SortConfig {
  field: string
  order: SortOrder
}

export interface FilterConfig {
  field: string
  operator: 'eq' | 'ne' | 'gt' | 'lt' | 'gte' | 'lte' | 'contains' | 'in'
  value: any
}