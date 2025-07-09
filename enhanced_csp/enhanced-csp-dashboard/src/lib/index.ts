// Re-export everything from lib/utils for backward compatibility
export * from '../lib/utils'

// Additional utility functions specific to the network dashboard
import type { NetworkNode, NetworkEvent, MetricsDataPoint } from '../types'

// Network-specific utilities
export function getNodeHealth(node: NetworkNode): 'healthy' | 'warning' | 'critical' {
  if (node.status === 'active' && node.connections > 0) {
    if (node.metrics) {
      const { cpu, memory, disk } = node.metrics
      if (cpu > 90 || memory > 90 || disk > 95) return 'critical'
      if (cpu > 75 || memory > 75 || disk > 85) return 'warning'
    }
    return 'healthy'
  }
  return 'critical'
}

export function calculateNetworkScore(nodes: NetworkNode[]): number {
  if (nodes.length === 0) return 0
  
  const activeNodes = nodes.filter(node => node.status === 'active')
  const baseScore = (activeNodes.length / nodes.length) * 100
  
  // Adjust score based on node health
  const healthScores = activeNodes.map(node => {
    const health = getNodeHealth(node)
    switch (health) {
      case 'healthy': return 1
      case 'warning': return 0.7
      case 'critical': return 0.3
      default: return 0
    }
  })
  
  const avgHealth = healthScores.length > 0 
    ? healthScores.reduce((a, b) => a + b, 0) / healthScores.length 
    : 0
  
  return Math.round(baseScore * avgHealth)
}

export function getEventPriority(event: NetworkEvent): number {
  switch (event.severity) {
    case 'error': return 4
    case 'warning': return 3
    case 'info': return 2
    case 'success': return 1
    default: return 0
  }
}

export function filterCriticalEvents(events: NetworkEvent[]): NetworkEvent[] {
  return events.filter(event => 
    event.severity === 'error' || event.severity === 'warning'
  )
}

export function calculateTrend(dataPoints: MetricsDataPoint[], field: keyof MetricsDataPoint): 'up' | 'down' | 'stable' {
  if (dataPoints.length < 2) return 'stable'
  
  const values = dataPoints.map(point => Number(point[field])).filter(val => !isNaN(val))
  if (values.length < 2) return 'stable'
  
  const first = values[0]
  const last = values[values.length - 1]
  const threshold = Math.abs(first) * 0.05 // 5% threshold
  
  if (last > first + threshold) return 'up'
  if (last < first - threshold) return 'down'
  return 'stable'
}

export function getMetricColor(value: number, thresholds: { warning: number; critical: number }): string {
  if (value >= thresholds.critical) return 'text-red-600'
  if (value >= thresholds.warning) return 'text-yellow-600'
  return 'text-green-600'
}

export function formatLatency(ms: number): string {
  if (ms < 1) return '<1ms'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

export function formatThroughput(bytesPerSecond: number): string {
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s']
  let size = bytesPerSecond
  let unitIndex = 0
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex++
  }
  
  return `${size.toFixed(1)} ${units[unitIndex]}`
}

export function parseTimeRange(range: string): { start: Date; end: Date } {
  const end = new Date()
  const start = new Date()
  
  switch (range) {
    case '1h':
      start.setHours(start.getHours() - 1)
      break
    case '6h':
      start.setHours(start.getHours() - 6)
      break
    case '24h':
      start.setHours(start.getHours() - 24)
      break
    case '7d':
      start.setDate(start.getDate() - 7)
      break
    case '30d':
      start.setDate(start.getDate() - 30)
      break
    default:
      start.setHours(start.getHours() - 24)
  }
  
  return { start, end }
}

export function generateMockData(count: number, baseValue: number, variance: number): number[] {
  return Array.from({ length: count }, () => 
    baseValue + (Math.random() - 0.5) * variance
  )
}

export function smoothDataPoints(points: number[], windowSize = 3): number[] {
  if (points.length <= windowSize) return points
  
  return points.map((point, index) => {
    const start = Math.max(0, index - Math.floor(windowSize / 2))
    const end = Math.min(points.length, start + windowSize)
    const window = points.slice(start, end)
    return window.reduce((sum, val) => sum + val, 0) / window.length
  })
}

export function detectAnomalies(values: number[], threshold = 2): boolean[] {
  if (values.length < 3) return values.map(() => false)
  
  const mean = values.reduce((sum, val) => sum + val, 0) / values.length
  const stdDev = Math.sqrt(
    values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
  )
  
  return values.map(value => Math.abs(value - mean) > threshold * stdDev)
}

export function exportToCSV(data: any[], filename: string): void {
  if (data.length === 0) return
  
  const headers = Object.keys(data[0])
  const csvContent = [
    headers.join(','),
    ...data.map(row => 
      headers.map(header => {
        const value = row[header]
        return typeof value === 'string' && value.includes(',') 
          ? `"${value}"` 
          : value
      }).join(',')
    )
  ].join('\n')
  
  const blob = new Blob([csvContent], { type: 'text/csv' })
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  window.URL.revokeObjectURL(url)
}

export function copyToClipboard(text: string): Promise<boolean> {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text)
      .then(() => true)
      .catch(() => false)
  } else {
    // Fallback for older browsers
    const textArea = document.createElement('textarea')
    textArea.value = text
    textArea.style.position = 'absolute'
    textArea.style.left = '-999999px'
    document.body.appendChild(textArea)
    textArea.focus()
    textArea.select()
    
    try {
      const successful = document.execCommand('copy')
      document.body.removeChild(textArea)
      return Promise.resolve(successful)
    } catch (error) {
      document.body.removeChild(textArea)
      return Promise.resolve(false)
    }
  }
}