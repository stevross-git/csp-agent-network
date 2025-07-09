import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'
import { format, formatDistanceToNow, parseISO } from 'date-fns'

// Tailwind CSS class utility
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Format utilities
export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

export function formatNumber(num: number, decimals = 2): string {
  if (num >= 1e9) return (num / 1e9).toFixed(decimals) + 'B'
  if (num >= 1e6) return (num / 1e6).toFixed(decimals) + 'M'
  if (num >= 1e3) return (num / 1e3).toFixed(decimals) + 'K'
  return num.toFixed(decimals)
}

export function formatPercentage(value: number, decimals = 2): string {
  return `${value.toFixed(decimals)}%`
}

export function formatDuration(seconds: number): string {
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  
  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${minutes}m`
  return `${minutes}m`
}

export function formatTimestamp(timestamp: string | Date): string {
  const date = typeof timestamp === 'string' ? parseISO(timestamp) : timestamp
  return format(date, 'HH:mm:ss')
}

export function formatRelativeTime(timestamp: string | Date): string {
  const date = typeof timestamp === 'string' ? parseISO(timestamp) : timestamp
  return formatDistanceToNow(date, { addSuffix: true })
}

// Status utilities
export function getStatusColor(status: string): string {
  const statusColors: Record<string, string> = {
    active: 'text-green-500',
    healthy: 'text-green-500',
    degraded: 'text-yellow-500',
    congested: 'text-yellow-500',
    failed: 'text-red-500',
    maintenance: 'text-blue-500',
    info: 'text-blue-500',
    warning: 'text-yellow-500',
    error: 'text-red-500',
    critical: 'text-red-600'
  }
  return statusColors[status] || 'text-gray-500'
}

export function getStatusBgColor(status: string): string {
  const statusBgColors: Record<string, string> = {
    active: 'bg-green-500',
    healthy: 'bg-green-500',
    degraded: 'bg-yellow-500',
    congested: 'bg-yellow-500',
    failed: 'bg-red-500',
    maintenance: 'bg-blue-500',
    info: 'bg-blue-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500',
    critical: 'bg-red-600'
  }
  return statusBgColors[status] || 'bg-gray-500'
}

// Debounce utility
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null
  
  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null
      func(...args)
    }
    
    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}

// Throttle utility
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false
  
  return function executedFunction(...args: Parameters<T>) {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}

// Local storage utilities
export const storage = {
  get: <T>(key: string, defaultValue: T): T => {
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : defaultValue
    } catch {
      return defaultValue
    }
  },
  
  set: <T>(key: string, value: T): void => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value))
    } catch (error) {
      console.error('Failed to save to localStorage:', error)
    }
  },
  
  remove: (key: string): void => {
    try {
      window.localStorage.removeItem(key)
    } catch (error) {
      console.error('Failed to remove from localStorage:', error)
    }
  }
}

// JWT utilities
export function parseJwt(token: string): any {
  try {
    const base64Url = token.split('.')[1]
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/')
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    )
    return JSON.parse(jsonPayload)
  } catch {
    return null
  }
}

export function isTokenExpired(token: string): boolean {
  const payload = parseJwt(token)
  if (!payload || !payload.exp) return true
  return Date.now() >= payload.exp * 1000
}

// Array utilities
export function groupBy<T>(array: T[], key: keyof T): Record<string, T[]> {
  return array.reduce((groups, item) => {
    const group = String(item[key])
    if (!groups[group]) groups[group] = []
    groups[group].push(item)
    return groups
  }, {} as Record<string, T[]>)
}

export function uniqueBy<T>(array: T[], key: keyof T): T[] {
  const seen = new Set()
  return array.filter(item => {
    const k = item[key]
    return seen.has(k) ? false : seen.add(k)
  })
}

// Color utilities for charts
export const chartColors = {
  primary: '#3b82f6',
  secondary: '#10b981',
  accent: '#f59e0b',
  danger: '#ef4444',
  warning: '#f97316',
  info: '#06b6d4',
  success: '#22c55e',
  muted: '#6b7280'
}

export function getChartColor(index: number): string {
  const colors = Object.values(chartColors)
  return colors[index % colors.length]
}

// Animation utilities
export const transitions = {
  base: 'transition-all duration-200 ease-in-out',
  fast: 'transition-all duration-150 ease-in-out',
  slow: 'transition-all duration-300 ease-in-out',
  spring: 'transition-all duration-300 ease-spring'
}

// Validation utilities
export function isValidEmail(email: string): boolean {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return re.test(email)
}

export function isValidUrl(url: string): boolean {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}

// Error handling utilities
export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  if (error && typeof error === 'object' && 'message' in error) {
    return String(error.message)
  }
  return 'An unknown error occurred'
}

// Performance utilities
export function measurePerformance<T>(
  name: string,
  fn: () => T
): T {
  const start = performance.now()
  const result = fn()
  const end = performance.now()
  console.log(`${name} took ${(end - start).toFixed(2)}ms`)
  return result
}

// Accessibility utilities
export function getAriaLabel(type: string, value: any): string {
  const labels: Record<string, string> = {
    node: `Network node ${value}`,
    status: `Status: ${value}`,
    metric: `Metric value: ${value}`,
    chart: `Chart showing ${value}`,
    button: `${value} button`
  }
  return labels[type] || value
}

// Random data generation for testing
export function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min
}

export function randomElement<T>(array: T[]): T {
  return array[Math.floor(Math.random() * array.length)]
}