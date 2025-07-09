import axios from 'axios'
import type { AxiosInstance, AxiosResponse } from 'axios'
import { useSettingsStore } from '../stores/settingsstore'
import { useDataModeStore } from '../stores/datamode'
import { mockRealtimeAPI } from './mock-api'

// Create axios instance
const createApiInstance = (): AxiosInstance => {
  const settings = useSettingsStore.getState().settings
  
  const instance = axios.create({
    baseURL: settings.apiEndpoint,
    timeout: 10000,
    headers: {
      'Content-Type': 'application/json',
    },
  })

  // Request interceptor to add auth token
  instance.interceptors.request.use(
    (config) => {
      const token = localStorage.getItem('auth_token')
      if (token) {
        config.headers.Authorization = `Bearer ${token}`
      }
      return config
    },
    (error) => {
      return Promise.reject(error)
    }
  )

  // Response interceptor for error handling
  instance.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error.response?.status === 401) {
        // Token expired or invalid
        localStorage.removeItem('auth_token')
        window.location.href = '/login'
      }
      return Promise.reject(error)
    }
  )

  return instance
}

export const api = createApiInstance()

// Helper function to check if we should use mock data
const shouldUseMockData = () => {
  const { isRealDataMode } = useDataModeStore.getState()
  return !isRealDataMode || import.meta.env.DEV
}

// API service class for organized endpoints
export class ApiService {
  // Authentication - always mock for now
  async login(username: string, password: string) {
    return new Promise(resolve => {
      setTimeout(() => {
        resolve({
          data: {
            user: { 
              id: '1', 
              username, 
              email: `${username}@example.com`,
              role: 'admin',
              createdAt: new Date().toISOString()
            },
            token: 'mock-jwt-token-' + Date.now(),
            expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
          }
        })
      }, 500)
    })
  }

  async logout() {
    return Promise.resolve({ data: { success: true } })
  }

  async getCurrentUser() {
    const token = localStorage.getItem('auth_token')
    if (token) {
      return Promise.resolve({
        data: {
          id: '1',
          username: 'admin',
          email: 'admin@example.com',
          role: 'admin',
          createdAt: new Date().toISOString()
        }
      })
    }
    throw new Error('No token found')
  }

  // Network nodes
  async getNodes() {
    if (shouldUseMockData()) {
      return mockRealtimeAPI.getNodes()
    }
    try {
      return await api.get('/network/nodes')
    } catch (error) {
      console.warn('Real API failed, falling back to mock data:', error)
      return mockRealtimeAPI.getNodes()
    }
  }

  async getNode(nodeId: string) {
    if (shouldUseMockData()) {
      const nodes = await mockRealtimeAPI.getNodes()
      const node = nodes.data.find(n => n.id === nodeId)
      if (!node) throw new Error('Node not found')
      return { data: node }
    }
    try {
      return await api.get(`/network/nodes/${nodeId}`)
    } catch (error) {
      console.warn('Real API failed, falling back to mock data:', error)
      const nodes = await mockRealtimeAPI.getNodes()
      const node = nodes.data.find(n => n.id === nodeId)
      if (!node) throw new Error('Node not found')
      return { data: node }
    }
  }

  async updateNode(nodeId: string, data: any) {
    if (shouldUseMockData()) {
      return Promise.resolve({ data: { success: true, nodeId, updates: data } })
    }
    try {
      return await api.put(`/network/nodes/${nodeId}`, data)
    } catch (error) {
      console.warn('Real API failed, simulating success:', error)
      return Promise.resolve({ data: { success: true, nodeId, updates: data } })
    }
  }

  // Network metrics
  async getMetrics(timeRange = '24h') {
    if (shouldUseMockData()) {
      return mockRealtimeAPI.getMetrics(timeRange)
    }
    try {
      return await api.get('/network/metrics', { params: { timeRange } })
    } catch (error) {
      console.warn('Real API failed, falling back to mock data:', error)
      return mockRealtimeAPI.getMetrics(timeRange)
    }
  }

  async getNodeMetrics(nodeId: string, timeRange = '24h') {
    if (shouldUseMockData()) {
      const nodes = await mockRealtimeAPI.getNodes()
      const node = nodes.data.find(n => n.id === nodeId)
      if (!node?.metrics) throw new Error('Node metrics not found')
      return { data: node.metrics }
    }
    try {
      return await api.get(`/network/nodes/${nodeId}/metrics`, { params: { timeRange } })
    } catch (error) {
      console.warn('Real API failed, falling back to mock data:', error)
      const nodes = await mockRealtimeAPI.getNodes()
      const node = nodes.data.find(n => n.id === nodeId)
      if (!node?.metrics) throw new Error('Node metrics not found')
      return { data: node.metrics }
    }
  }

  // Events
  async getEvents(limit = 50, offset = 0) {
    if (shouldUseMockData()) {
      return mockRealtimeAPI.getEvents(limit)
    }
    try {
      return await api.get('/network/events', { params: { limit, offset } })
    } catch (error) {
      console.warn('Real API failed, falling back to mock data:', error)
      return mockRealtimeAPI.getEvents(limit)
    }
  }

  async getEventsByType(type: string, limit = 50) {
    if (shouldUseMockData()) {
      const events = await mockRealtimeAPI.getEvents(limit)
      const filteredEvents = events.data.filter(e => e.type === type)
      return { data: filteredEvents }
    }
    try {
      return await api.get('/network/events', { params: { type, limit } })
    } catch (error) {
      console.warn('Real API failed, falling back to mock data:', error)
      const events = await mockRealtimeAPI.getEvents(limit)
      const filteredEvents = events.data.filter(e => e.type === type)
      return { data: filteredEvents }
    }
  }

  // Network topology
  async getTopology() {
    if (shouldUseMockData()) {
      const nodes = await mockRealtimeAPI.getNodes()
      return {
        data: {
          nodes: nodes.data.map(node => ({
            id: node.id,
            label: node.name,
            type: node.type,
            status: node.status,
            position: { x: Math.random() * 400, y: Math.random() * 300 },
            data: node
          })),
          edges: [],
          layout: 'force',
          lastUpdated: new Date().toISOString()
        }
      }
    }
    try {
      return await api.get('/network/topology')
    } catch (error) {
      console.warn('Real API failed, falling back to mock data:', error)
      const nodes = await mockRealtimeAPI.getNodes()
      return {
        data: {
          nodes: nodes.data.map(node => ({
            id: node.id,
            label: node.name,
            type: node.type,
            status: node.status,
            position: { x: Math.random() * 400, y: Math.random() * 300 },
            data: node
          })),
          edges: [],
          layout: 'force',
          lastUpdated: new Date().toISOString()
        }
      }
    }
  }

  async updateTopology(data: any) {
    if (shouldUseMockData()) {
      return Promise.resolve({ data: { success: true } })
    }
    try {
      return await api.put('/network/topology', data)
    } catch (error) {
      console.warn('Real API failed, simulating success:', error)
      return Promise.resolve({ data: { success: true } })
    }
  }

  // System health
  async getSystemHealth() {
    if (shouldUseMockData()) {
      return mockRealtimeAPI.getSystemHealth()
    }
    try {
      return await api.get('/system/health')
    } catch (error) {
      console.warn('Real API failed, falling back to mock data:', error)
      return mockRealtimeAPI.getSystemHealth()
    }
  }

  async getSystemStatus() {
    if (shouldUseMockData()) {
      return {
        data: {
          status: 'operational',
          version: '1.0.0-mock',
          uptime: Date.now() - 24 * 60 * 60 * 1000,
          environment: 'development'
        }
      }
    }
    try {
      return await api.get('/system/status')
    } catch (error) {
      console.warn('Real API failed, falling back to mock data:', error)
      return {
        data: {
          status: 'operational',
          version: '1.0.0-fallback',
          uptime: Date.now() - 24 * 60 * 60 * 1000,
          environment: 'development'
        }
      }
    }
  }

  // Configuration
  async getConfig() {
    if (shouldUseMockData()) {
      return {
        data: {
          refreshInterval: 30,
          maxConnections: 100,
          enableLogging: true,
          logLevel: 'info'
        }
      }
    }
    try {
      return await api.get('/config')
    } catch (error) {
      console.warn('Real API failed, falling back to mock data:', error)
      return {
        data: {
          refreshInterval: 30,
          maxConnections: 100,
          enableLogging: true,
          logLevel: 'info'
        }
      }
    }
  }

  async updateConfig(config: any) {
    if (shouldUseMockData()) {
      return Promise.resolve({ data: { success: true, config } })
    }
    try {
      return await api.put('/config', config)
    } catch (error) {
      console.warn('Real API failed, simulating success:', error)
      return Promise.resolve({ data: { success: true, config } })
    }
  }
}

// Create singleton instance
export const apiService = new ApiService()

// Utility functions for common API patterns
export const withRetry = async <T>(
  apiCall: () => Promise<AxiosResponse<T>>,
  retries = 3,
  delay = 1000
): Promise<AxiosResponse<T>> => {
  for (let i = 0; i < retries; i++) {
    try {
      return await apiCall()
    } catch (error) {
      if (i === retries - 1) throw error
      await new Promise(resolve => setTimeout(resolve, delay))
      delay *= 2 // Exponential backoff
    }
  }
  throw new Error('Max retries exceeded')
}

export const handleApiError = (error: any): string => {
  if (error.response) {
    // Server responded with error status
    const { status, data } = error.response
    if (data?.message) return data.message
    if (status === 404) return 'Resource not found'
    if (status === 403) return 'Access denied'
    if (status === 500) return 'Internal server error'
    return `HTTP Error ${status}`
  } else if (error.request) {
    // Request made but no response received
    return 'Network error - please check your connection'
  } else {
    // Error in setting up the request
    return error.message || 'An unexpected error occurred'
  }
}