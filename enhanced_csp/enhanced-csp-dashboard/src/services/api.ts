import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios'
import { storage } from '@/utils'
import type { ApiResponse, ApiError } from '@/types'

class ApiClient {
  private client: AxiosInstance
  private refreshPromise: Promise<string> | null = null

  constructor() {
    this.client = axios.create({
      baseURL: import.meta.env.VITE_API_URL || '/api',
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = storage.get('auth_token', null)
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
        return config
      },
      (error) => Promise.reject(error)
    )

    // Response interceptor to handle token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean }

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true

          try {
            const newToken = await this.refreshToken()
            if (newToken && originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${newToken}`
              return this.client(originalRequest)
            }
          } catch (refreshError) {
            // Refresh failed, redirect to login
            storage.remove('auth_token')
            storage.remove('refresh_token')
            window.location.href = '/login'
            return Promise.reject(refreshError)
          }
        }

        return Promise.reject(this.formatError(error))
      }
    )
  }

  private async refreshToken(): Promise<string> {
    if (this.refreshPromise) {
      return this.refreshPromise
    }

    this.refreshPromise = new Promise(async (resolve, reject) => {
      try {
        const refreshToken = storage.get('refresh_token', null)
        if (!refreshToken) {
          throw new Error('No refresh token available')
        }

        const response = await axios.post('/api/auth/refresh', {
          refreshToken,
        })

        const { token } = response.data
        storage.set('auth_token', token)
        resolve(token)
      } catch (error) {
        reject(error)
      } finally {
        this.refreshPromise = null
      }
    })

    return this.refreshPromise
  }

  private formatError(error: AxiosError): ApiError {
    if (error.response?.data) {
      const data = error.response.data as any
      return {
        code: data.code || 'UNKNOWN_ERROR',
        message: data.message || error.message,
        details: data.details || {},
        timestamp: new Date().toISOString(),
      }
    }

    return {
      code: 'NETWORK_ERROR',
      message: error.message || 'Network request failed',
      timestamp: new Date().toISOString(),
    }
  }

  // Generic request methods
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.client.get<ApiResponse<T>>(url, config)
    return response.data
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.client.post<ApiResponse<T>>(url, data, config)
    return response.data
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.client.put<ApiResponse<T>>(url, data, config)
    return response.data
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.client.delete<ApiResponse<T>>(url, config)
    return response.data
  }

  // Set custom headers
  setHeader(key: string, value: string) {
    this.client.defaults.headers.common[key] = value
  }

  // Remove custom headers
  removeHeader(key: string) {
    delete this.client.defaults.headers.common[key]
  }
}

export const apiClient = new ApiClient()

// Specific API endpoints
export const api = {
  // Auth
  auth: {
    login: (credentials: { email: string; password: string }) =>
      apiClient.post('/auth/login', credentials),
    logout: () => apiClient.post('/auth/logout'),
    refresh: (refreshToken: string) =>
      apiClient.post('/auth/refresh', { refreshToken }),
    me: () => apiClient.get('/auth/me'),
  },

  // Network
  network: {
    getNodes: () => apiClient.get('/network/nodes'),
    getNode: (id: string) => apiClient.get(`/network/nodes/${id}`),
    getMetrics: () => apiClient.get('/network/metrics'),
    getTopology: () => apiClient.get('/network/topology'),
    getEvents: (params?: any) => apiClient.get('/network/events', { params }),
    drainNode: (id: string) => apiClient.post(`/network/nodes/${id}/drain`),
    restartNode: (id: string) => apiClient.post(`/network/nodes/${id}/restart`),
  },

  // Prometheus metrics proxy
  prometheus: {
    query: (query: string, time?: number) =>
      apiClient.get('/prometheus/query', { params: { query, time } }),
    queryRange: (query: string, start: number, end: number, step: number) =>
      apiClient.get('/prometheus/query_range', {
        params: { query, start, end, step },
      }),
  },
}