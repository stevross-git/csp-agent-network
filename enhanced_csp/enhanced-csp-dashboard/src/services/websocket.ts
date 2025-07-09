import { io, Socket } from 'socket.io-client'
import { storage } from '@/utils'
import type { WSMessage, NetworkEvent, MetricPoint } from '@/types'

type EventHandler = (data: any) => void
type ConnectionHandler = () => void

export class WebSocketService {
  private socket: Socket | null = null
  private eventHandlers: Map<string, Set<EventHandler>> = new Map()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private isIntentionalDisconnect = false

  constructor() {
    this.connect()
  }

  private connect() {
    const token = storage.get('auth_token', null)
    
    this.socket = io(import.meta.env.VITE_WS_URL || '/', {
      transports: ['websocket'],
      auth: {
        token,
      },
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay,
    })

    this.setupEventListeners()
  }

  private setupEventListeners() {
    if (!this.socket) return

    this.socket.on('connect', () => {
      console.log('WebSocket connected')
      this.reconnectAttempts = 0
      this.emit('connected')
    })

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason)
      if (!this.isIntentionalDisconnect) {
        this.emit('disconnected')
      }
    })

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error)
      this.reconnectAttempts++
      
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        this.emit('max_reconnect_attempts')
      }
    })

    // Handle incoming messages
    this.socket.on('message', (message: WSMessage) => {
      this.handleMessage(message)
    })

    // Handle specific event types
    this.socket.on('metric_update', (data: any) => {
      this.emit('metric_update', data)
    })

    this.socket.on('node_update', (data: any) => {
      this.emit('node_update', data)
    })

    this.socket.on('event', (event: NetworkEvent) => {
      this.emit('event', event)
    })

    this.socket.on('topology_update', (data: any) => {
      this.emit('topology_update', data)
    })
  }

  private handleMessage(message: WSMessage) {
    switch (message.type) {
      case 'metric':
        this.emit('metric', message.payload)
        break
      case 'event':
        this.emit('event', message.payload)
        break
      case 'node_update':
        this.emit('node_update', message.payload)
        break
      case 'topology_update':
        this.emit('topology_update', message.payload)
        break
      default:
        console.warn('Unknown WebSocket message type:', message.type)
    }
  }

  // Public methods
  on(event: string, handler: EventHandler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set())
    }
    this.eventHandlers.get(event)!.add(handler)

    // Return unsubscribe function
    return () => {
      this.off(event, handler)
    }
  }

  off(event: string, handler: EventHandler) {
    const handlers = this.eventHandlers.get(event)
    if (handlers) {
      handlers.delete(handler)
      if (handlers.size === 0) {
        this.eventHandlers.delete(event)
      }
    }
  }

  emit(event: string, data?: any) {
    const handlers = this.eventHandlers.get(event)
    if (handlers) {
      handlers.forEach(handler => handler(data))
    }
  }

  send(event: string, data: any) {
    if (this.socket && this.socket.connected) {
      this.socket.emit(event, data)
    } else {
      console.warn('WebSocket not connected, cannot send:', event)
    }
  }

  subscribe(channel: string) {
    this.send('subscribe', { channel })
  }

  unsubscribe(channel: string) {
    this.send('unsubscribe', { channel })
  }

  disconnect() {
    this.isIntentionalDisconnect = true
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
  }

  reconnect() {
    this.isIntentionalDisconnect = false
    this.disconnect()
    this.connect()
  }

  isConnected(): boolean {
    return this.socket?.connected || false
  }
}

// Create singleton instance
export const wsService = new WebSocketService()

// SSE Service for event stream
export class SSEService {
  private eventSource: EventSource | null = null
  private listeners: Map<string, Set<EventHandler>> = new Map()
  private reconnectTimeout: NodeJS.Timeout | null = null

  connect(url: string = '/api/network/events/stream') {
    const token = storage.get('auth_token', null)
    const fullUrl = `${url}${token ? `?token=${token}` : ''}`

    this.eventSource = new EventSource(fullUrl)

    this.eventSource.onopen = () => {
      console.log('SSE connected')
      this.emit('connected')
    }

    this.eventSource.onerror = (error) => {
      console.error('SSE error:', error)
      this.emit('error', error)
      this.scheduleReconnect()
    }

    this.eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        this.emit('message', data)
        
        // Emit specific event types
        if (data.type) {
          this.emit(data.type, data)
        }
      } catch (error) {
        console.error('Failed to parse SSE message:', error)
      }
    }

    // Listen for specific event types
    this.eventSource.addEventListener('event', (event: any) => {
      try {
        const data = JSON.parse(event.data)
        this.emit('event', data)
      } catch (error) {
        console.error('Failed to parse SSE event:', error)
      }
    })
  }

  private scheduleReconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
    }

    this.reconnectTimeout = setTimeout(() => {
      this.reconnect()
    }, 5000)
  }

  on(event: string, handler: EventHandler) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set())
    }
    this.listeners.get(event)!.add(handler)

    return () => this.off(event, handler)
  }

  off(event: string, handler: EventHandler) {
    const handlers = this.listeners.get(event)
    if (handlers) {
      handlers.delete(handler)
      if (handlers.size === 0) {
        this.listeners.delete(event)
      }
    }
  }

  emit(event: string, data?: any) {
    const handlers = this.listeners.get(event)
    if (handlers) {
      handlers.forEach(handler => handler(data))
    }
  }

  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }

    if (this.eventSource) {
      this.eventSource.close()
      this.eventSource = null
    }
  }

  reconnect() {
    this.disconnect()
    this.connect()
  }

  isConnected(): boolean {
    return this.eventSource?.readyState === EventSource.OPEN
  }
}

export const sseService = new SSEService()