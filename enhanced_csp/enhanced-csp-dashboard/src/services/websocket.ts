import { io, Socket } from 'socket.io-client'
import { useSettingsStore } from '../stores/settingsstore'
import { useNetworkStore } from '../stores/networkstore'
import type { NetworkNode, NetworkMetrics, NetworkEvent } from '../types'

class WebSocketService {
  private socket: Socket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectInterval = 1000
  private isConnecting = false

  connect(): void {
    if (this.socket?.connected || this.isConnecting) {
      return
    }

    this.isConnecting = true
    const settings = useSettingsStore.getState().settings
    const token = localStorage.getItem('auth_token')

    try {
      this.socket = io(settings.websocketUrl, {
        auth: {
          token
        },
        transports: ['websocket'],
        forceNew: true,
        reconnection: true,
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: this.reconnectInterval,
        reconnectionDelayMax: 5000,
        timeout: 10000
      })

      this.setupEventListeners()
    } catch (error) {
      console.error('Failed to create socket connection:', error)
      this.isConnecting = false
    }
  }

  private setupEventListeners(): void {
    if (!this.socket) return

    // Connection events
    this.socket.on('connect', () => {
      console.log('WebSocket connected')
      this.isConnecting = false
      this.reconnectAttempts = 0
      useNetworkStore.getState().setConnectionStatus(true)
    })

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason)
      useNetworkStore.getState().setConnectionStatus(false)
    })

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error)
      this.isConnecting = false
      this.handleReconnection()
    })

    // Network data events
    this.socket.on('nodes_update', (nodes: NetworkNode[]) => {
      useNetworkStore.getState().setNodes(nodes)
    })

    this.socket.on('node_update', (data: { nodeId: string; updates: Partial<NetworkNode> }) => {
      useNetworkStore.getState().updateNode(data.nodeId, data.updates)
    })

    this.socket.on('metrics_update', (metrics: NetworkMetrics) => {
      useNetworkStore.getState().setMetrics(metrics)
    })

    this.socket.on('new_event', (event: NetworkEvent) => {
      useNetworkStore.getState().addEvent(event)
    })

    // System events
    this.socket.on('system_alert', (alert: { message: string; severity: string }) => {
      console.warn('System alert:', alert)
      // Could trigger a toast notification here
    })

    // Heartbeat
    this.socket.on('ping', () => {
      this.socket?.emit('pong')
    })
  }

  private handleReconnection(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
      
      setTimeout(() => {
        if (!this.socket?.connected) {
          this.connect()
        }
      }, this.reconnectInterval * this.reconnectAttempts)
    } else {
      console.error('Max reconnection attempts reached')
      useNetworkStore.getState().setConnectionStatus(false)
    }
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
    this.reconnectAttempts = 0
    this.isConnecting = false
    useNetworkStore.getState().setConnectionStatus(false)
  }

  // Send data to server
  emit(event: string, data?: any): void {
    if (this.socket?.connected) {
      this.socket.emit(event, data)
    } else {
      console.warn('WebSocket not connected, cannot emit event:', event)
    }
  }

  // Subscribe to custom events
  on(event: string, callback: (...args: any[]) => void): void {
    this.socket?.on(event, callback)
  }

  // Unsubscribe from events
  off(event: string, callback?: (...args: any[]) => void): void {
    this.socket?.off(event, callback)
  }

  // Check connection status
  isConnected(): boolean {
    return this.socket?.connected || false
  }

  // Request specific data
  requestNodesUpdate(): void {
    this.emit('request_nodes_update')
  }

  requestMetricsUpdate(): void {
    this.emit('request_metrics_update')
  }

  requestTopologyUpdate(): void {
    this.emit('request_topology_update')
  }

  // Send commands to network
  sendNodeCommand(nodeId: string, command: string, params?: any): void {
    this.emit('node_command', { nodeId, command, params })
  }

  // Update subscription preferences
  updateSubscriptions(subscriptions: string[]): void {
    this.emit('update_subscriptions', subscriptions)
  }
}

// Create singleton instance
export const websocketService = new WebSocketService()

// Hook for React components
export const useWebSocket = () => {
  const connect = () => websocketService.connect()
  const disconnect = () => websocketService.disconnect()
  const isConnected = () => websocketService.isConnected()
  const emit = (event: string, data?: any) => websocketService.emit(event, data)
  
  return {
    connect,
    disconnect,
    isConnected,
    emit,
    requestNodesUpdate: () => websocketService.requestNodesUpdate(),
    requestMetricsUpdate: () => websocketService.requestMetricsUpdate(),
    requestTopologyUpdate: () => websocketService.requestTopologyUpdate(),
    sendNodeCommand: (nodeId: string, command: string, params?: any) => 
      websocketService.sendNodeCommand(nodeId, command, params)
  }
}