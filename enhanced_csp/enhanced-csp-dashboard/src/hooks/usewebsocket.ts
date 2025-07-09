import { useEffect, useCallback, useRef } from 'react'
import { websocketService } from '../services/websocket'
import { useNetworkStore } from '../stores/networkstore'
import { useAuthStore } from '../stores/authstore'
import { useSettingsStore } from '../stores/settingsstore'

export const useWebSocket = () => {
  const isAuthenticated = useAuthStore(state => state.isAuthenticated)
  const isConnected = useNetworkStore(state => state.isConnected)
  const enableAutoRefresh = useSettingsStore(state => state.settings.enableAutoRefresh)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()

  // Connect when authenticated and auto-refresh is enabled
  useEffect(() => {
    if (isAuthenticated && enableAutoRefresh) {
      websocketService.connect()
    } else {
      websocketService.disconnect()
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [isAuthenticated, enableAutoRefresh])

  // Auto-reconnect logic
  useEffect(() => {
    if (isAuthenticated && enableAutoRefresh && !isConnected) {
      reconnectTimeoutRef.current = setTimeout(() => {
        websocketService.connect()
      }, 5000) // Retry after 5 seconds
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [isAuthenticated, enableAutoRefresh, isConnected])

  const connect = useCallback(() => {
    websocketService.connect()
  }, [])

  const disconnect = useCallback(() => {
    websocketService.disconnect()
  }, [])

  const emit = useCallback((event: string, data?: any) => {
    websocketService.emit(event, data)
  }, [])

  const requestUpdates = useCallback(() => {
    websocketService.requestNodesUpdate()
    websocketService.requestMetricsUpdate()
    websocketService.requestTopologyUpdate()
  }, [])

  const sendNodeCommand = useCallback((nodeId: string, command: string, params?: any) => {
    websocketService.sendNodeCommand(nodeId, command, params)
  }, [])

  return {
    isConnected,
    connect,
    disconnect,
    emit,
    requestUpdates,
    sendNodeCommand
  }
}

// Hook for subscribing to specific WebSocket events
export const useWebSocketEvent = (event: string, callback: (...args: any[]) => void) => {
  useEffect(() => {
    websocketService.on(event, callback)
    
    return () => {
      websocketService.off(event, callback)
    }
  }, [event, callback])
}

// Hook for managing real-time data subscriptions
export const useRealtimeData = (subscriptions: string[] = []) => {
  const { isConnected, requestUpdates } = useWebSocket()

  useEffect(() => {
    if (isConnected && subscriptions.length > 0) {
      // Subscribe to specific data streams
      websocketService.updateSubscriptions(subscriptions)
      
      // Request initial data
      requestUpdates()
    }
  }, [isConnected, subscriptions, requestUpdates])

  return { isConnected }
}

// Hook for handling connection status changes
export const useConnectionStatus = (
  onConnect?: () => void,
  onDisconnect?: () => void
) => {
  const isConnected = useNetworkStore(state => state.isConnected)
  const prevConnectedRef = useRef(isConnected)

  useEffect(() => {
    const prevConnected = prevConnectedRef.current
    
    if (isConnected && !prevConnected && onConnect) {
      onConnect()
    } else if (!isConnected && prevConnected && onDisconnect) {
      onDisconnect()
    }
    
    prevConnectedRef.current = isConnected
  }, [isConnected, onConnect, onDisconnect])

  return isConnected
}