import React, { createContext, useContext, useEffect, useState } from 'react'
import { useWebSocket, useRealtimeData } from '../../hooks/usewebsocket'
import { useNetworkStore } from '../../stores/networkstore'
import { useAuthStore } from '../../stores/authstore'
import { useToast } from '../ui/use-toast'

interface RealtimeContextType {
  isConnected: boolean
  lastUpdate: Date | null
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error'
  reconnect: () => void
  subscribeToUpdates: (subscriptions: string[]) => void
}

const RealtimeContext = createContext<RealtimeContextType | undefined>(undefined)

interface RealtimeProviderProps {
  children: React.ReactNode
}

export const RealtimeProvider: React.FC<RealtimeProviderProps> = ({ children }) => {
  const { isConnected, connect, disconnect, requestUpdates } = useWebSocket()
  const isAuthenticated = useAuthStore(state => state.isAuthenticated)
  const { toast } = useToast()
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected')

  // Subscribe to real-time data streams
  useRealtimeData([
    'nodes_update',
    'metrics_update', 
    'events_update',
    'topology_update'
  ])

  // Monitor connection status
  useEffect(() => {
    if (isConnected) {
      setConnectionStatus('connected')
      setLastUpdate(new Date())
      toast({
        title: "Connected",
        description: "Real-time data connection established",
      })
    } else if (isAuthenticated) {
      setConnectionStatus('connecting')
    } else {
      setConnectionStatus('disconnected')
    }
  }, [isConnected, isAuthenticated, toast])

  // Auto-connect when authenticated
  useEffect(() => {
    if (isAuthenticated && !isConnected) {
      connect()
    } else if (!isAuthenticated && isConnected) {
      disconnect()
    }
  }, [isAuthenticated, isConnected, connect, disconnect])

  // Update timestamp when data changes
  const networkData = useNetworkStore(state => ({ 
    nodes: state.nodes, 
    metrics: state.metrics, 
    events: state.events 
  }))
  
  useEffect(() => {
    setLastUpdate(new Date())
  }, [networkData])

  const reconnect = () => {
    disconnect()
    setTimeout(() => {
      connect()
    }, 1000)
  }

  const subscribeToUpdates = (subscriptions: string[]) => {
    if (isConnected) {
      // This would be implemented in the WebSocket service
      requestUpdates()
    }
  }

  const contextValue: RealtimeContextType = {
    isConnected,
    lastUpdate,
    connectionStatus,
    reconnect,
    subscribeToUpdates
  }

  return (
    <RealtimeContext.Provider value={contextValue}>
      {children}
    </RealtimeContext.Provider>
  )
}

export const useRealtime = (): RealtimeContextType => {
  const context = useContext(RealtimeContext)
  if (!context) {
    throw new Error('useRealtime must be used within a RealtimeProvider')
  }
  return context
}