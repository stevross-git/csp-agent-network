import { useEffect, useRef, useCallback } from 'react'
import { wsService } from '@/services/websocket'
import { useNetworkStore } from '@/stores/networkStore'
import type { NetworkEvent, NetworkNode } from '@/types'

export function useWebSocket() {
  const { addEvent, updateNode } = useNetworkStore()
  const cleanupRef = useRef<(() => void)[]>([])

  useEffect(() => {
    // Subscribe to WebSocket events
    const unsubscribeHandlers = [
      wsService.on('event', (event: NetworkEvent) => {
        addEvent(event)
      }),
      
      wsService.on('node_update', (data: Partial<NetworkNode> & { id: string }) => {
        updateNode(data.id, data)
      }),
      
      wsService.on('connected', () => {
        console.log('WebSocket connected in hook')
      }),
      
      wsService.on('disconnected', () => {
        console.log('WebSocket disconnected in hook')
      }),
    ]

    cleanupRef.current = unsubscribeHandlers

    // Subscribe to channels
    wsService.subscribe('network_events')
    wsService.subscribe('node_updates')

    return () => {
      // Cleanup subscriptions
      cleanupRef.current.forEach(unsubscribe => unsubscribe())
      wsService.unsubscribe('network_events')
      wsService.unsubscribe('node_updates')
    }
  }, [addEvent, updateNode])

  const sendMessage = useCallback((event: string, data: any) => {
    wsService.send(event, data)
  }, [])

  const isConnected = wsService.isConnected()

  return {
    isConnected,
    sendMessage,
  }
}

export function useSSE() {
  const { addEvent } = useNetworkStore()
  const cleanupRef = useRef<(() => void)[]>([])

  useEffect(() => {
    // Import SSE service dynamically to avoid issues during SSR
    import('@/services/websocket').then(({ sseService }) => {
      // Connect to SSE
      sseService.connect()

      // Subscribe to events
      const unsubscribeHandlers = [
        sseService.on('event', (event: NetworkEvent) => {
          addEvent(event)
        }),
        
        sseService.on('connected', () => {
          console.log('SSE connected')
        }),
        
        sseService.on('error', (error) => {
          console.error('SSE error:', error)
        }),
      ]

      cleanupRef.current = unsubscribeHandlers
    })

    return () => {
      // Cleanup
      cleanupRef.current.forEach(unsubscribe => unsubscribe())
      import('@/services/websocket').then(({ sseService }) => {
        sseService.disconnect()
      })
    }
  }, [addEvent])
}