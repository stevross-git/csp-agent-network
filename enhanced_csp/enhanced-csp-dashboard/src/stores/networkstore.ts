import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import type { NetworkNode, NetworkMetrics, NetworkEvent } from '../types'

interface NetworkState {
  nodes: NetworkNode[]
  metrics: NetworkMetrics | null
  events: NetworkEvent[]
  isConnected: boolean
  lastUpdate: Date | null
  
  // Actions
  setNodes: (nodes: NetworkNode[]) => void
  updateNode: (nodeId: string, updates: Partial<NetworkNode>) => void
  setMetrics: (metrics: NetworkMetrics) => void
  addEvent: (event: NetworkEvent) => void
  setConnectionStatus: (connected: boolean) => void
  clearEvents: () => void
  
  // Getters
  getActiveNodes: () => NetworkNode[]
  getNodeById: (id: string) => NetworkNode | undefined
  getRecentEvents: (limit: number) => NetworkEvent[]
}

export const useNetworkStore = create<NetworkState>()(
  subscribeWithSelector((set, get) => ({
    nodes: [],
    metrics: null,
    events: [],
    isConnected: false,
    lastUpdate: null,

    setNodes: (nodes) => {
      set({ 
        nodes, 
        lastUpdate: new Date() 
      })
    },

    updateNode: (nodeId, updates) => {
      set((state) => ({
        nodes: state.nodes.map(node =>
          node.id === nodeId ? { ...node, ...updates } : node
        ),
        lastUpdate: new Date()
      }))
    },

    setMetrics: (metrics) => {
      set({ 
        metrics, 
        lastUpdate: new Date() 
      })
    },

    addEvent: (event) => {
      set((state) => ({
        events: [event, ...state.events].slice(0, 100), // Keep last 100 events
        lastUpdate: new Date()
      }))
    },

    setConnectionStatus: (connected) => {
      set({ isConnected: connected })
    },

    clearEvents: () => {
      set({ events: [] })
    },

    // Getters
    getActiveNodes: () => {
      return get().nodes.filter(node => node.status === 'active')
    },

    getNodeById: (id) => {
      return get().nodes.find(node => node.id === id)
    },

    getRecentEvents: (limit = 10) => {
      return get().events.slice(0, limit)
    }
  }))
)

// Selectors for common use cases
export const selectActiveNodeCount = (state: NetworkState) => 
  state.nodes.filter(node => node.status === 'active').length

export const selectTotalConnections = (state: NetworkState) =>
  state.nodes.reduce((total, node) => total + (node.connections || 0), 0)

export const selectCriticalEvents = (state: NetworkState) =>
  state.events.filter(event => event.severity === 'error' || event.severity === 'warning')

// Subscribe to network state changes for logging
useNetworkStore.subscribe(
  (state) => state.isConnected,
  (connected, prevConnected) => {
    if (connected !== prevConnected) {
      console.log(`Network connection ${connected ? 'established' : 'lost'}`)
    }
  }
)