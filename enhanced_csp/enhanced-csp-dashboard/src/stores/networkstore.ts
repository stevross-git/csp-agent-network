import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import type { NetworkNode, NetworkEvent, TopologyData, DashboardKPI, NodeFilters } from '@/types'

interface NetworkStore {
  // State
  nodes: NetworkNode[]
  events: NetworkEvent[]
  topology: TopologyData | null
  kpis: DashboardKPI
  selectedNodeId: string | null
  filters: NodeFilters
  isLoading: boolean
  error: string | null

  // Actions
  setNodes: (nodes: NetworkNode[]) => void
  updateNode: (nodeId: string, updates: Partial<NetworkNode>) => void
  addEvent: (event: NetworkEvent) => void
  setEvents: (events: NetworkEvent[]) => void
  clearEvents: () => void
  setTopology: (topology: TopologyData) => void
  setSelectedNode: (nodeId: string | null) => void
  setFilters: (filters: NodeFilters) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  
  // Computed
  getFilteredNodes: () => NetworkNode[]
  getNodeById: (id: string) => NetworkNode | undefined
  getRecentEvents: (count: number) => NetworkEvent[]
}

export const useNetworkStore = create<NetworkStore>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    nodes: [],
    events: [],
    topology: null,
    kpis: {
      activeNodes: 0,
      failedNodes: 0,
      avgLatency: 0,
      packetLoss: 0,
      throughput: 0,
      totalNodes: 0,
      degradedNodes: 0,
    },
    selectedNodeId: null,
    filters: {},
    isLoading: false,
    error: null,

    // Actions
    setNodes: (nodes) => {
      set({ nodes })
      
      // Calculate KPIs
      const kpis: DashboardKPI = {
        totalNodes: nodes.length,
        activeNodes: nodes.filter(n => n.status === 'active').length,
        failedNodes: nodes.filter(n => n.status === 'failed').length,
        degradedNodes: nodes.filter(n => n.status === 'degraded').length,
        avgLatency: nodes.reduce((sum, n) => sum + n.metrics.latency, 0) / nodes.length || 0,
        packetLoss: nodes.reduce((sum, n) => sum + n.metrics.packetLoss, 0) / nodes.length || 0,
        throughput: nodes.reduce((sum, n) => sum + n.metrics.throughput, 0),
      }
      
      set({ kpis })
    },

    addEvent: (event) => {
      set((state) => ({
        events: [event, ...state.events].slice(0, 1000), // Keep last 1000 events
      }))
    },

    setEvents: (events) => {
      set({ events })
    },

    clearEvents: () => {
      set({ events: [] })
    },

    setTopology: (topology) => {
      set({ topology })
    },

    setSelectedNode: (selectedNodeId) => {
      set({ selectedNodeId })
    },

    setFilters: (filters) => {
      set({ filters })
    },

    setLoading: (isLoading) => {
      set({ isLoading })
    },

    setError: (error) => {
      set({ error })
    },

    // Computed
    getFilteredNodes: () => {
      const { nodes, filters } = get()
      
      return nodes.filter(node => {
        if (filters.status?.length && !filters.status.includes(node.status)) {
          return false
        }
        
        if (filters.region?.length && !filters.region.includes(node.region)) {
          return false
        }
        
        if (filters.asg?.length && node.asg && !filters.asg.includes(node.asg)) {
          return false
        }
        
        if (filters.type?.length && !filters.type.includes(node.type)) {
          return false
        }
        
        if (filters.search) {
          const search = filters.search.toLowerCase()
          return (
            node.name.toLowerCase().includes(search) ||
            node.id.toLowerCase().includes(search) ||
            node.ipAddress.includes(search) ||
            node.region.toLowerCase().includes(search)
          )
        }
        
        return true
      })
    },

    getNodeById: (id) => {
      return get().nodes.find(node => node.id === id)
    },

    getRecentEvents: (count) => {
      return get().events.slice(0, count)
    },
  }))
)status === 'active').length,
        failedNodes: nodes.filter(n => n.status === 'failed').length,
        degradedNodes: nodes.filter(n => n.status === 'degraded').length,
        avgLatency: nodes.reduce((sum, n) => sum + n.metrics.latency, 0) / nodes.length || 0,
        packetLoss: nodes.reduce((sum, n) => sum + n.metrics.packetLoss, 0) / nodes.length || 0,
        throughput: nodes.reduce((sum, n) => sum + n.metrics.throughput, 0),
      }
      
      set({ kpis })
    },

    updateNode: (nodeId, updates) => {
      set((state) => ({
        nodes: state.nodes.map(node =>
          node.id === nodeId ? { ...node, ...updates } : node
        ),
      }))
      
      // Recalculate KPIs
      const nodes = get().nodes
      const kpis: DashboardKPI = {
        totalNodes: nodes.length,
        activeNodes: nodes.filter(n => n.status === 'active').length,
        failedNodes: nodes.filter(n => n.