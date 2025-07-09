import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/services/api'
import { useSettingsStore } from '@/stores/settingsstore'
import type { NetworkNode, TopologyData, PrometheusMetric } from '@/types'

// Query keys
export const queryKeys = {
  nodes: ['network', 'nodes'] as const,
  node: (id: string) => ['network', 'nodes', id] as const,
  topology: ['network', 'topology'] as const,
  metrics: ['network', 'metrics'] as const,
  events: (filters?: any) => ['network', 'events', filters] as const,
  prometheus: (query: string) => ['prometheus', query] as const,
}

// Nodes queries
export function useNodes() {
  const refreshInterval = useSettingsStore(state => state.refreshInterval)
  
  return useQuery({
    queryKey: queryKeys.nodes,
    queryFn: async () => {
      const response = await api.network.getNodes()
      return response.data as NetworkNode[]
    },
    refetchInterval: refreshInterval * 1000,
    staleTime: (refreshInterval * 1000) / 2,
  })
}

export function useNode(nodeId: string) {
  return useQuery({
    queryKey: queryKeys.node(nodeId),
    queryFn: async () => {
      const response = await api.network.getNode(nodeId)
      return response.data as NetworkNode
    },
    enabled: !!nodeId,
  })
}

// Topology query
export function useTopology() {
  const refreshInterval = useSettingsStore(state => state.refreshInterval)
  
  return useQuery({
    queryKey: queryKeys.topology,
    queryFn: async () => {
      const response = await api.network.getTopology()
      return response.data as TopologyData
    },
    refetchInterval: refreshInterval * 1000 * 2, // Refresh less frequently
    staleTime: refreshInterval * 1000,
  })
}

// Metrics query
export function useMetrics() {
  const refreshInterval = useSettingsStore(state => state.refreshInterval)
  
  return useQuery({
    queryKey: queryKeys.metrics,
    queryFn: async () => {
      const response = await api.network.getMetrics()
      return response.data
    },
    refetchInterval: refreshInterval * 1000,
    staleTime: (refreshInterval * 1000) / 2,
  })
}

// Events query
export function useEvents(filters?: any) {
  return useQuery({
    queryKey: queryKeys.events(filters),
    queryFn: async () => {
      const response = await api.network.getEvents(filters)
      return response.data
    },
  })
}

// Prometheus query
export function usePrometheusQuery(query: string, enabled = true) {
  const refreshInterval = useSettingsStore(state => state.refreshInterval)
  
  return useQuery({
    queryKey: queryKeys.prometheus(query),
    queryFn: async () => {
      const response = await api.prometheus.query(query)
      return response.data as PrometheusMetric[]
    },
    enabled: enabled && !!query,
    refetchInterval: refreshInterval * 1000,
    staleTime: (refreshInterval * 1000) / 2,
  })
}

// Node mutations
export function useNodeActions() {
  const queryClient = useQueryClient()
  
  const drainNode = useMutation({
    mutationFn: (nodeId: string) => api.network.drainNode(nodeId),
    onSuccess: (_, nodeId) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.node(nodeId) })
      queryClient.invalidateQueries({ queryKey: queryKeys.nodes })
    },
  })
  
  const restartNode = useMutation({
    mutationFn: (nodeId: string) => api.network.restartNode(nodeId),
    onSuccess: (_, nodeId) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.node(nodeId) })
      queryClient.invalidateQueries({ queryKey: queryKeys.nodes })
    },
  })
  
  return {
    drainNode,
    restartNode,
  }
}

// Prefetch utilities
export function usePrefetch() {
  const queryClient = useQueryClient()
  
  const prefetchNode = (nodeId: string) => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.node(nodeId),
      queryFn: async () => {
        const response = await api.network.getNode(nodeId)
        return response.data
      },
      staleTime: 5 * 60 * 1000, // 5 minutes
    })
  }
  
  const prefetchTopology = () => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.topology,
      queryFn: async () => {
        const response = await api.network.getTopology()
        return response.data
      },
      staleTime: 5 * 60 * 1000,
    })
  }
  
  return {
    prefetchNode,
    prefetchTopology,
  }
}