import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService } from '../services/api'
import { useSettingsStore } from '../stores/settingsstore'
import type { NetworkNode, NetworkMetrics, NetworkEvent, TimeRange } from '../types'

// Query keys
export const queryKeys = {
  nodes: ['nodes'] as const,
  node: (id: string) => ['nodes', id] as const,
  metrics: (timeRange: TimeRange) => ['metrics', timeRange] as const,
  nodeMetrics: (id: string, timeRange: TimeRange) => ['nodes', id, 'metrics', timeRange] as const,
  events: (limit: number, offset: number) => ['events', limit, offset] as const,
  eventsByType: (type: string, limit: number) => ['events', type, limit] as const,
  topology: ['topology'] as const,
  systemHealth: ['system', 'health'] as const,
  systemStatus: ['system', 'status'] as const,
  config: ['config'] as const
}

// Nodes queries
export const useNodes = () => {
  const refreshInterval = useSettingsStore(state => state.settings.refreshInterval)
  
  return useQuery({
    queryKey: queryKeys.nodes,
    queryFn: () => apiService.getNodes().then(res => res.data),
    refetchInterval: refreshInterval * 1000,
    staleTime: 30000, // 30 seconds
    gcTime: 5 * 60 * 1000, // 5 minutes
  })
}

export const useNode = (nodeId: string) => {
  return useQuery({
    queryKey: queryKeys.node(nodeId),
    queryFn: () => apiService.getNode(nodeId).then(res => res.data),
    enabled: !!nodeId,
    staleTime: 30000,
  })
}

// Metrics queries
export const useNetworkMetrics = (timeRange: TimeRange = '24h') => {
  const refreshInterval = useSettingsStore(state => state.settings.refreshInterval)
  
  return useQuery({
    queryKey: queryKeys.metrics(timeRange),
    queryFn: () => apiService.getMetrics(timeRange).then(res => res.data),
    refetchInterval: refreshInterval * 1000,
    staleTime: 60000, // 1 minute
  })
}

export const useNodeMetrics = (nodeId: string, timeRange: TimeRange = '24h') => {
  return useQuery({
    queryKey: queryKeys.nodeMetrics(nodeId, timeRange),
    queryFn: () => apiService.getNodeMetrics(nodeId, timeRange).then(res => res.data),
    enabled: !!nodeId,
    staleTime: 60000,
  })
}

// Events queries
export const useEvents = (limit = 50, offset = 0) => {
  return useQuery({
    queryKey: queryKeys.events(limit, offset),
    queryFn: () => apiService.getEvents(limit, offset).then(res => res.data),
    staleTime: 30000,
  })
}

export const useEventsByType = (type: string, limit = 50) => {
  return useQuery({
    queryKey: queryKeys.eventsByType(type, limit),
    queryFn: () => apiService.getEventsByType(type, limit).then(res => res.data),
    enabled: !!type,
    staleTime: 30000,
  })
}

// Topology query
export const useTopology = () => {
  return useQuery({
    queryKey: queryKeys.topology,
    queryFn: () => apiService.getTopology().then(res => res.data),
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

// System queries
export const useSystemHealth = () => {
  const refreshInterval = useSettingsStore(state => state.settings.refreshInterval)
  
  return useQuery({
    queryKey: queryKeys.systemHealth,
    queryFn: () => apiService.getSystemHealth().then(res => res.data),
    refetchInterval: refreshInterval * 1000,
    staleTime: 30000,
  })
}

export const useSystemStatus = () => {
  return useQuery({
    queryKey: queryKeys.systemStatus,
    queryFn: () => apiService.getSystemStatus().then(res => res.data),
    staleTime: 60000,
  })
}

// Configuration query
export const useConfig = () => {
  return useQuery({
    queryKey: queryKeys.config,
    queryFn: () => apiService.getConfig().then(res => res.data),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

// Mutations
export const useUpdateNode = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ nodeId, data }: { nodeId: string; data: Partial<NetworkNode> }) =>
      apiService.updateNode(nodeId, data),
    onSuccess: (_, variables) => {
      // Invalidate and refetch node data
      queryClient.invalidateQueries({ queryKey: queryKeys.node(variables.nodeId) })
      queryClient.invalidateQueries({ queryKey: queryKeys.nodes })
    },
  })
}

export const useUpdateTopology = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiService.updateTopology(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.topology })
    },
  })
}

export const useUpdateConfig = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (config: any) => apiService.updateConfig(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.config })
    },
  })
}

// Utility hooks
export const useRefreshData = () => {
  const queryClient = useQueryClient()
  
  const refreshAll = () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.nodes })
    queryClient.invalidateQueries({ queryKey: ['metrics'] })
    queryClient.invalidateQueries({ queryKey: ['events'] })
    queryClient.invalidateQueries({ queryKey: queryKeys.topology })
  }
  
  const refreshNodes = () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.nodes })
  }
  
  const refreshMetrics = () => {
    queryClient.invalidateQueries({ queryKey: ['metrics'] })
  }
  
  const refreshEvents = () => {
    queryClient.invalidateQueries({ queryKey: ['events'] })
  }
  
  return {
    refreshAll,
    refreshNodes,
    refreshMetrics,
    refreshEvents
  }
}

// Prefetch hooks for performance
export const usePrefetchNode = () => {
  const queryClient = useQueryClient()
  
  return (nodeId: string) => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.node(nodeId),
      queryFn: () => apiService.getNode(nodeId).then(res => res.data),
      staleTime: 30000,
    })
  }
}

export const usePrefetchNodeMetrics = () => {
  const queryClient = useQueryClient()
  
  return (nodeId: string, timeRange: TimeRange = '24h') => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.nodeMetrics(nodeId, timeRange),
      queryFn: () => apiService.getNodeMetrics(nodeId, timeRange).then(res => res.data),
      staleTime: 60000,
    })
  }
}