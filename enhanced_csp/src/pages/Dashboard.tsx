import { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import { MainLayout } from '@/components/layout/MainLayout'
import { DashboardOverview } from '@/components/features/dashboard/DashboardOverview'
import { TopologyView } from '@/components/features/topology/TopologyView'
import { MetricsView } from '@/components/features/metrics/MetricsView'
import { NodesTable } from '@/components/features/nodes/NodesTable'
import { EventsFeed } from '@/components/features/events/EventsFeed'
import { useWebSocket, useSSE } from '@/hooks/useWebSocket'
import { useNodes } from '@/hooks/useQueries'
import { useNetworkStore } from '@/stores/networkStore'

export default function Dashboard() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const { isConnected } = useWebSocket()
  const { data: nodes, isLoading } = useNodes()
  const { setNodes } = useNetworkStore()

  // Update store when nodes data changes
  if (nodes) {
    setNodes(nodes)
  }

  // Use SSE for events
  useSSE()

  return (
    <MainLayout
      sidebarOpen={sidebarOpen}
      setSidebarOpen={setSidebarOpen}
      isConnected={isConnected}
    >
      <Routes>
        <Route path="/" element={<DashboardOverview isLoading={isLoading} />} />
        <Route path="/topology" element={<TopologyView />} />
        <Route path="/metrics" element={<MetricsView />} />
        <Route path="/nodes" element={<NodesTable />} />
        <Route path="/events" element={<EventsFeed />} />
      </Routes>
    </MainLayout>
  )
}