import React, { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { SettingsDrawer } from '../components/features/settings/settingsdrawer'
import { apiService } from '../services/api'
import { useDataModeStore } from '../stores/datamode'
import { 
  Settings, 
  RefreshCw, 
  Clock,
  CheckCircle,
  Activity,
  Users,
  Server,
  Zap,
  Database,
  Wifi,
  AlertCircle
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

// Query hooks
const useNodes = () => {
  return useQuery({
    queryKey: ['nodes'],
    queryFn: () => apiService.getNodes().then(res => res.data),
    refetchInterval: 30 * 1000, // 30 seconds
    staleTime: 30000,
  })
}

const useNetworkMetrics = (timeRange = '1h') => {
  return useQuery({
    queryKey: ['metrics', timeRange],
    queryFn: () => apiService.getMetrics(timeRange).then(res => res.data),
    refetchInterval: 30 * 1000,
    staleTime: 60000,
  })
}

const useEvents = (limit = 10) => {
  return useQuery({
    queryKey: ['events', limit],
    queryFn: () => apiService.getEvents(limit).then(res => res.data),
    staleTime: 30000,
  })
}

// Data Mode Toggle Component
const DataModeToggle: React.FC = () => {
  const { isRealDataMode, toggleDataMode } = useDataModeStore()
  const queryClient = useQueryClient()

  const handleToggle = () => {
    toggleDataMode()
    // Invalidate all queries to fetch data from the new source
    queryClient.invalidateQueries()
  }

  return (
    <div className="flex items-center space-x-2">
      <div className="flex items-center space-x-1 text-xs text-gray-500">
        {isRealDataMode ? (
          <>
            <Wifi className="h-3 w-3" />
            <span>Real API</span>
          </>
        ) : (
          <>
            <Database className="h-3 w-3" />
            <span>Mock Data</span>
          </>
        )}
      </div>
      <Button
        size="sm"
        variant={isRealDataMode ? "default" : "outline"}
        onClick={handleToggle}
        className="min-w-[120px]"
      >
        {isRealDataMode ? (
          <>
            <Wifi className="h-3 w-3 mr-2" />
            Real Data
          </>
        ) : (
          <>
            <Database className="h-3 w-3 mr-2" />
            Mock Data
          </>
        )}
      </Button>
    </div>
  )
}

// KPI Card Component
const KpiCard: React.FC<{
  title: string
  value: string
  change: string
  trend: 'up' | 'down'
  icon: React.ComponentType<{ className?: string }>
}> = ({ title, value, change, trend, icon: Icon }) => {
  const trendColor = trend === 'up' ? 'text-green-600' : 'text-red-600'
  const badgeVariant = trend === 'up' ? 'default' : 'destructive'

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-gray-600">
          {title}
        </CardTitle>
        <Icon className="h-4 w-4 text-gray-500" />
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-2xl font-bold text-gray-900">{value}</div>
            <div className="flex items-center space-x-1 mt-1">
              <Badge variant={badgeVariant} className="text-xs">
                {change}
              </Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// Dashboard Overview Component
const DashboardOverview: React.FC = () => {
  const { data: nodes } = useNodes()
  const { data: metrics } = useNetworkMetrics()
  
  const kpiData = [
    {
      title: 'Active Nodes',
      value: nodes?.filter(n => n.status === 'active')?.length?.toString() || '0',
      change: '+12%',
      trend: 'up' as const,
      icon: Server
    },
    {
      title: 'Connected Peers',
      value: nodes?.reduce((sum, node) => sum + node.connections, 0)?.toString() || '0',
      change: '+8%',
      trend: 'up' as const,
      icon: Users
    },
    {
      title: 'Network Throughput',
      value: metrics?.totalThroughput ? `${metrics.totalThroughput.toFixed(1)} GB/s` : '0 GB/s',
      change: '+15%',
      trend: 'up' as const,
      icon: Activity
    },
    {
      title: 'Network Uptime',
      value: metrics?.networkUptime ? `${Math.round(metrics.networkUptime)}%` : '0%',
      change: '+2%',
      trend: 'up' as const,
      icon: Zap
    }
  ]

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {kpiData.map((kpi, index) => (
          <KpiCard key={index} {...kpi} />
        ))}
      </div>

      {/* Node Status Table */}
      <Card>
        <CardHeader>
          <CardTitle>Network Nodes</CardTitle>
        </CardHeader>
        <CardContent>
          {nodes && nodes.length > 0 ? (
            <div className="space-y-3">
              {nodes.map((node) => (
                <div key={node.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${
                      node.status === 'active' ? 'bg-green-500' :
                      node.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                    }`}></div>
                    <div>
                      <p className="font-medium">{node.name}</p>
                      <p className="text-sm text-gray-500">{node.ipAddress}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium">{node.connections} connections</p>
                    <p className="text-xs text-gray-500 capitalize">{node.type}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              Loading network data...
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// Real-time Stats Component
const RealtimeStats: React.FC = () => {
  const { data: nodes, isLoading: nodesLoading, dataUpdatedAt, refetch: refetchNodes } = useNodes()
  const { data: metrics, isLoading: metricsLoading, refetch: refetchMetrics } = useNetworkMetrics('1h')
  const { data: events, isLoading: eventsLoading, refetch: refetchEvents } = useEvents(10)
  const { isRealDataMode } = useDataModeStore()

  const handleRefreshAll = () => {
    refetchNodes()
    refetchMetrics()
    refetchEvents()
  }

  return (
    <>
      {/* Connection Status */}
      <Card className="mb-4">
        <CardContent className="py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {isRealDataMode ? (
                <>
                  <AlertCircle className="h-4 w-4 text-orange-600" />
                  <span className="text-sm font-medium text-orange-600">Real Data Mode</span>
                  <span className="text-xs text-gray-500">(May fall back to mock if API unavailable)</span>
                </>
              ) : (
                <>
                  <CheckCircle className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-medium text-blue-600">Mock Data Mode</span>
                </>
              )}
              {dataUpdatedAt && (
                <div className="flex items-center space-x-1 text-xs text-gray-500">
                  <Clock className="h-3 w-3" />
                  <span>Updated {formatDistanceToNow(dataUpdatedAt, { addSuffix: true })}</span>
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-2">
              <DataModeToggle />
              <Button size="sm" variant="outline" onClick={handleRefreshAll}>
                <RefreshCw className="h-3 w-3 mr-1" />
                Refresh
              </Button>
              <Badge variant={isRealDataMode ? "destructive" : "secondary"}>
                {isRealDataMode ? "REAL" : "DEMO"}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
              Data Status
              {!nodesLoading && dataUpdatedAt && (
                <Badge variant="outline" className="text-xs">
                  {formatDistanceToNow(dataUpdatedAt, { addSuffix: true })}
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${isRealDataMode ? 'text-orange-600' : 'text-blue-600'}`}>
              {nodesLoading ? '...' : isRealDataMode ? 'REAL' : 'MOCK'}
            </div>
            <p className="text-xs text-gray-500">
              {isRealDataMode ? 'Live API data' : 'Mock data simulation'}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Active Nodes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {nodesLoading ? '...' : nodes?.filter(n => n.status === 'active')?.length || 0}
            </div>
            <p className="text-xs text-gray-500">
              of {nodesLoading ? '...' : nodes?.length || 0} total
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Network Health</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {metricsLoading ? '...' : metrics?.networkUptime ? `${Math.round(metrics.networkUptime)}%` : 'N/A'}
            </div>
            <p className="text-xs text-gray-500">Uptime</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Recent Events</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {eventsLoading ? '...' : events?.length || 0}
            </div>
            <p className="text-xs text-gray-500">Generated events</p>
          </CardContent>
        </Card>
      </div>
    </>
  )
}

// Main Dashboard Component
const Dashboard: React.FC = () => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const { isRealDataMode } = useDataModeStore()

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Enhanced CSP Dashboard</h1>
            <p className="text-gray-600">
              Real-time network monitoring with {isRealDataMode ? 'live API data' : 'mock data simulation'}
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <DataModeToggle />
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsSettingsOpen(true)}
            >
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Button>
          </div>
        </div>

        <RealtimeStats />

        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="topology">Network Topology</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="logs">Events & Logs</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <DashboardOverview />
          </TabsContent>

          <TabsContent value="topology" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Network Topology</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-96 flex flex-col items-center justify-center text-gray-500">
                  <div className="text-lg font-semibold mb-2">Interactive Network Topology</div>
                  <p className="text-center text-sm">
                    Real-time visualization of network nodes and connections
                    <br />
                    Coming soon: Interactive graph with drag-and-drop functionality
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="metrics" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-96 flex flex-col items-center justify-center text-gray-500">
                  <div className="text-lg font-semibold mb-2">Live Performance Charts</div>
                  <p className="text-center text-sm">
                    Real-time charts showing throughput, latency, CPU, and memory usage
                    <br />
                    Coming soon: Time-series graphs with historical data
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="logs" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>System Events & Logs</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-96 flex flex-col items-center justify-center text-gray-500">
                  <div className="text-lg font-semibold mb-2">Real-time Event Stream</div>
                  <p className="text-center text-sm">
                    Live stream of system events, alerts, and network activity
                    <br />
                    Coming soon: Filterable event log with severity levels
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      <SettingsDrawer
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </div>
  )
}

export default Dashboard