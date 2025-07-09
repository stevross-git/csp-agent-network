import React, { useEffect, useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { Alert, AlertDescription } from '../components/ui/alert'
import DashboardOverview from '../components/features/dashboard/dashboardoverview'
import { SettingsDrawer } from '../components/features/settings/settingsdrawer'
import { RealtimeProvider, useRealtime } from '../components/providers/realtime-provider'
import { useNodes, useNetworkMetrics, useEvents } from '../hooks/usequeries'
import { useNetworkStore } from '../stores/networkstore'
import { 
  Settings, 
  Wifi, 
  WifiOff, 
  RefreshCw, 
  Clock,
  AlertTriangle,
  CheckCircle 
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

const ConnectionStatus: React.FC = () => {
  const { isConnected, lastUpdate, connectionStatus, reconnect } = useRealtime()
  
  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected': return <CheckCircle className="h-4 w-4 text-green-600" />
      case 'connecting': return <RefreshCw className="h-4 w-4 text-yellow-600 animate-spin" />
      case 'error': return <AlertTriangle className="h-4 w-4 text-red-600" />
      default: return <WifiOff className="h-4 w-4 text-gray-400" />
    }
  }

  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Real-time Connected'
      case 'connecting': return 'Connecting...'
      case 'error': return 'Connection Error'
      default: return 'Disconnected'
    }
  }

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-600'
      case 'connecting': return 'text-yellow-600'
      case 'error': return 'text-red-600'
      default: return 'text-gray-400'
    }
  }

  return (
    <Card className="mb-4">
      <CardContent className="py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {getStatusIcon()}
            <span className={`text-sm font-medium ${getStatusColor()}`}>
              {getStatusText()}
            </span>
            {lastUpdate && (
              <div className="flex items-center space-x-1 text-xs text-gray-500">
                <Clock className="h-3 w-3" />
                <span>Updated {formatDistanceToNow(lastUpdate, { addSuffix: true })}</span>
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            {connectionStatus === 'error' && (
              <Button size="sm" variant="outline" onClick={reconnect}>
                <RefreshCw className="h-3 w-3 mr-1" />
                Retry
              </Button>
            )}
            <Badge variant={isConnected ? 'default' : 'secondary'}>
              {isConnected ? 'LIVE' : 'OFFLINE'}
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

const RealtimeStats: React.FC = () => {
  const { data: nodes, isLoading: nodesLoading, error: nodesError } = useNodes()
  const { data: metrics, isLoading: metricsLoading } = useNetworkMetrics('1h')
  const { data: events, isLoading: eventsLoading } = useEvents(10)

  const [updateCount, setUpdateCount] = useState(0)

  // Track data updates
  useEffect(() => {
    setUpdateCount(prev => prev + 1)
  }, [nodes, metrics, events])

  if (nodesError) {
    return (
      <Alert variant="destructive" className="mb-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Failed to load network data: {nodesError.message}
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-gray-600">Data Updates</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{updateCount}</div>
          <p className="text-xs text-gray-500">Total refreshes</p>
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
          <p className="text-xs text-gray-500">Last hour</p>
        </CardContent>
      </Card>
    </div>
  )
}

const DashboardContent: React.FC = () => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Enhanced CSP Dashboard</h1>
            <p className="text-gray-600">Real-time network monitoring and control</p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsSettingsOpen(true)}
          >
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>

        <ConnectionStatus />
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
                <div className="h-96 flex items-center justify-center text-gray-500">
                  Interactive network topology visualization coming soon...
                  <br />
                  <small className="text-xs">Will show real-time node connections and data flow</small>
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
                <div className="h-96 flex items-center justify-center text-gray-500">
                  Real-time performance charts coming soon...
                  <br />
                  <small className="text-xs">Will show live throughput, latency, and system metrics</small>
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
                <div className="h-96 flex items-center justify-center text-gray-500">
                  Live event stream coming soon...
                  <br />
                  <small className="text-xs">Will show real-time system events and alerts</small>
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

const Dashboard: React.FC = () => {
  return (
    <RealtimeProvider>
      <DashboardContent />
    </RealtimeProvider>
  )
}

export default Dashboard