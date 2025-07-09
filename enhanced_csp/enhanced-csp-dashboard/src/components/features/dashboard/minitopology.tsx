import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card'
import { Badge } from '../../ui/badge'

const MiniTopology: React.FC = () => {
  // Mock network topology data
  const nodes = [
    { id: 'node-1', name: 'Central Hub', status: 'active', connections: 8, type: 'hub' },
    { id: 'node-2', name: 'Edge Server A', status: 'active', connections: 4, type: 'edge' },
    { id: 'node-3', name: 'Edge Server B', status: 'active', connections: 3, type: 'edge' },
    { id: 'node-4', name: 'Peer Node 1', status: 'warning', connections: 2, type: 'peer' },
    { id: 'node-5', name: 'Peer Node 2', status: 'active', connections: 1, type: 'peer' },
    { id: 'node-6', name: 'Storage Node', status: 'inactive', connections: 0, type: 'storage' }
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500'
      case 'warning': return 'bg-yellow-500'
      case 'inactive': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active': return <Badge variant="default" className="bg-green-100 text-green-800">Active</Badge>
      case 'warning': return <Badge variant="default" className="bg-yellow-100 text-yellow-800">Warning</Badge>
      case 'inactive': return <Badge variant="destructive">Inactive</Badge>
      default: return <Badge variant="secondary">Unknown</Badge>
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'hub': return '🏢'
      case 'edge': return '🌐'
      case 'peer': return '💻'
      case 'storage': return '💾'
      default: return '⚪'
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Network Topology</CardTitle>
        <CardDescription>
          Overview of connected nodes and their status
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Visual representation */}
          <div className="bg-gray-50 rounded-lg p-4 min-h-32 flex items-center justify-center">
            <div className="grid grid-cols-3 gap-4 w-full max-w-md">
              {nodes.slice(0, 6).map((node, index) => (
                <div key={node.id} className="flex flex-col items-center space-y-1">
                  <div className={`w-8 h-8 rounded-full ${getStatusColor(node.status)} flex items-center justify-center text-white text-xs font-bold relative`}>
                    {getTypeIcon(node.type)}
                    {node.connections > 0 && (
                      <div className="absolute -top-1 -right-1 w-4 h-4 bg-blue-500 rounded-full flex items-center justify-center text-xs text-white">
                        {node.connections}
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-center text-gray-600 max-w-16 truncate">
                    {node.name}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Node list */}
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Node Status</h4>
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {nodes.map((node) => (
                <div key={node.id} className="flex items-center justify-between py-2 px-3 bg-gray-50 rounded-md">
                  <div className="flex items-center space-x-3">
                    <div className={`w-2 h-2 rounded-full ${getStatusColor(node.status)}`}></div>
                    <span className="text-sm font-medium">{node.name}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-500">{node.connections} conn.</span>
                    {getStatusBadge(node.status)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default MiniTopology