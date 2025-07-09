import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card'
import { Badge } from '../../ui/badge'
import { ScrollArea } from '../../ui/scroll-area'
import { formatDistanceToNow } from 'date-fns'

const RecentEvents: React.FC = () => {
  // Mock events data
  const events = [
    {
      id: '1',
      type: 'connection',
      message: 'New peer connected from 192.168.1.45',
      timestamp: new Date(Date.now() - 2 * 60 * 1000), // 2 minutes ago
      severity: 'info'
    },
    {
      id: '2',
      type: 'performance',
      message: 'Network throughput increased by 15%',
      timestamp: new Date(Date.now() - 8 * 60 * 1000), // 8 minutes ago
      severity: 'success'
    },
    {
      id: '3',
      type: 'security',
      message: 'Suspicious activity detected from node edge-server-c',
      timestamp: new Date(Date.now() - 15 * 60 * 1000), // 15 minutes ago
      severity: 'warning'
    },
    {
      id: '4',
      type: 'system',
      message: 'Backup completed successfully',
      timestamp: new Date(Date.now() - 32 * 60 * 1000), // 32 minutes ago
      severity: 'success'
    },
    {
      id: '5',
      type: 'error',
      message: 'Failed to establish connection with peer 10.0.0.23',
      timestamp: new Date(Date.now() - 45 * 60 * 1000), // 45 minutes ago
      severity: 'error'
    },
    {
      id: '6',
      type: 'maintenance',
      message: 'Scheduled maintenance completed for central hub',
      timestamp: new Date(Date.now() - 68 * 60 * 1000), // 68 minutes ago
      severity: 'info'
    }
  ]

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case 'success':
        return <Badge variant="default" className="bg-green-100 text-green-800">Success</Badge>
      case 'warning':
        return <Badge variant="default" className="bg-yellow-100 text-yellow-800">Warning</Badge>
      case 'error':
        return <Badge variant="destructive">Error</Badge>
      case 'info':
      default:
        return <Badge variant="secondary">Info</Badge>
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'connection': return '🔗'
      case 'performance': return '📈'
      case 'security': return '🛡️'
      case 'system': return '⚙️'
      case 'error': return '❌'
      case 'maintenance': return '🔧'
      default: return 'ℹ️'
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Events</CardTitle>
        <CardDescription>
          Latest network events and system notifications
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-80">
          <div className="space-y-4">
            {events.map((event) => (
              <div key={event.id} className="flex items-start space-x-3 p-3 rounded-lg border border-gray-100 hover:bg-gray-50 transition-colors">
                <div className="text-lg mt-0.5">
                  {getTypeIcon(event.type)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {event.message}
                    </p>
                    {getSeverityBadge(event.severity)}
                  </div>
                  <div className="flex items-center space-x-2 mt-1">
                    <span className="text-xs text-gray-500 capitalize">
                      {event.type}
                    </span>
                    <span className="text-xs text-gray-400">•</span>
                    <span className="text-xs text-gray-500">
                      {formatDistanceToNow(event.timestamp, { addSuffix: true })}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

export default RecentEvents