import { useNavigate } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ArrowRight, AlertCircle, Info, AlertTriangle, XCircle } from 'lucide-react'
import { useNetworkStore } from '@/stores/networkStore'
import { formatRelativeTime, cn } from '@/utils'
import type { NetworkEvent } from '@/types'

const eventIcons = {
  info: Info,
  warning: AlertTriangle,
  error: AlertCircle,
  critical: XCircle,
}

const eventColors = {
  info: 'bg-blue-500/10 text-blue-700 hover:bg-blue-500/20',
  warning: 'bg-yellow-500/10 text-yellow-700 hover:bg-yellow-500/20',
  error: 'bg-red-500/10 text-red-700 hover:bg-red-500/20',
  critical: 'bg-red-600/10 text-red-800 hover:bg-red-600/20',
}

export function RecentEvents() {
  const navigate = useNavigate()
  const events = useNetworkStore(state => state.getRecentEvents(10))

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Recent Events</CardTitle>
          <CardDescription>Latest network activities and alerts</CardDescription>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate('/events')}
          className="text-muted-foreground"
        >
          View all
          <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[300px] pr-4">
          {events.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              <p>No recent events</p>
            </div>
          ) : (
            <div className="space-y-2">
              {events.map((event) => (
                <EventItem key={event.id} event={event} />
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

function EventItem({ event }: { event: NetworkEvent }) {
  const Icon = eventIcons[event.type]
  const colorClass = eventColors[event.type]

  return (
    <div className={cn(
      'flex items-start gap-3 rounded-lg p-3 transition-colors',
      'hover:bg-accent cursor-pointer'
    )}>
      <div className={cn('rounded-full p-2', colorClass)}>
        <Icon className="h-4 w-4" />
      </div>
      <div className="flex-1 space-y-1">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium">{event.title}</p>
          <span className="text-xs text-muted-foreground">
            {formatRelativeTime(event.timestamp)}
          </span>
        </div>
        <p className="text-sm text-muted-foreground">{event.message}</p>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            {event.category}
          </Badge>
          {event.nodeId && (
            <Badge variant="outline" className="text-xs">
              Node: {event.nodeId}
            </Badge>
          )}
        </div>
      </div>
    </div>
  )
}