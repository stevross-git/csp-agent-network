import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card'
import { Badge } from '../../ui/badge'
import { TrendingUp, TrendingDown } from 'lucide-react'
import type { IconType } from '../../../types'

interface KpiCardProps {
  title: string
  value: string
  change: string
  trend: 'up' | 'down'
  icon: IconType
}

const KpiCard: React.FC<KpiCardProps> = ({
  title,
  value,
  change,
  trend,
  icon: Icon
}) => {
  const TrendIcon = trend === 'up' ? TrendingUp : TrendingDown
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
              <TrendIcon className={`h-3 w-3 ${trendColor}`} />
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

export default KpiCard