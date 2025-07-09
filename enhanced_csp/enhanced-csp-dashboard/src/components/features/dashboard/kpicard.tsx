import { motion } from 'framer-motion'
import { ArrowUp, ArrowDown, LucideIcon } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { formatNumber } from '@/utils'

interface KPICardProps {
  title: string
  value: number
  total?: number
  unit?: string
  icon: LucideIcon
  color: string
  bgColor: string
  trend?: {
    value: number
    isPositive: boolean
  }
  isLoading?: boolean
}

export function KPICard({
  title,
  value,
  total,
  unit,
  icon: Icon,
  color,
  bgColor,
  trend,
  isLoading,
}: KPICardProps) {
  const percentage = total ? (value / total) * 100 : null

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <div className="h-4 w-24 animate-pulse bg-muted rounded" />
          <div className={cn('h-8 w-8 rounded-lg', bgColor, 'animate-pulse')} />
        </CardHeader>
        <CardContent>
          <div className="h-8 w-20 animate-pulse bg-muted rounded mb-2" />
          <div className="h-3 w-16 animate-pulse bg-muted rounded" />
        </CardContent>
      </Card>
    )
  }

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: 'spring', stiffness: 400, damping: 17 }}
    >
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">{title}</CardTitle>
          <div className={cn('rounded-lg p-2', bgColor)}>
            <Icon className={cn('h-4 w-4', color)} />
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {formatNumber(value, unit === '%' ? 2 : 0)}
            {unit && <span className="text-sm font-normal ml-1">{unit}</span>}
            {total && (
              <span className="text-sm font-normal text-muted-foreground">
                {' '}
                / {total}
              </span>
            )}
          </div>
          {percentage !== null && (
            <div className="mt-2">
              <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${percentage}%` }}
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                  className={cn('h-full', 
                    percentage > 80 ? 'bg-red-500' : 
                    percentage > 60 ? 'bg-yellow-500' : 
                    'bg-green-500'
                  )}
                />
              </div>
            </div>
          )}
          {trend && (
            <p className="text-xs text-muted-foreground flex items-center mt-2">
              {trend.isPositive ? (
                <ArrowUp className="h-3 w-3 text-green-500 mr-1" />
              ) : (
                <ArrowDown className="h-3 w-3 text-red-500 mr-1" />
              )}
              <span className={trend.isPositive ? 'text-green-500' : 'text-red-500'}>
                {trend.value}%
              </span>
              <span className="ml-1">from last hour</span>
            </p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}