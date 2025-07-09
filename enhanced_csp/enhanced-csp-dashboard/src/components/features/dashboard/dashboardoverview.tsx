import React from 'react'
import KpiCard from './kpicard'
import MetricsChart from './metricschart'
import MiniTopology from './minitopology'
import RecentEvents from './recentevents'
import { Activity, Users, Server, Zap } from 'lucide-react'
import type { IconType } from '../../../types'

const DashboardOverview: React.FC = () => {
  const kpiData: Array<{
    title: string
    value: string
    change: string
    trend: 'up' | 'down'
    icon: IconType
  }> = [
    {
      title: 'Active Nodes',
      value: '24',
      change: '+12%',
      trend: 'up' as const,
      icon: Server
    },
    {
      title: 'Connected Peers',
      value: '156',
      change: '+8%',
      trend: 'up' as const,
      icon: Users
    },
    {
      title: 'Network Throughput',
      value: '2.4 GB/s',
      change: '+15%',
      trend: 'up' as const,
      icon: Activity
    },
    {
      title: 'Power Efficiency',
      value: '94%',
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

      {/* Charts and Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="lg:col-span-1">
          <MetricsChart />
        </div>
        <div className="lg:col-span-1">
          <MiniTopology />
        </div>
      </div>

      {/* Recent Events */}
      <RecentEvents />
    </div>
  )
}

export default DashboardOverview