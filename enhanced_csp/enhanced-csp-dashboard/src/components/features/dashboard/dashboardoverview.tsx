import { motion } from 'framer-motion'
import { useNetworkStore } from '@/stores/networkstore'
import { KPICard } from './kpicard'
import { MiniTopology } from './minitopology'
import { RecentEvents } from './recentevents'
import { MetricsChart } from './metricschart'
import {
  Activity,
  AlertTriangle,
  Gauge,
  Network,
  TrendingDown,
  Zap,
} from 'lucide-react'

interface DashboardOverviewProps {
  isLoading: boolean
}

export function DashboardOverview({ isLoading }: DashboardOverviewProps) {
  const { kpis } = useNetworkStore()

  const kpiData = [
    {
      title: 'Active Nodes',
      value: kpis.activeNodes,
      total: kpis.totalNodes,
      icon: Network,
      color: 'text-green-500',
      bgColor: 'bg-green-500/10',
      trend: { value: 5, isPositive: true },
    },
    {
      title: 'Failed Nodes',
      value: kpis.failedNodes,
      total: kpis.totalNodes,
      icon: AlertTriangle,
      color: 'text-red-500',
      bgColor: 'bg-red-500/10',
      trend: { value: 2, isPositive: false },
    },
    {
      title: 'Avg Latency',
      value: kpis.avgLatency,
      unit: 'ms',
      icon: Gauge,
      color: 'text-blue-500',
      bgColor: 'bg-blue-500/10',
      trend: { value: 3, isPositive: false },
    },
    {
      title: 'Packet Loss',
      value: kpis.packetLoss,
      unit: '%',
      icon: TrendingDown,
      color: 'text-yellow-500',
      bgColor: 'bg-yellow-500/10',
      trend: { value: 0.5, isPositive: false },
    },
    {
      title: 'Throughput',
      value: kpis.throughput,
      unit: 'Mbps',
      icon: Zap,
      color: 'text-purple-500',
      bgColor: 'bg-purple-500/10',
      trend: { value: 12, isPositive: true },
    },
  ]

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100,
      },
    },
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Network Dashboard</h1>
        <p className="text-muted-foreground">
          Real-time monitoring of your CSP network infrastructure
        </p>
      </div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid gap-4 md:grid-cols-2 lg:grid-cols-5"
      >
        {kpiData.map((kpi, index) => (
          <motion.div key={kpi.title} variants={itemVariants}>
            <KPICard {...kpi} isLoading={isLoading} />
          </motion.div>
        ))}
      </motion.div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="md:col-span-4"
        >
          <MetricsChart />
        </motion.div>
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
          className="md:col-span-3"
        >
          <MiniTopology />
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <RecentEvents />
      </motion.div>
    </div>
  )
}