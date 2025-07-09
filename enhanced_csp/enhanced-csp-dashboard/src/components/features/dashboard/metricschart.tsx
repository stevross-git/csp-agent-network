import { useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { chartColors } from '@/utils'

// Mock data - replace with real data from API
const generateMockData = (points: number) => {
  const now = Date.now()
  const interval = 5 * 60 * 1000 // 5 minutes
  
  return Array.from({ length: points }, (_, i) => {
    const time = new Date(now - (points - i - 1) * interval)
    return {
      time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      throughput: Math.random() * 100 + 300,
      latency: Math.random() * 20 + 10,
      packetLoss: Math.random() * 0.5,
    }
  })
}

export function MetricsChart() {
  const [timeRange, setTimeRange] = useState('1h')
  
  const dataPoints = {
    '5m': 12,
    '1h': 12,
    '24h': 48,
  }[timeRange] || 12

  const data = generateMockData(dataPoints)

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="rounded-lg border bg-background p-2 shadow-sm">
          <p className="text-sm font-medium">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(2)}
              {entry.name === 'Throughput' && ' Mbps'}
              {entry.name === 'Latency' && ' ms'}
              {entry.name === 'Packet Loss' && '%'}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Network Metrics</CardTitle>
            <CardDescription>Real-time performance indicators</CardDescription>
          </div>
          <Tabs value={timeRange} onValueChange={setTimeRange}>
            <TabsList>
              <TabsTrigger value="5m">5m</TabsTrigger>
              <TabsTrigger value="1h">1h</TabsTrigger>
              <TabsTrigger value="24h">24h</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis 
                dataKey="time" 
                className="text-xs"
                stroke="currentColor"
              />
              <YAxis 
                yAxisId="left"
                className="text-xs"
                stroke="currentColor"
              />
              <YAxis 
                yAxisId="right"
                orientation="right"
                className="text-xs"
                stroke="currentColor"
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend 
                wrapperStyle={{
                  paddingTop: '20px',
                  fontSize: '12px',
                }}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="throughput"
                stroke={chartColors.primary}
                strokeWidth={2}
                dot={false}
                name="Throughput"
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="latency"
                stroke={chartColors.secondary}
                strokeWidth={2}
                dot={false}
                name="Latency"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="packetLoss"
                stroke={chartColors.danger}
                strokeWidth={2}
                dot={false}
                name="Packet Loss"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}