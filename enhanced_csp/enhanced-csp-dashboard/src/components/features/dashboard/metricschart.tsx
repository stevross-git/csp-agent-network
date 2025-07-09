import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../ui/card'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const MetricsChart: React.FC = () => {
  // Mock data for demonstration
  const data = [
    { time: '00:00', throughput: 1.2, latency: 45, cpu: 30 },
    { time: '04:00', throughput: 1.8, latency: 38, cpu: 45 },
    { time: '08:00', throughput: 2.4, latency: 42, cpu: 65 },
    { time: '12:00', throughput: 3.1, latency: 35, cpu: 80 },
    { time: '16:00', throughput: 2.8, latency: 40, cpu: 70 },
    { time: '20:00', throughput: 2.2, latency: 48, cpu: 55 },
    { time: '24:00', throughput: 1.6, latency: 44, cpu: 40 }
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Network Performance</CardTitle>
        <CardDescription>
          Real-time metrics over the last 24 hours
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200" />
              <XAxis 
                dataKey="time" 
                className="text-xs text-gray-500"
                axisLine={false}
                tickLine={false}
              />
              <YAxis 
                className="text-xs text-gray-500"
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Line
                type="monotone"
                dataKey="throughput"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="Throughput (GB/s)"
              />
              <Line
                type="monotone"
                dataKey="latency"
                stroke="#ef4444"
                strokeWidth={2}
                dot={false}
                name="Latency (ms)"
              />
              <Line
                type="monotone"
                dataKey="cpu"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
                name="CPU Usage (%)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        <div className="flex justify-center space-x-6 mt-4">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span className="text-sm text-gray-600">Throughput</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span className="text-sm text-gray-600">Latency</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-600">CPU Usage</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default MetricsChart