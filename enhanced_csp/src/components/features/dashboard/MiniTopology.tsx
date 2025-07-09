import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Maximize2 } from 'lucide-react'
import { useTopology } from '@/hooks/useQueries'
import { getStatusColor } from '@/utils'

export function MiniTopology() {
  const navigate = useNavigate()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { data: topology } = useTopology()

  useEffect(() => {
    if (!canvasRef.current || !topology) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * window.devicePixelRatio
    canvas.height = rect.height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height)

    // Simple force-directed layout
    const nodes = topology.nodes.map((node, i) => ({
      ...node,
      x: Math.random() * (rect.width - 40) + 20,
      y: Math.random() * (rect.height - 40) + 20,
      vx: 0,
      vy: 0,
    }))

    // Simulate for a few iterations
    for (let i = 0; i < 50; i++) {
      // Apply forces
      nodes.forEach((node, i) => {
        // Repulsion between nodes
        nodes.forEach((other, j) => {
          if (i === j) return
          const dx = node.x - other.x
          const dy = node.y - other.y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 100) {
            const force = (100 - dist) / 100 * 2
            node.vx += dx / dist * force
            node.vy += dy / dist * force
          }
        })

        // Attraction to center
        const cx = rect.width / 2
        const cy = rect.height / 2
        node.vx += (cx - node.x) * 0.01
        node.vy += (cy - node.y) * 0.01

        // Damping
        node.vx *= 0.8
        node.vy *= 0.8

        // Update position
        node.x += node.vx
        node.y += node.vy

        // Keep in bounds
        node.x = Math.max(20, Math.min(rect.width - 20, node.x))
        node.y = Math.max(20, Math.min(rect.height - 20, node.y))
      })
    }

    // Draw links
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 1
    topology.links.forEach(link => {
      const source = nodes.find(n => n.id === link.source)
      const target = nodes.find(n => n.id === link.target)
      if (source && target) {
        ctx.beginPath()
        ctx.moveTo(source.x, source.y)
        ctx.lineTo(target.x, target.y)
        ctx.stroke()
      }
    })

    // Draw nodes
    nodes.forEach(node => {
      const color = getStatusColor(node.status).replace('text-', '')
      ctx.fillStyle = color === 'green-500' ? '#10b981' :
                      color === 'yellow-500' ? '#f59e0b' :
                      color === 'red-500' ? '#ef4444' : '#3b82f6'
      
      ctx.beginPath()
      ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI)
      ctx.fill()
      
      // Node label
      ctx.fillStyle = '#6b7280'
      ctx.font = '10px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(node.label, node.x, node.y + 20)
    })
  }, [topology])

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Network Topology</CardTitle>
          <CardDescription>Live network map</CardDescription>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => navigate('/topology')}
        >
          <Maximize2 className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent>
        <canvas
          ref={canvasRef}
          className="w-full h-[250px] bg-muted/20 rounded-lg"
        />
      </CardContent>
    </Card>
  )
}