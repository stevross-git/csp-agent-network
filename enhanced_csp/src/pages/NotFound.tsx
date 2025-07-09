import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Home, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background px-4 py-16 sm:px-6 sm:py-24 md:grid md:place-items-center lg:px-8">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="mx-auto max-w-max"
      >
        <main className="sm:flex">
          <div className="sm:ml-6">
            <div className="sm:border-l sm:border-gray-200 sm:pl-6">
              <div className="flex items-center">
                <AlertTriangle className="h-12 w-12 text-warning mr-4" />
                <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
                  404
                </h1>
              </div>
              <p className="mt-2 text-base text-muted-foreground">
                Page not found
              </p>
            </div>
            <div className="mt-10 flex space-x-3 sm:border-l sm:border-transparent sm:pl-6">
              <Link to="/">
                <Button>
                  <Home className="mr-2 h-4 w-4" />
                  Go back home
                </Button>
              </Link>
            </div>
          </div>
        </main>
      </motion.div>
    </div>
  )
}