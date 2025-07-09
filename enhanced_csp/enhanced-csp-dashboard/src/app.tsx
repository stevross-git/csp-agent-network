import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { BrowserRouter } from 'react-router-dom'
import { ErrorBoundary } from 'react-error-boundary'
import { Toaster } from 'sonner'
import { ErrorFallback } from '@/components/ui/error-fallback'
import { useAuthStore } from '@/stores/authstore'
import { useEffect, Suspense, lazy } from 'react'

// Lazy load the main Dashboard component
const Dashboard = lazy(() => import('@/pages/dashboard'))

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
})

function App() {
  const checkAuth = useAuthStore(state => state.checkAuth)

  useEffect(() => {
    checkAuth()
  }, [checkAuth])

  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onError={(error, errorInfo) => {
        console.error('App error:', error, errorInfo)
      }}
    >
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Suspense fallback={<div className="flex items-center justify-center h-screen">Loading...</div>}>
            <Dashboard />
          </Suspense>
          <Toaster />
          {process.env.NODE_ENV === 'development' && <ReactQueryDevtools />}
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  )
}

export default App
