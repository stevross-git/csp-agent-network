import React from 'react'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { ErrorBoundary } from 'react-error-boundary'
import { Toaster } from './components/ui/toaster'
import { ErrorFallback } from './components/ui/error-fallback'
import MainLayout from './components/layout/mainlayout'
import { useAuthStore } from './stores/authstore'
import { useEffect } from 'react'

// Create a client with real-time optimized settings
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 30 * 1000, // 30 seconds - shorter for real-time data
      gcTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: true, // Refetch when window gets focus
      refetchInterval: 30 * 1000, // Auto-refetch every 30 seconds
    },
    mutations: {
      retry: 1,
    },
  },
})

function App() {
  const checkAuth = useAuthStore((state) => state.checkAuth)

  useEffect(() => {
    // Check authentication on app start
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
          <MainLayout />
          <Toaster />
          {process.env.NODE_ENV === 'development' && <ReactQueryDevtools initialIsOpen={false} />}
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  )
}

export default App