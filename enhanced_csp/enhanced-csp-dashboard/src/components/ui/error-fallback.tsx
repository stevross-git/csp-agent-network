interface ErrorFallbackProps {
  error: Error
  resetErrorBoundary: () => void
}

export function ErrorFallback({ error, resetErrorBoundary }: ErrorFallbackProps) {
  return (
    <div className="flex min-h-[400px] flex-col items-center justify-center space-y-4 p-8">
      <div className="flex items-center space-x-2 text-red-500">
        <span className="text-2xl">⚠️</span>
        <h2 className="text-xl font-semibold">Something went wrong</h2>
      </div>
      
      <div className="max-w-md text-center text-sm text-gray-600">
        <p>We encountered an unexpected error. This has been logged and we're working to fix it.</p>
        {process.env.NODE_ENV === 'development' && (
          <details className="mt-4 text-left">
            <summary className="cursor-pointer font-medium">Error Details</summary>
            <pre className="mt-2 whitespace-pre-wrap text-xs text-red-600">
              {error.message}
              {error.stack}
            </pre>
          </details>
        )}
      </div>
      
      <div className="flex space-x-2">
        <button
          onClick={resetErrorBoundary}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          🔄 Try again
        </button>
        
        <button
          onClick={() => window.location.href = '/'}
          className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          🏠 Go to Dashboard
        </button>
      </div>
    </div>
  )
}
