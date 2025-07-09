import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Dashboard from '../../pages/dashboard'
import Login from '../../pages/login'
import NotFound from '../../pages/notfound'
import { useAuthStore } from '../../stores/authstore'
import { LoadingSpinner } from '../ui/loading-spinner'

const MainLayout: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuthStore()

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Routes>
        <Route 
          path="/login" 
          element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Login />} 
        />
        <Route 
          path="/dashboard" 
          element={isAuthenticated ? <Dashboard /> : <Navigate to="/login" replace />} 
        />
        <Route 
          path="/" 
          element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} replace />} 
        />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </div>
  )
}

export default MainLayout