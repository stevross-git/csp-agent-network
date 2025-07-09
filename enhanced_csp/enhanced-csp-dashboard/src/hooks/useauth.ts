import { useCallback, useEffect } from 'react'
import { useAuthStore } from '../stores/authstore'
import { useToast } from '../components/ui/use-toast'

export const useAuth = () => {
  const {
    user,
    isAuthenticated,
    isLoading,
    error,
    login,
    logout,
    checkAuth,
    clearError
  } = useAuthStore()
  
  const { toast } = useToast()

  // Auto-check authentication on mount
  useEffect(() => {
    checkAuth()
  }, [checkAuth])

  // Show error toast when authentication error occurs
  useEffect(() => {
    if (error) {
      toast({
        title: 'Authentication Error',
        description: error,
        variant: 'destructive'
      })
    }
  }, [error, toast])

  const handleLogin = useCallback(async (username: string, password: string) => {
    try {
      await login(username, password)
      toast({
        title: 'Welcome!',
        description: 'You have been logged in successfully.'
      })
      return true
    } catch (err) {
      // Error is already handled by the store and shown via useEffect above
      return false
    }
  }, [login, toast])

  const handleLogout = useCallback(() => {
    logout()
    toast({
      title: 'Goodbye!',
      description: 'You have been logged out successfully.'
    })
  }, [logout, toast])

  const dismissError = useCallback(() => {
    clearError()
  }, [clearError])

  return {
    // State
    user,
    isAuthenticated,
    isLoading,
    error,
    
    // Actions
    login: handleLogin,
    logout: handleLogout,
    checkAuth,
    dismissError,
    
    // Computed values
    isAdmin: user?.role === 'admin',
    canEdit: user?.role === 'admin' || user?.role === 'user',
    canView: !!user
  }
}