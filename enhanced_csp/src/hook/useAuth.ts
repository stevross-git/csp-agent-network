import { useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '@/stores/authStore'

const publicPaths = ['/login', '/forgot-password', '/reset-password']

export function useAuth() {
  const navigate = useNavigate()
  const location = useLocation()
  const { isAuthenticated, checkAuth, user, token } = useAuthStore()

  useEffect(() => {
    checkAuth()
  }, [checkAuth])

  useEffect(() => {
    const isPublicPath = publicPaths.some(path => location.pathname.startsWith(path))

    if (!isAuthenticated && !isPublicPath) {
      navigate('/login', { state: { from: location.pathname } })
    } else if (isAuthenticated && location.pathname === '/login') {
      const from = location.state?.from || '/'
      navigate(from)
    }
  }, [isAuthenticated, location.pathname, navigate])

  return {
    isAuthenticated,
    user,
    token,
  }
}

export function useRequireAuth() {
  const { isAuthenticated } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login', { state: { from: location.pathname } })
    }
  }, [isAuthenticated, location.pathname, navigate])

  return isAuthenticated
}

export function usePermission(requiredRole: string) {
  const { user } = useAuthStore()

  const roleHierarchy: Record<string, number> = {
    viewer: 1,
    operator: 2,
    admin: 3,
  }

  const hasPermission = user ? roleHierarchy[user.role] >= roleHierarchy[requiredRole] : false

  return hasPermission
}