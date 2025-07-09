import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { apiService } from '../services/api'
import type { User } from '../types'

interface AuthState {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  login: (username: string, password: string) => Promise<void>
  logout: () => void
  checkAuth: () => Promise<void>
  clearError: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (username: string, password: string) => {
        set({ isLoading: true, error: null })
        
        try {
          // Always use mock authentication in development
          const mockUser: User = {
            id: '1',
            username,
            email: `${username}@example.com`,
            role: 'admin',
            createdAt: new Date().toISOString(),
            lastLogin: new Date().toISOString()
          }
          
          const mockToken = 'mock-jwt-token-' + Date.now()
          
          // Store token in localStorage
          localStorage.setItem('auth_token', mockToken)
          
          set({
            user: mockUser,
            isAuthenticated: true,
            isLoading: false,
            error: null
          })
          
          console.log('Mock login successful for user:', username)
        } catch (error: any) {
          const errorMessage = 'Login failed - please try again'
          set({
            error: errorMessage,
            isLoading: false,
            isAuthenticated: false,
            user: null
          })
          throw error
        }
      },

      logout: () => {
        localStorage.removeItem('auth_token')
        set({
          user: null,
          isAuthenticated: false,
          error: null
        })
      },

      checkAuth: async () => {
        const token = localStorage.getItem('auth_token')
        
        if (!token) {
          set({ isAuthenticated: false, user: null, isLoading: false })
          return
        }

        set({ isLoading: true })
        
        try {
          // In development, if we have a token, assume we're authenticated
          if (token.startsWith('mock-jwt-token-')) {
            const mockUser: User = {
              id: '1',
              username: 'admin',
              email: 'admin@example.com',
              role: 'admin',
              createdAt: new Date().toISOString(),
              lastLogin: new Date().toISOString()
            }
            
            set({
              user: mockUser,
              isAuthenticated: true,
              isLoading: false,
              error: null
            })
          } else {
            // For production, we would check with the API
            const response = await apiService.getCurrentUser()
            const user = response.data
            
            set({
              user,
              isAuthenticated: true,
              isLoading: false,
              error: null
            })
          }
        } catch (error) {
          localStorage.removeItem('auth_token')
          set({
            user: null,
            isAuthenticated: false,
            isLoading: false,
            error: null
          })
        }
      },

      clearError: () => {
        set({ error: null })
      }
    }),
    {
      name: 'auth-store',
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated
      })
    }
  )
)