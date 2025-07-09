import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { api } from '@/services/api'
import { storage, isTokenExpired, parseJwt } from '@/utils'
import type { AuthState, User, LoginCredentials, AuthResponse } from '@/types'

interface AuthStore extends AuthState {
  login: (credentials: LoginCredentials) => Promise<void>
  logout: () => Promise<void>
  refreshToken: () => Promise<void>
  checkAuth: () => void
  setUser: (user: User | null) => void
  setToken: (token: string | null) => void
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,

      login: async (credentials: LoginCredentials) => {
        try {
          const response = await api.auth.login(credentials)
          const data = response.data as AuthResponse

          set({
            user: data.user,
            token: data.token,
            refreshToken: data.refreshToken,
            isAuthenticated: true,
          })

          storage.set('auth_token', data.token)
          storage.set('refresh_token', data.refreshToken)

          // Schedule token refresh
          const payload = parseJwt(data.token)
          if (payload?.exp) {
            const expiresIn = payload.exp * 1000 - Date.now()
            const refreshTime = expiresIn - 60000 // Refresh 1 minute before expiry
            if (refreshTime > 0) {
              setTimeout(() => get().refreshToken(), refreshTime)
            }
          }
        } catch (error) {
          set({
            user: null,
            token: null,
            refreshToken: null,
            isAuthenticated: false,
          })
          throw error
        }
      },

      logout: async () => {
        try {
          await api.auth.logout()
        } catch (error) {
          console.error('Logout error:', error)
        } finally {
          set({
            user: null,
            token: null,
            refreshToken: null,
            isAuthenticated: false,
          })
          storage.remove('auth_token')
          storage.remove('refresh_token')
        }
      },

      refreshToken: async () => {
        const refreshToken = get().refreshToken
        if (!refreshToken) {
          throw new Error('No refresh token available')
        }

        try {
          const response = await api.auth.refresh(refreshToken)
          const data = response.data as { token: string }

          set({ token: data.token })
          storage.set('auth_token', data.token)

          // Schedule next refresh
          const payload = parseJwt(data.token)
          if (payload?.exp) {
            const expiresIn = payload.exp * 1000 - Date.now()
            const refreshTime = expiresIn - 60000
            if (refreshTime > 0) {
              setTimeout(() => get().refreshToken(), refreshTime)
            }
          }
        } catch (error) {
          // Refresh failed, logout user
          get().logout()
          throw error
        }
      },

      checkAuth: () => {
        const token = storage.get('auth_token', null)
        const refreshToken = storage.get('refresh_token', null)

        if (!token || isTokenExpired(token)) {
          set({
            user: null,
            token: null,
            refreshToken: null,
            isAuthenticated: false,
          })
          return
        }

        const payload = parseJwt(token)
        if (payload) {
          set({
            token,
            refreshToken,
            isAuthenticated: true,
          })

          // Fetch user data
          api.auth.me().then(response => {
            set({ user: response.data as User })
          }).catch(() => {
            get().logout()
          })

          // Schedule token refresh
          if (payload.exp) {
            const expiresIn = payload.exp * 1000 - Date.now()
            const refreshTime = expiresIn - 60000
            if (refreshTime > 0) {
              setTimeout(() => get().refreshToken(), refreshTime)
            }
          }
        }
      },

      setUser: (user: User | null) => set({ user }),
      setToken: (token: string | null) => set({ token }),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        token: state.token,
        refreshToken: state.refreshToken,
      }),
    }
  )
)