import { create } from 'zustand'

interface AuthState {
  user: any | null
  isAuthenticated: boolean
  token: string | null
}

interface AuthStore extends AuthState {
  checkAuth: () => void
  login: (credentials: any) => Promise<void>
  logout: () => void
}

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,
  isAuthenticated: false,
  token: null,

  checkAuth: () => {
    // Basic auth check for testing
    const token = sessionStorage.getItem('csp_auth_token')
    if (token) {
      set({ isAuthenticated: true, token })
    }
  },

  login: async (credentials) => {
    // Mock login for testing
    set({ 
      isAuthenticated: true, 
      token: 'mock-token',
      user: { name: 'Test User' }
    })
  },

  logout: () => {
    sessionStorage.removeItem('csp_auth_token')
    set({ 
      isAuthenticated: false, 
      token: null,
      user: null 
    })
  },
}))
