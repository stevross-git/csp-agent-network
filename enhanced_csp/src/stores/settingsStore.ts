import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { AppSettings } from '@/types'

interface SettingsStore extends AppSettings {
  updateTheme: (theme: AppSettings['theme']) => void
  updateRefreshInterval: (interval: number) => void
  updateMetricUnits: (units: Partial<AppSettings['metricUnits']>) => void
  updateNotifications: (notifications: Partial<AppSettings['notifications']>) => void
  updateLanguage: (language: string) => void
  updateTimeZone: (timeZone: string) => void
  resetSettings: () => void
}

const defaultSettings: AppSettings = {
  theme: 'system',
  refreshInterval: 5, // seconds
  metricUnits: {
    throughput: 'Mbps',
    storage: 'GB',
  },
  notifications: {
    sound: true,
    desktop: true,
    criticalOnly: false,
  },
  language: 'en',
  timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
}

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      ...defaultSettings,

      updateTheme: (theme) => {
        set({ theme })
        
        // Apply theme to document
        const root = window.document.documentElement
        root.classList.remove('light', 'dark')
        
        if (theme === 'system') {
          const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
          root.classList.add(systemTheme)
        } else {
          root.classList.add(theme)
        }
      },

      updateRefreshInterval: (refreshInterval) => {
        if (refreshInterval >= 1 && refreshInterval <= 60) {
          set({ refreshInterval })
        }
      },

      updateMetricUnits: (units) => {
        set((state) => ({
          metricUnits: {
            ...state.metricUnits,
            ...units,
          },
        }))
      },

      updateNotifications: (notifications) => {
        set((state) => ({
          notifications: {
            ...state.notifications,
            ...notifications,
          },
        }))

        // Request notification permission if enabled
        if (notifications.desktop && 'Notification' in window && Notification.permission === 'default') {
          Notification.requestPermission()
        }
      },

      updateLanguage: (language) => {
        set({ language })
        // TODO: Integrate with i18n
      },

      updateTimeZone: (timeZone) => {
        set({ timeZone })
      },

      resetSettings: () => {
        set(defaultSettings)
      },
    }),
    {
      name: 'app-settings',
      onRehydrateStorage: () => (state) => {
        // Apply theme on app load
        if (state?.theme) {
          const root = window.document.documentElement
          root.classList.remove('light', 'dark')
          
          if (state.theme === 'system') {
            const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
            root.classList.add(systemTheme)
          } else {
            root.classList.add(state.theme)
          }
        }
      },
    }
  )
)

// Listen for system theme changes
if (typeof window !== 'undefined') {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    const settings = useSettingsStore.getState()
    if (settings.theme === 'system') {
      const root = window.document.documentElement
      root.classList.remove('light', 'dark')
      root.classList.add(e.matches ? 'dark' : 'light')
    }
  })
}