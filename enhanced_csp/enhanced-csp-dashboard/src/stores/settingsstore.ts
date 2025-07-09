import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface Settings {
  // UI Preferences
  darkMode: boolean
  enableNotifications: boolean
  refreshInterval: number // in seconds
  maxEvents: number
  
  // Network Configuration
  apiEndpoint: string
  websocketUrl: string
  
  // Dashboard Layout
  showMiniTopology: boolean
  defaultTab: string
  
  // Performance
  enableAutoRefresh: boolean
  chartAnimations: boolean
}

interface SettingsState {
  settings: Settings
  updateSettings: (newSettings: Partial<Settings>) => void
  resetSettings: () => void
  exportSettings: () => string
  importSettings: (settingsJson: string) => boolean
}

const defaultSettings: Settings = {
  // UI Preferences
  darkMode: false,
  enableNotifications: true,
  refreshInterval: 30,
  maxEvents: 50,
  
  // Network Configuration
  apiEndpoint: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  websocketUrl: import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
  
  // Dashboard Layout
  showMiniTopology: true,
  defaultTab: 'overview',
  
  // Performance
  enableAutoRefresh: true,
  chartAnimations: true
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      settings: defaultSettings,

      updateSettings: (newSettings) => {
        set((state) => ({
          settings: {
            ...state.settings,
            ...newSettings
          }
        }))
      },

      resetSettings: () => {
        set({ settings: defaultSettings })
      },

      exportSettings: () => {
        const { settings } = get()
        return JSON.stringify(settings, null, 2)
      },

      importSettings: (settingsJson) => {
        try {
          const importedSettings = JSON.parse(settingsJson)
          
          // Validate that imported settings have the correct structure
          const validatedSettings: Partial<Settings> = {}
          
          Object.keys(defaultSettings).forEach((key) => {
            const settingKey = key as keyof Settings
            if (importedSettings[settingKey] !== undefined) {
              // Type check based on default value type
              const defaultValue = defaultSettings[settingKey]
              const importedValue = importedSettings[settingKey]
              
              if (typeof defaultValue === typeof importedValue) {
                validatedSettings[settingKey] = importedValue
              }
            }
          })
          
          set((state) => ({
            settings: {
              ...state.settings,
              ...validatedSettings
            }
          }))
          
          return true
        } catch (error) {
          console.error('Failed to import settings:', error)
          return false
        }
      }
    }),
    {
      name: 'settings-store',
      version: 1,
      migrate: (persistedState: any, version: number) => {
        if (version === 0) {
          // Migration logic for older versions if needed
          return {
            settings: {
              ...defaultSettings,
              ...persistedState.settings
            }
          }
        }
        return persistedState
      }
    }
  )
)

// Selectors for common settings
export const useTheme = () => useSettingsStore((state) => state.settings.darkMode ? 'dark' : 'light')
export const useRefreshInterval = () => useSettingsStore((state) => state.settings.refreshInterval)
export const useApiEndpoint = () => useSettingsStore((state) => state.settings.apiEndpoint)
export const useWebSocketUrl = () => useSettingsStore((state) => state.settings.websocketUrl)