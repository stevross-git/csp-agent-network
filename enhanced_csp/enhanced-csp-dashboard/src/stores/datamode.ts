import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface DataModeState {
  isRealDataMode: boolean
  toggleDataMode: () => void
  setDataMode: (isReal: boolean) => void
}

export const useDataModeStore = create<DataModeState>()(
  persist(
    (set, get) => ({
      isRealDataMode: false, // Start with mock data by default

      toggleDataMode: () => {
        set((state) => ({ isRealDataMode: !state.isRealDataMode }))
      },

      setDataMode: (isReal: boolean) => {
        set({ isRealDataMode: isReal })
      }
    }),
    {
      name: 'data-mode-store'
    }
  )
)