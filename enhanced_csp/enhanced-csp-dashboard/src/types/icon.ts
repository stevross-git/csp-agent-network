import type { LucideProps } from 'lucide-react'

// Type for any Lucide React icon component
export type IconType = React.ComponentType<LucideProps>

// Common icon props interface
export interface IconProps extends LucideProps {
  className?: string
}

// Re-export LucideProps for convenience
export type { LucideProps }