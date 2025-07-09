import * as React from "react"
import { toast as sonnerToast } from "sonner"

export interface ToastProps {
  title?: string
  description?: string
  action?: React.ReactNode
  variant?: "default" | "destructive"
}

export const useToast = () => {
  const toast = React.useCallback(({ title, description, variant, ...props }: ToastProps) => {
    if (variant === "destructive") {
      return sonnerToast.error(title || description, {
        description: title ? description : undefined,
        ...props
      })
    }
    
    return sonnerToast(title || description, {
      description: title ? description : undefined,
      ...props
    })
  }, [])

  return {
    toast,
    success: (message: string, options?: any) => sonnerToast.success(message, options),
    error: (message: string, options?: any) => sonnerToast.error(message, options),
    warning: (message: string, options?: any) => sonnerToast.warning(message, options),
    info: (message: string, options?: any) => sonnerToast.info(message, options),
    dismiss: (toastId?: string | number) => sonnerToast.dismiss(toastId),
  }
}