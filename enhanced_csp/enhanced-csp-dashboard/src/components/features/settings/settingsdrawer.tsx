import React from 'react'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '../../ui/sheet'
import { Button } from '../../ui/button'
import { Label } from '../../ui/label'
import { Input } from '../../ui/input'
import { useSettingsStore } from '../../../stores/settingsstore'
import { useAuthStore } from '../../../stores/authstore'
import { useToast } from '../../ui/use-toast'
import { LogOut, Save } from 'lucide-react'

interface SettingsDrawerProps {
  isOpen: boolean
  onClose: () => void
}

export const SettingsDrawer: React.FC<SettingsDrawerProps> = ({
  isOpen,
  onClose
}) => {
  const { settings, updateSettings } = useSettingsStore()
  const { logout, user } = useAuthStore()
  const { toast } = useToast()
  const [localSettings, setLocalSettings] = React.useState(settings)

  React.useEffect(() => {
    setLocalSettings(settings)
  }, [settings])

  const handleSave = () => {
    updateSettings(localSettings)
    toast({
      title: "Settings Saved",
      description: "Your settings have been updated successfully.",
    })
    onClose()
  }

  const handleLogout = () => {
    logout()
    toast({
      title: "Logged Out",
      description: "You have been logged out successfully.",
    })
    onClose()
  }

  const handleInputChange = (field: string, value: string | number | boolean) => {
    setLocalSettings(prev => ({
      ...prev,
      [field]: value
    }))
  }

  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent className="w-[400px] sm:w-[540px]">
        <SheetHeader>
          <SheetTitle>Settings</SheetTitle>
          <SheetDescription>
            Manage your dashboard preferences and account settings.
          </SheetDescription>
        </SheetHeader>
        
        <div className="mt-6 space-y-6">
          {/* User Information */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium">User Information</h3>
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                value={user?.username || ''}
                disabled
                className="bg-gray-50"
              />
            </div>
          </div>

          {/* Dashboard Settings */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Dashboard Preferences</h3>
            
            <div className="space-y-2">
              <Label htmlFor="refresh-interval">Auto Refresh Interval (seconds)</Label>
              <Input
                id="refresh-interval"
                type="number"
                min="5"
                max="300"
                value={localSettings.refreshInterval}
                onChange={(e) => handleInputChange('refreshInterval', parseInt(e.target.value))}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="max-events">Max Events to Display</Label>
              <Input
                id="max-events"
                type="number"
                min="10"
                max="100"
                value={localSettings.maxEvents}
                onChange={(e) => handleInputChange('maxEvents', parseInt(e.target.value))}
              />
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="notifications"
                checked={localSettings.enableNotifications}
                onChange={(e) => handleInputChange('enableNotifications', e.target.checked)}
                className="rounded border-gray-300"
              />
              <Label htmlFor="notifications">Enable Notifications</Label>
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="dark-mode"
                checked={localSettings.darkMode}
                onChange={(e) => handleInputChange('darkMode', e.target.checked)}
                className="rounded border-gray-300"
              />
              <Label htmlFor="dark-mode">Dark Mode</Label>
            </div>
          </div>

          {/* Network Settings */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Network Settings</h3>
            
            <div className="space-y-2">
              <Label htmlFor="api-endpoint">API Endpoint</Label>
              <Input
                id="api-endpoint"
                value={localSettings.apiEndpoint}
                onChange={(e) => handleInputChange('apiEndpoint', e.target.value)}
                placeholder="http://localhost:8000/api"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="websocket-url">WebSocket URL</Label>
              <Input
                id="websocket-url"
                value={localSettings.websocketUrl}
                onChange={(e) => handleInputChange('websocketUrl', e.target.value)}
                placeholder="ws://localhost:8000"
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col space-y-3 pt-6 border-t">
            <Button onClick={handleSave} className="w-full">
              <Save className="w-4 h-4 mr-2" />
              Save Settings
            </Button>
            
            <Button 
              variant="destructive" 
              onClick={handleLogout} 
              className="w-full"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}