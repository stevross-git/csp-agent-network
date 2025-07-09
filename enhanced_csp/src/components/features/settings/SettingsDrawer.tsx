import { Settings } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { useSettingsStore } from '@/stores/settingsStore'

export function SettingsDrawer() {
  const settings = useSettingsStore()

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon">
          <Settings className="h-5 w-5" />
        </Button>
      </SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>Settings</SheetTitle>
          <SheetDescription>
            Configure your dashboard preferences
          </SheetDescription>
        </SheetHeader>
        <div className="mt-6 space-y-6">
          {/* Theme */}
          <div className="space-y-3">
            <Label>Theme</Label>
            <RadioGroup
              value={settings.theme}
              onValueChange={settings.updateTheme}
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="light" id="light" />
                <Label htmlFor="light" className="font-normal">
                  Light
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="dark" id="dark" />
                <Label htmlFor="dark" className="font-normal">
                  Dark
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="system" id="system" />
                <Label htmlFor="system" className="font-normal">
                  System
                </Label>
              </div>
            </RadioGroup>
          </div>

          {/* Refresh Interval */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Refresh Interval</Label>
              <span className="text-sm text-muted-foreground">
                {settings.refreshInterval}s
              </span>
            </div>
            <Slider
              value={[settings.refreshInterval]}
              onValueChange={([value]) => settings.updateRefreshInterval(value)}
              min={1}
              max={60}
              step={1}
            />
          </div>

          {/* Notifications */}
          <div className="space-y-3">
            <Label>Notifications</Label>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="sound" className="font-normal">
                  Sound alerts
                </Label>
                <Switch
                  id="sound"
                  checked={settings.notifications.sound}
                  onCheckedChange={(checked) =>
                    settings.updateNotifications({ sound: checked })
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <Label htmlFor="desktop" className="font-normal">
                  Desktop notifications
                </Label>
                <Switch
                  id="desktop"
                  checked={settings.notifications.desktop}
                  onCheckedChange={(checked) =>
                    settings.updateNotifications({ desktop: checked })
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <Label htmlFor="critical" className="font-normal">
                  Critical alerts only
                </Label>
                <Switch
                  id="critical"
                  checked={settings.notifications.criticalOnly}
                  onCheckedChange={(checked) =>
                    settings.updateNotifications({ criticalOnly: checked })
                  }
                />
              </div>
            </div>
          </div>

          {/* Units */}
          <div className="space-y-3">
            <Label>Metric Units</Label>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="throughput-unit" className="font-normal">
                  Throughput
                </Label>
                <select
                  id="throughput-unit"
                  value={settings.metricUnits.throughput}
                  onChange={(e) =>
                    settings.updateMetricUnits({
                      throughput: e.target.value as 'Mbps' | 'Gbps',
                    })
                  }
                  className="h-8 rounded-md border border-input bg-background px-2 text-sm"
                >
                  <option value="Mbps">Mbps</option>
                  <option value="Gbps">Gbps</option>
                </select>
              </div>
              <div className="flex items-center justify-between">
                <Label htmlFor="storage-unit" className="font-normal">
                  Storage
                </Label>
                <select
                  id="storage-unit"
                  value={settings.metricUnits.storage}
                  onChange={(e) =>
                    settings.updateMetricUnits({
                      storage: e.target.value as 'GB' | 'TB',
                    })
                  }
                  className="h-8 rounded-md border border-input bg-background px-2 text-sm"
                >
                  <option value="GB">GB</option>
                  <option value="TB">TB</option>
                </select>
              </div>
            </div>
          </div>

          <div className="pt-4">
            <Button
              variant="outline"
              className="w-full"
              onClick={() => settings.resetSettings()}
            >
              Reset to defaults
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}