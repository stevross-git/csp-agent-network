import { Fragment } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Dialog, Transition } from '@headlessui/react'
import {
  Activity,
  BarChart3,
  Calendar,
  Home,
  Menu,
  Network,
  Settings,
  X,
  LogOut,
  Bell,
  Wifi,
  WifiOff,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { useAuthStore } from '@/stores/authstore'
import { SettingsDrawer } from '@/components/features/settings/settingsdrawer'
import { NotificationCenter } from '@/components/features/notifications/notificationcenter'

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Topology', href: '/topology', icon: Network },
  { name: 'Metrics', href: '/metrics', icon: BarChart3 },
  { name: 'Nodes', href: '/nodes', icon: Activity },
  { name: 'Events', href: '/events', icon: Calendar },
]

interface MainLayoutProps {
  children: React.ReactNode
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
  isConnected: boolean
}

export function MainLayout({
  children,
  sidebarOpen,
  setSidebarOpen,
  isConnected,
}: MainLayoutProps) {
  const location = useLocation()
  const { user, logout } = useAuthStore()

  return (
    <>
      <div>
        <Transition.Root show={sidebarOpen} as={Fragment}>
          <Dialog as="div" className="relative z-50 lg:hidden" onClose={setSidebarOpen}>
            <Transition.Child
              as={Fragment}
              enter="transition-opacity ease-linear duration-300"
              enterFrom="opacity-0"
              enterTo="opacity-100"
              leave="transition-opacity ease-linear duration-300"
              leaveFrom="opacity-100"
              leaveTo="opacity-0"
            >
              <div className="fixed inset-0 bg-gray-900/80" />
            </Transition.Child>

            <div className="fixed inset-0 flex">
              <Transition.Child
                as={Fragment}
                enter="transition ease-in-out duration-300 transform"
                enterFrom="-translate-x-full"
                enterTo="translate-x-0"
                leave="transition ease-in-out duration-300 transform"
                leaveFrom="translate-x-0"
                leaveTo="-translate-x-full"
              >
                <Dialog.Panel className="relative mr-16 flex w-full max-w-xs flex-1">
                  <Transition.Child
                    as={Fragment}
                    enter="ease-in-out duration-300"
                    enterFrom="opacity-0"
                    enterTo="opacity-100"
                    leave="ease-in-out duration-300"
                    leaveFrom="opacity-100"
                    leaveTo="opacity-0"
                  >
                    <div className="absolute left-full top-0 flex w-16 justify-center pt-5">
                      <button
                        type="button"
                        className="-m-2.5 p-2.5"
                        onClick={() => setSidebarOpen(false)}
                      >
                        <span className="sr-only">Close sidebar</span>
                        <X className="h-6 w-6 text-white" aria-hidden="true" />
                      </button>
                    </div>
                  </Transition.Child>
                  <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-background px-6 pb-4">
                    <div className="flex h-16 shrink-0 items-center">
                      <Network className="h-8 w-8 text-primary" />
                      <span className="ml-2 text-lg font-semibold">CSP Network</span>
                    </div>
                    <nav className="flex flex-1 flex-col">
                      <ul role="list" className="flex flex-1 flex-col gap-y-7">
                        <li>
                          <ul role="list" className="-mx-2 space-y-1">
                            {navigation.map((item) => (
                              <li key={item.name}>
                                <Link
                                  to={item.href}
                                  className={cn(
                                    location.pathname === item.href
                                      ? 'bg-accent text-accent-foreground'
                                      : 'text-muted-foreground hover:text-foreground hover:bg-accent',
                                    'group flex gap-x-3 rounded-md p-2 text-sm leading-6 font-semibold'
                                  )}
                                >
                                  <item.icon
                                    className={cn(
                                      location.pathname === item.href
                                        ? 'text-accent-foreground'
                                        : 'text-muted-foreground group-hover:text-foreground',
                                      'h-6 w-6 shrink-0'
                                    )}
                                    aria-hidden="true"
                                  />
                                  {item.name}
                                </Link>
                              </li>
                            ))}
                          </ul>
                        </li>
                      </ul>
                    </nav>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </Dialog>
        </Transition.Root>

        {/* Static sidebar for desktop */}
        <div className="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-72 lg:flex-col">
          <div className="flex grow flex-col gap-y-5 overflow-y-auto border-r border-border bg-background px-6 pb-4">
            <div className="flex h-16 shrink-0 items-center">
              <Network className="h-8 w-8 text-primary" />
              <span className="ml-2 text-lg font-semibold">CSP Network</span>
            </div>
            <nav className="flex flex-1 flex-col">
              <ul role="list" className="flex flex-1 flex-col gap-y-7">
                <li>
                  <ul role="list" className="-mx-2 space-y-1">
                    {navigation.map((item) => (
                      <li key={item.name}>
                        <Link
                          to={item.href}
                          className={cn(
                            location.pathname === item.href
                              ? 'bg-accent text-accent-foreground'
                              : 'text-muted-foreground hover:text-foreground hover:bg-accent',
                            'group flex gap-x-3 rounded-md p-2 text-sm leading-6 font-semibold'
                          )}
                        >
                          <item.icon
                            className={cn(
                              location.pathname === item.href
                                ? 'text-accent-foreground'
                                : 'text-muted-foreground group-hover:text-foreground',
                              'h-6 w-6 shrink-0'
                            )}
                            aria-hidden="true"
                          />
                          {item.name}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </li>
                <li className="mt-auto">
                  <div className="flex items-center gap-x-4 px-2 py-3 text-sm font-semibold leading-6">
                    <div className="flex-1">
                      <p className="text-sm font-medium">{user?.name}</p>
                      <p className="text-xs text-muted-foreground">{user?.email}</p>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => logout()}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      <LogOut className="h-5 w-5" />
                    </Button>
                  </div>
                </li>
              </ul>
            </nav>
          </div>
        </div>

        <div className="lg:pl-72">
          <div className="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-x-4 border-b border-border bg-background px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:px-8">
            <button
              type="button"
              className="-m-2.5 p-2.5 text-foreground lg:hidden"
              onClick={() => setSidebarOpen(true)}
            >
              <span className="sr-only">Open sidebar</span>
              <Menu className="h-6 w-6" aria-hidden="true" />
            </button>

            <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
              <div className="flex flex-1 items-center justify-between">
                <div className="flex items-center gap-x-2">
                  {isConnected ? (
                    <div className="flex items-center text-sm text-success">
                      <Wifi className="h-4 w-4 mr-1" />
                      Connected
                    </div>
                  ) : (
                    <div className="flex items-center text-sm text-destructive">
                      <WifiOff className="h-4 w-4 mr-1" />
                      Disconnected
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-x-2">
                  <NotificationCenter />
                  <SettingsDrawer />
                </div>
              </div>
            </div>
          </div>

          <main className="py-10">
            <div className="px-4 sm:px-6 lg:px-8">{children}</div>
          </main>
        </div>
      </div>
    </>
  )
}