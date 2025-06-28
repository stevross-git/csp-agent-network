/**
 * Central fallback data used throughout the frontend.
 * All exports are named for tree-shaking.
 * @module apiFallbackData
 */

/** Fallback metrics returned by the `/metrics` endpoint. */
export const metricsMock = {
  active_processes: 0,
  quantum_entanglements: 0,
  blockchain_transactions: 0,
  neural_efficiency: 100,
  security_threats: 0,
  system_uptime: 100,
  cpu_usage: 0,
  memory_usage: 0,
};

/** Fallback process list returned by `/processes`. */
export const processesMock = { processes: [], count: 0 };

/** Infrastructure metrics used when the API is unreachable. */
export const infrastructureMetricsMock = {
  cpu: { current: 45, max: 100, unit: "%" },
  memory: { current: 62, max: 100, unit: "%" },
  disk: { current: 78, max: 100, unit: "%" },
  network: { current: 23, max: 100, unit: "%" },
  uptime: { current: 99.5, max: 100, unit: "%" },
  requests: { current: 1250, max: null, unit: "/min" },
};

/** Service definitions for infrastructure monitoring. */
export const infrastructureServicesMock = [
  { name: "Web Server", status: "running", uptime: "15d 4h 23m", port: 80 },
  { name: "Database", status: "running", uptime: "15d 4h 23m", port: 5432 },
  { name: "Redis Cache", status: "running", uptime: "15d 4h 23m", port: 6379 },
  { name: "API Gateway", status: "running", uptime: "15d 4h 23m", port: 8000 },
  { name: "Message Queue", status: "warning", uptime: "2d 1h 15m", port: 5672 },
];

/** Application settings defaults. */
export const settingsMock = [
  {
    key: "app_name",
    value: "Enhanced CSP System",
    description: "Application name",
    widget: "text",
    category: "Application",
  },
  {
    key: "debug",
    value: false,
    description: "Enable debug mode",
    widget: "switch",
    category: "Application",
  },
  {
    key: "environment",
    value: "development",
    description: "Application environment",
    widget: "select",
    options: ["development", "testing", "staging", "production"],
    category: "Application",
  },
  {
    key: "enable_ai",
    value: true,
    description: "Enable AI features",
    widget: "switch",
    category: "Features",
  },
  {
    key: "enable_websockets",
    value: true,
    description: "Enable WebSocket support",
    widget: "switch",
    category: "Features",
  },
  {
    key: "database_host",
    value: "localhost",
    description: "Database host address",
    widget: "text",
    category: "Database",
  },
  {
    key: "database_port",
    value: 5432,
    description: "Database port",
    widget: "number",
    category: "Database",
  },
  {
    key: "database_pool_size",
    value: 20,
    description: "Database connection pool size",
    widget: "number",
    category: "Database",
  },
  {
    key: "redis_host",
    value: "localhost",
    description: "Redis host address",
    widget: "text",
    category: "Cache",
  },
  {
    key: "redis_port",
    value: 6379,
    description: "Redis port",
    widget: "number",
    category: "Cache",
  },
  {
    key: "ai_max_requests_per_minute",
    value: 60,
    description: "AI API rate limit (requests/min)",
    widget: "number",
    category: "AI",
  },
  {
    key: "ai_max_daily_cost",
    value: 100.0,
    description: "Maximum daily AI cost limit ($)",
    widget: "number",
    category: "AI",
  },
  {
    key: "security_max_login_attempts",
    value: 5,
    description: "Maximum login attempts before lockout",
    widget: "number",
    category: "Security",
  },
  {
    key: "api_rate_limit_requests_per_minute",
    value: 100,
    description: "API rate limit (requests/min/user)",
    widget: "number",
    category: "API",
  },
  {
    key: "log_level",
    value: "INFO",
    description: "Application log level",
    widget: "select",
    options: ["DEBUG", "INFO", "WARNING", "ERROR"],
    category: "Monitoring",
  },
];

/** Metric history values for graph examples. */
export const infrastructureMetricHistory = [45, 47, 46, 48, 50];

/**
 * Generate random dashboard metrics.
 * @returns {{totalAgents:number,activeAgents:number,totalExecutions:number,successRate:number,averageResponseTime:number,systemUptime:number}}
 */
export function generateStatsGridData() {
  return {
    totalAgents: Math.floor(Math.random() * 50) + 10,
    activeAgents: Math.floor(Math.random() * 30) + 5,
    totalExecutions: Math.floor(Math.random() * 10000) + 1000,
    successRate: Math.random() * 20 + 80,
    averageResponseTime: Math.random() * 500 + 100,
    systemUptime: Math.random() * 720 + 720,
  };
}

/** Recent activity entries for the admin dashboard. */
export const recentActivityMock = [
  {
    time: "14:35",
    action: "Agent Created",
    details: 'New AI agent "Content Analyzer" deployed',
    user: "Admin",
  },
  {
    time: "14:22",
    action: "User Login",
    details: "Sarah Johnson logged in",
    user: "System",
  },
  {
    time: "14:15",
    action: "Task Completed",
    details: "Data processing task #1247 completed",
    user: "Agent-001",
  },
  {
    time: "14:08",
    action: "Alert Resolved",
    details: "Security alert #SA-456 resolved",
    user: "Agent-003",
  },
  {
    time: "13:55",
    action: "System Update",
    details: "CSP engine updated to v2.1.3",
    user: "Admin",
  },
];

/** Example users displayed in admin views. */
export const usersMock = [
  {
    id: 1,
    name: "John Smith",
    email: "john@company.com",
    role: "Admin",
    status: "Active",
    lastLogin: "2024-01-15 09:30",
  },
  {
    id: 2,
    name: "Sarah Johnson",
    email: "sarah@company.com",
    role: "User",
    status: "Active",
    lastLogin: "2024-01-15 14:22",
  },
  {
    id: 3,
    name: "Mike Chen",
    email: "mike@company.com",
    role: "Developer",
    status: "Active",
    lastLogin: "2024-01-15 11:45",
  },
  {
    id: 4,
    name: "Emily Davis",
    email: "emily@company.com",
    role: "Manager",
    status: "Inactive",
    lastLogin: "2024-01-10 16:30",
  },
];

/** Monitoring service examples shown when no API is available. */
export const monitoringDataMock = [
  {
    service: "CSP Engine",
    status: "Running",
    cpu: "23%",
    memory: "45%",
    uptime: "15d 4h",
  },
  {
    service: "AI Agent Manager",
    status: "Running",
    cpu: "12%",
    memory: "32%",
    uptime: "15d 4h",
  },
  {
    service: "Database",
    status: "Running",
    cpu: "8%",
    memory: "67%",
    uptime: "15d 4h",
  },
  {
    service: "Web Server",
    status: "Running",
    cpu: "5%",
    memory: "28%",
    uptime: "15d 4h",
  },
  {
    service: "Message Queue",
    status: "Warning",
    cpu: "45%",
    memory: "78%",
    uptime: "15d 4h",
  },
];

/** Model statistics used in dashboards. */
export const modelsDataMock = [
  {
    model: "GPT-4 Turbo",
    type: "LLM",
    status: "Active",
    requests: "1,247",
    responseTime: "1.8s",
    successRate: "99.2%",
  },
  {
    model: "Claude-3 Sonnet",
    type: "LLM",
    status: "Active",
    requests: "892",
    responseTime: "2.1s",
    successRate: "98.8%",
  },
  {
    model: "Gemini Pro",
    type: "Multimodal",
    status: "Active",
    requests: "634",
    responseTime: "1.4s",
    successRate: "97.5%",
  },
  {
    model: "Text-Embedding-Ada-002",
    type: "Embedding",
    status: "Active",
    requests: "3,421",
    responseTime: "0.3s",
    successRate: "99.9%",
  },
];

/** Log entries displayed in admin pages. */
export const logsDataMock = [
  {
    timestamp: "2024-01-15 14:35:22",
    level: "INFO",
    component: "AgentManager",
    message: 'Agent "Content Analyzer" started successfully',
  },
  {
    timestamp: "2024-01-15 14:35:18",
    level: "DEBUG",
    component: "CSPEngine",
    message: "Process synchronization completed",
  },
  {
    timestamp: "2024-01-15 14:35:15",
    level: "WARN",
    component: "MessageQueue",
    message: "Queue capacity at 78%, consider scaling",
  },
  {
    timestamp: "2024-01-15 14:35:10",
    level: "INFO",
    component: "Security",
    message: "Authentication successful for user sarah@company.com",
  },
  {
    timestamp: "2024-01-15 14:35:05",
    level: "ERROR",
    component: "Database",
    message: "Connection timeout resolved after retry",
  },
];

/** Default agent configurations for demos. */
export const defaultAgentsMock = [
  {
    id: "agent-001",
    name: "DataAnalyzer Pro",
    type: "autonomous",
    description:
      "Advanced data analysis and pattern recognition for business intelligence",
    status: "active",
    model: "gpt-4",
    priority: "high",
    capabilities: ["data_analysis", "text_processing", "api_integration"],
    tasksCompleted: 247,
    uptime: "99.8%",
    lastActivity: "2 minutes ago",
    executionMode: "continuous",
    communicationChannel: "internal_api",
    maxTasks: 10,
    timeout: 300,
    autoRestart: true,
    created: "2024-01-15",
    logs: [
      {
        time: "14:35:22",
        level: "SUCCESS",
        message: "Data analysis pipeline completed successfully",
      },
      {
        time: "14:32:18",
        level: "INFO",
        message: "Processing new dataset: sales_q4_2024.csv",
      },
      {
        time: "14:28:45",
        level: "INFO",
        message: "Generated insights report for marketing team",
      },
    ],
  },
  {
    id: "agent-002",
    name: "Customer Support Assistant",
    type: "collaborative",
    description:
      "AI-powered customer service with natural language understanding",
    status: "active",
    model: "claude-3",
    priority: "critical",
    capabilities: ["text_processing", "api_integration"],
    tasksCompleted: 1832,
    uptime: "99.9%",
    lastActivity: "30 seconds ago",
    executionMode: "event_driven",
    communicationChannel: "webhooks",
    maxTasks: 25,
    timeout: 180,
    autoRestart: true,
    created: "2024-01-10",
    logs: [
      {
        time: "14:36:40",
        level: "INFO",
        message: "Resolved customer ticket #CS-2024-0892",
      },
      {
        time: "14:34:15",
        level: "SUCCESS",
        message: "Escalated complex issue to human agent",
      },
    ],
  },
  {
    id: "agent-003",
    name: "CodeReview Specialist",
    type: "specialized",
    description: "Automated code review and security vulnerability detection",
    status: "active",
    model: "gpt-4",
    priority: "high",
    capabilities: ["code_generation", "text_processing"],
    tasksCompleted: 156,
    uptime: "98.7%",
    lastActivity: "5 minutes ago",
    executionMode: "on_demand",
    communicationChannel: "message_queue",
    maxTasks: 5,
    timeout: 600,
    autoRestart: true,
    created: "2024-01-20",
    logs: [
      {
        time: "14:30:55",
        level: "WARNING",
        message: "Security vulnerability detected in auth.js",
      },
      {
        time: "14:27:12",
        level: "SUCCESS",
        message: "Code review completed for PR #247",
      },
    ],
  },
  {
    id: "agent-004",
    name: "System Health Monitor",
    type: "monitoring",
    description: "Continuous system monitoring and performance tracking",
    status: "active",
    model: "gemini",
    priority: "critical",
    capabilities: ["monitoring", "api_integration"],
    tasksCompleted: 892,
    uptime: "100%",
    lastActivity: "15 seconds ago",
    executionMode: "continuous",
    communicationChannel: "internal_api",
    maxTasks: 3,
    timeout: 60,
    autoRestart: true,
    created: "2024-01-05",
    logs: [
      {
        time: "14:37:00",
        level: "INFO",
        message: "System health check completed - all services running",
      },
      {
        time: "14:36:30",
        level: "WARNING",
        message: "CPU usage spike detected: 87%",
      },
    ],
  },
  {
    id: "agent-005",
    name: "Content Moderation AI",
    type: "specialized",
    description: "Real-time content filtering and safety compliance monitoring",
    status: "paused",
    model: "claude-3",
    priority: "normal",
    capabilities: ["text_processing", "image_processing"],
    tasksCompleted: 3421,
    uptime: "97.2%",
    lastActivity: "1 hour ago",
    executionMode: "event_driven",
    communicationChannel: "webhooks",
    maxTasks: 15,
    timeout: 120,
    autoRestart: false,
    created: "2024-01-12",
    logs: [
      {
        time: "13:45:22",
        level: "INFO",
        message: "Agent paused for routine maintenance",
      },
      {
        time: "13:42:15",
        level: "SUCCESS",
        message: "Flagged inappropriate content in user submission",
      },
    ],
  },
  {
    id: "agent-006",
    name: "WebScraper Intelligence",
    type: "autonomous",
    description:
      "Intelligent web scraping and data extraction with respect for robots.txt",
    status: "active",
    model: "gpt-4",
    priority: "normal",
    capabilities: ["data_analysis", "api_integration"],
    tasksCompleted: 678,
    uptime: "96.5%",
    lastActivity: "8 minutes ago",
    executionMode: "scheduled",
    communicationChannel: "database",
    maxTasks: 8,
    timeout: 900,
    autoRestart: true,
    created: "2024-01-18",
    logs: [
      {
        time: "14:28:45",
        level: "SUCCESS",
        message: "Scraped 1,247 product listings from e-commerce sites",
      },
      {
        time: "14:25:12",
        level: "INFO",
        message: "Respecting rate limits: 2 requests per second",
      },
    ],
  },
  {
    id: "agent-007",
    name: "API Integration Hub",
    type: "collaborative",
    description: "Multi-service API orchestration and data synchronization",
    status: "error",
    model: "gpt-4",
    priority: "high",
    capabilities: ["api_integration", "data_analysis"],
    tasksCompleted: 445,
    uptime: "89.3%",
    lastActivity: "12 minutes ago",
    executionMode: "continuous",
    communicationChannel: "message_queue",
    maxTasks: 12,
    timeout: 450,
    autoRestart: true,
    created: "2024-01-22",
    logs: [
      {
        time: "14:25:18",
        level: "ERROR",
        message: "Connection timeout to payment-api.service.com",
      },
      {
        time: "14:22:45",
        level: "WARNING",
        message: "API rate limit reached for external-data-provider",
      },
    ],
  },
  {
    id: "agent-008",
    name: "Document Intelligence",
    type: "specialized",
    description:
      "Advanced document processing, OCR, and legal compliance analysis",
    status: "paused",
    model: "claude-3",
    priority: "normal",
    capabilities: ["text_processing", "image_processing", "data_analysis"],
    tasksCompleted: 234,
    uptime: "95.8%",
    lastActivity: "4 minutes ago",
    executionMode: "on_demand",
    communicationChannel: "internal_api",
    maxTasks: 6,
    timeout: 800,
    autoRestart: true,
    created: "2024-01-25",
    logs: [
      {
        time: "14:31:40",
        level: "SUCCESS",
        message: "Legal document analysis completed",
      },
      {
        time: "14:28:22",
        level: "INFO",
        message: "Processing contract for compliance review",
      },
    ],
  },
];

/** Simplified agent list for AgentManager tests. */
export const managerSampleAgents = [
  {
    id: "agent-001",
    name: "Security Monitor",
    type: "security",
    status: "active",
    description: "Monitors system security and detects threats",
    priority: "high",
    created: new Date("2024-01-15"),
    lastActivity: new Date(),
    cpuUsage: 15,
    memoryUsage: 32,
    tasksCompleted: 1247,
    uptime: "5d 12h 30m",
    autoRestart: true,
    logging: true,
  },
  {
    id: "agent-002",
    name: "Performance Optimizer",
    type: "performance",
    status: "active",
    description: "Optimizes system performance and resource usage",
    priority: "medium",
    created: new Date("2024-02-01"),
    lastActivity: new Date(Date.now() - 300000),
    cpuUsage: 8,
    memoryUsage: 28,
    tasksCompleted: 892,
    uptime: "3d 8h 15m",
    autoRestart: true,
    logging: false,
  },
  {
    id: "agent-003",
    name: "Backup Coordinator",
    type: "backup",
    status: "inactive",
    description: "Manages automated backup processes",
    priority: "low",
    created: new Date("2024-01-20"),
    lastActivity: new Date(Date.now() - 3600000),
    cpuUsage: 0,
    memoryUsage: 0,
    tasksCompleted: 156,
    uptime: "0h 0m",
    autoRestart: false,
    logging: true,
  },
  {
    id: "agent-004",
    name: "Health Monitor",
    type: "monitoring",
    status: "active",
    description: "Monitors system health and performance metrics",
    priority: "high",
    created: new Date("2024-01-10"),
    lastActivity: new Date(Date.now() - 60000),
    cpuUsage: 12,
    memoryUsage: 24,
    tasksCompleted: 2341,
    uptime: "7d 14h 22m",
    autoRestart: true,
    logging: true,
  },
];

/** Default user objects used by the UserManager. */
export const defaultUsersMock = [
  {
    id: "1",
    full_name: "John Smith",
    email: "john@company.com",
    roles: ["admin"],
    is_active: true,
    last_login: "2024-01-15T09:30:00Z",
    created_at: "2024-01-01T00:00:00Z",
  },
  {
    id: "2",
    full_name: "Sarah Johnson",
    email: "sarah@company.com",
    roles: ["user"],
    is_active: true,
    last_login: "2024-01-15T14:22:00Z",
    created_at: "2024-01-02T00:00:00Z",
  },
  {
    id: "3",
    full_name: "Mike Chen",
    email: "mike@company.com",
    roles: ["developer"],
    is_active: false,
    last_login: "2024-01-10T16:30:00Z",
    created_at: "2024-01-03T00:00:00Z",
  },
];

/** Default role definitions. */
export const defaultRolesMock = [
  {
    id: "admin",
    name: "Administrator",
    permissions: ["manage_users", "manage_system"],
  },
  { id: "user", name: "User", permissions: ["view_system"] },
];

/** Fallback alert objects returned when monitoring APIs are offline. */
export const fallbackAlertsMock = [
  {
    id: "alert-1",
    name: "High Memory Usage",
    severity: "critical",
    status: "firing",
    message: "Memory usage above 90% on csp_postgres container",
    source: "csp_postgres",
    timestamp: new Date(Date.now() - 300000),
    labels: { container: "csp_postgres", severity: "critical" },
  },
  {
    id: "alert-2",
    name: "Redis Connection Slow",
    severity: "warning",
    status: "firing",
    message: "Redis response time exceeding 100ms",
    source: "csp_redis",
    timestamp: new Date(Date.now() - 600000),
    labels: { service: "redis", severity: "warning" },
  },
  {
    id: "alert-3",
    name: "Disk Space Low",
    severity: "warning",
    status: "resolved",
    message: "Disk usage was above 85%",
    source: "host",
    timestamp: new Date(Date.now() - 3600000),
    endsAt: new Date(Date.now() - 1800000),
    labels: { severity: "warning", filesystem: "/var/lib/docker" },
  },
];

/** Fallback incidents returned when external services are not reachable. */
export const fallbackIncidentsMock = [
  {
    id: "inc-001",
    title: "Database Performance Degradation",
    description: "PostgreSQL queries running slower than normal",
    severity: "high",
    status: "investigating",
    assignee: "System Admin",
    created: new Date(Date.now() - 1800000),
    updated: new Date(Date.now() - 300000),
    affectedServices: ["csp_postgres", "csp_api"],
    timeline: [
      {
        time: new Date(Date.now() - 1800000),
        action: "Incident created",
        user: "System",
      },
      {
        time: new Date(Date.now() - 1500000),
        action: "Investigation started",
        user: "Admin",
      },
      {
        time: new Date(Date.now() - 300000),
        action: "Root cause identified",
        user: "Admin",
      },
    ],
  },
];

/** Aggregate object for legacy code paths. */
export const apiFallbackData = {
  "/metrics": metricsMock,
  "/processes": processesMock,
  "/api/infrastructure/metrics": infrastructureMetricsMock,
  "/api/infrastructure/services": infrastructureServicesMock,
  "/api/settings": settingsMock,
  infrastructureMetricHistory,
  generateStatsGridData,
};
