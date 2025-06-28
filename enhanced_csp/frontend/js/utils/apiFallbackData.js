window.apiFallbackData = {
  '/metrics': {
    active_processes: 0,
    quantum_entanglements: 0,
    blockchain_transactions: 0,
    neural_efficiency: 100,
    security_threats: 0,
    system_uptime: 100,
    cpu_usage: 0,
    memory_usage: 0
  },
  '/processes': {
    processes: [],
    count: 0
  },
  '/api/infrastructure/metrics': {
    cpu: { current: 45, max: 100, unit: '%' },
    memory: { current: 62, max: 100, unit: '%' },
    disk: { current: 78, max: 100, unit: '%' },
    network: { current: 23, max: 100, unit: '%' },
    uptime: { current: 99.5, max: 100, unit: '%' },
    requests: { current: 1250, max: null, unit: '/min' }
  },

  '/api/infrastructure/services': [
    { name: 'Web Server', status: 'running', uptime: '15d 4h 23m', port: 80 },
    { name: 'Database', status: 'running', uptime: '15d 4h 23m', port: 5432 },
    { name: 'Redis Cache', status: 'running', uptime: '15d 4h 23m', port: 6379 },
    { name: 'API Gateway', status: 'running', uptime: '15d 4h 23m', port: 8000 },
    { name: 'Message Queue', status: 'warning', uptime: '2d 1h 15m', port: 5672 }
  ],

  '/api/settings': [
    { key: 'app_name', value: 'Enhanced CSP System', description: 'Application name', widget: 'text', category: 'Application' },
    { key: 'debug', value: false, description: 'Enable debug mode', widget: 'switch', category: 'Application' },
    { key: 'environment', value: 'development', description: 'Application environment', widget: 'select', options: ['development', 'testing', 'staging', 'production'], category: 'Application' },
    { key: 'enable_ai', value: true, description: 'Enable AI features', widget: 'switch', category: 'Features' },
    { key: 'enable_websockets', value: true, description: 'Enable WebSocket support', widget: 'switch', category: 'Features' },
    { key: 'database_host', value: 'localhost', description: 'Database host address', widget: 'text', category: 'Database' },
    { key: 'database_port', value: 5432, description: 'Database port', widget: 'number', category: 'Database' },
    { key: 'database_pool_size', value: 20, description: 'Database connection pool size', widget: 'number', category: 'Database' },
    { key: 'redis_host', value: 'localhost', description: 'Redis host address', widget: 'text', category: 'Cache' },
    { key: 'redis_port', value: 6379, description: 'Redis port', widget: 'number', category: 'Cache' },
    { key: 'ai_max_requests_per_minute', value: 60, description: 'AI API rate limit (requests/min)', widget: 'number', category: 'AI' },
    { key: 'ai_max_daily_cost', value: 100.0, description: 'Maximum daily AI cost limit ($)', widget: 'number', category: 'AI' },
    { key: 'security_max_login_attempts', value: 5, description: 'Maximum login attempts before lockout', widget: 'number', category: 'Security' },
    { key: 'api_rate_limit_requests_per_minute', value: 100, description: 'API rate limit (requests/min/user)', widget: 'number', category: 'API' },
    { key: 'log_level', value: 'INFO', description: 'Application log level', widget: 'select', options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'], category: 'Monitoring' }
  ],

  infrastructureMetricHistory: [45, 47, 46, 48, 50],

  generateStatsGridData() {
    return {
      totalAgents: Math.floor(Math.random() * 50) + 10,
      activeAgents: Math.floor(Math.random() * 30) + 5,
      totalExecutions: Math.floor(Math.random() * 10000) + 1000,
      successRate: Math.random() * 20 + 80,
      averageResponseTime: Math.random() * 500 + 100,
      systemUptime: Math.random() * 720 + 720
    };
  }
};
