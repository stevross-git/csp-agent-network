// frontend/js/utils/apiFallbackData.js
/**
 * Central API Fallback Data Module
 * ================================
 * 
 * This module contains all mock/fallback data for the Enhanced CSP System.
 * Used when API endpoints are unavailable or during development/testing.
 * 
 * @author Enhanced CSP System
 * @version 1.0.0
 */

/**
 * System settings mock data
 * Used by: systemManager.js
 */
export const settingsMock = [
  { 
    key: 'app_name', 
    value: 'Enhanced CSP System', 
    description: 'Application name', 
    widget: 'text', 
    category: 'Application' 
  },
  { 
    key: 'debug', 
    value: false, 
    description: 'Enable debug mode', 
    widget: 'switch', 
    category: 'Application' 
  },
  { 
    key: 'environment', 
    value: 'development', 
    description: 'Application environment', 
    widget: 'select', 
    options: ['development', 'testing', 'staging', 'production'], 
    category: 'Application' 
  },
  { 
    key: 'enable_ai', 
    value: true, 
    description: 'Enable AI features', 
    widget: 'switch', 
    category: 'Features' 
  },
  { 
    key: 'ai_model', 
    value: 'gpt-3.5-turbo', 
    description: 'Default AI model', 
    widget: 'select', 
    options: ['gpt-3.5-turbo', 'gpt-4', 'claude-3'], 
    category: 'AI' 
  },
  { 
    key: 'max_agents', 
    value: 50, 
    description: 'Maximum number of agents', 
    widget: 'number', 
    category: 'Limits' 
  },
  { 
    key: 'session_timeout', 
    value: 3600, 
    description: 'Session timeout (seconds)', 
    widget: 'number', 
    category: 'Security' 
  },
  { 
    key: 'enable_websocket', 
    value: true, 
    description: 'Enable WebSocket connections', 
    widget: 'switch', 
    category: 'Features' 
  },
  { 
    key: 'log_level', 
    value: 'INFO', 
    description: 'Application log level', 
    widget: 'select', 
    options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
    category: 'Logging' 
  },
  { 
    key: 'backup_enabled', 
    value: true, 
    description: 'Enable automatic backups', 
    widget: 'switch', 
    category: 'Backup' 
  }
];

/**
 * System statistics mock data generation
 * Used by: StatsGrid.js
 */
export const generateStatsMock = () => ({
  totalAgents: Math.floor(Math.random() * 50) + 10,
  activeAgents: Math.floor(Math.random() * 30) + 5,
  totalExecutions: Math.floor(Math.random() * 10000) + 1000,
  successRate: Math.random() * 20 + 80, // 80-100%
  averageResponseTime: Math.random() * 500 + 100, // 100-600ms
  systemUptime: Math.random() * 720 + 720 // 12-36 hours in minutes
});

/**
 * System metrics mock data
 * Used by: web_dashboard_ui.html, infrastructureManager.js
 */
export const metricsMock = {
  system: {
    cpu_usage: 45.2,
    memory_usage: 67.8,
    disk_usage: 23.4,
    uptime: 86400,
    temperature: 58.5,
    network_throughput: 125.7
  },
  processes: {
    active: 24,
    total: 50,
    success_rate: 0.98,
    failed: 1,
    pending: 3
  },
  quantum: {
    entanglements: 8,
    average_fidelity: 0.94,
    quantum_volume: 256,
    coherence_time: 150.3
  },
  blockchain: {
    transactions: 1847,
    gas_efficiency: 0.23,
    consensus_time: 2.3,
    block_height: 125847
  },
  ai: {
    models_loaded: 5,
    inference_rate: 32.1,
    accuracy: 0.96,
    training_jobs: 2
  },
  security: {
    threat_level: 'LOW',
    blocked_attempts: 12,
    active_sessions: 45,
    certificate_status: 'VALID'
  }
};

/**
 * Infrastructure services mock data
 * Used by: infrastructureManager.js
 */
export const servicesMock = [
  {
    name: 'Web Server',
    status: 'running',
    uptime: '1d 2h 15m',
    port: 8000,
    health: 'healthy',
    cpu: 12.5,
    memory: 256.8,
    connections: 127
  },
  {
    name: 'Database',
    status: 'running',
    uptime: '1d 2h 15m',
    port: 5432,
    health: 'healthy',
    cpu: 8.3,
    memory: 512.0,
    connections: 45
  },
  {
    name: 'Redis Cache',
    status: 'running',
    uptime: '1d 2h 15m',
    port: 6379,
    health: 'healthy',
    cpu: 2.1,
    memory: 128.5,
    connections: 89
  },
  {
    name: 'AI Engine',
    status: 'running',
    uptime: '23h 47m',
    port: 8001,
    health: 'healthy',
    cpu: 45.7,
    memory: 1024.0,
    connections: 12
  },
  {
    name: 'WebSocket Server',
    status: 'running',
    uptime: '1d 2h 15m',
    port: 8080,
    health: 'healthy',
    cpu: 5.2,
    memory: 64.2,
    connections: 234
  }
];

/**
 * Users mock data
 * Used by: userManager.js, admin-modals.js
 */
export const usersMock = [
  {
    id: 'usr_001',
    username: 'admin',
    email: 'admin@enhanced-csp.com',
    full_name: 'System Administrator',
    role: 'admin',
    status: 'active',
    last_login: '2025-06-28T10:30:00Z',
    created_at: '2025-01-01T00:00:00Z',
    permissions: ['read', 'write', 'delete', 'admin']
  },
  {
    id: 'usr_002',
    username: 'developer',
    email: 'dev@enhanced-csp.com',
    full_name: 'Lead Developer',
    role: 'developer',
    status: 'active',
    last_login: '2025-06-28T09:15:00Z',
    created_at: '2025-01-15T00:00:00Z',
    permissions: ['read', 'write']
  },
  {
    id: 'usr_003',
    username: 'analyst',
    email: 'analyst@enhanced-csp.com',
    full_name: 'Data Analyst',
    role: 'analyst',
    status: 'active',
    last_login: '2025-06-27T16:45:00Z',
    created_at: '2025-02-01T00:00:00Z',
    permissions: ['read']
  }
];

/**
 * AI Agents mock data
 * Used by: agentManager.js, AgentService.js
 */
export const agentsMock = [
  {
    id: 'agent_001',
    name: 'Data Processor Alpha',
    type: 'data_processor',
    status: 'active',
    model: 'gpt-3.5-turbo',
    version: '1.2.3',
    capabilities: ['text_processing', 'data_analysis', 'report_generation'],
    performance: {
      success_rate: 0.98,
      avg_response_time: 245,
      total_executions: 1547
    },
    created_at: '2025-06-01T00:00:00Z',
    last_execution: '2025-06-28T10:25:00Z'
  },
  {
    id: 'agent_002',
    name: 'Security Monitor Beta',
    type: 'security_monitor',
    status: 'active',
    model: 'claude-3',
    version: '2.1.0',
    capabilities: ['threat_detection', 'log_analysis', 'incident_response'],
    performance: {
      success_rate: 0.995,
      avg_response_time: 123,
      total_executions: 3421
    },
    created_at: '2025-05-15T00:00:00Z',
    last_execution: '2025-06-28T10:28:00Z'
  },
  {
    id: 'agent_003',
    name: 'Neural Optimizer Gamma',
    type: 'neural_optimizer',
    status: 'training',
    model: 'gpt-4',
    version: '1.0.1',
    capabilities: ['optimization', 'learning', 'adaptation'],
    performance: {
      success_rate: 0.92,
      avg_response_time: 1250,
      total_executions: 234
    },
    created_at: '2025-06-20T00:00:00Z',
    last_execution: '2025-06-28T09:45:00Z'
  }
];

/**
 * Components mock data
 * Used by: designer components, test files
 */
export const componentsMock = {
  ai_agent: {
    component_type: 'ai_agent',
    name: 'AI Agent',
    description: 'Intelligent processing agent',
    input_ports: ['data_input', 'config_input'],
    output_ports: ['result_output', 'status_output'],
    properties: {
      model: { type: 'string', default: 'gpt-3.5-turbo', options: ['gpt-3.5-turbo', 'gpt-4', 'claude-3'] },
      temperature: { type: 'number', default: 0.7, min: 0, max: 1 },
      max_tokens: { type: 'number', default: 1000, min: 1, max: 4000 }
    },
    visual: {
      color: '#4CAF50',
      icon: '🤖'
    }
  },
  data_processor: {
    component_type: 'data_processor',
    name: 'Data Processor',
    description: 'Process and transform data',
    input_ports: ['data_input'],
    output_ports: ['processed_output'],
    properties: {
      operation: { type: 'string', default: 'transform', options: ['transform', 'filter', 'aggregate'] },
      batch_size: { type: 'number', default: 100, min: 1, max: 1000 }
    },
    visual: {
      color: '#2196F3',
      icon: '⚙️'
    }
  },
  input_validator: {
    component_type: 'input_validator',
    name: 'Input Validator',
    description: 'Validate input data',
    input_ports: ['data_input'],
    output_ports: ['valid_output', 'invalid_output'],
    properties: {
      validation_type: { type: 'string', default: 'schema', options: ['schema', 'regex', 'custom'] },
      strict_mode: { type: 'boolean', default: true }
    },
    visual: {
      color: '#FF9800',
      icon: '✅'
    }
  }
};

/**
 * Design templates mock data
 * Used by: designer, test files
 */
export const designTemplatesMock = [
  {
    id: 'template_001',
    name: 'Basic AI Pipeline',
    description: 'Simple AI processing pipeline',
    category: 'ai_processing',
    nodes: [
      {
        node_id: 'input_1',
        component_type: 'input_validator',
        position: { x: 100, y: 100 },
        properties: { validation_type: 'schema' }
      },
      {
        node_id: 'agent_1',
        component_type: 'ai_agent',
        position: { x: 300, y: 100 },
        properties: { model: 'gpt-3.5-turbo', temperature: 0.7 }
      },
      {
        node_id: 'processor_1',
        component_type: 'data_processor',
        position: { x: 500, y: 100 },
        properties: { operation: 'transform' }
      }
    ],
    connections: [
      {
        connection_id: 'conn_1',
        from_node_id: 'input_1',
        to_node_id: 'agent_1',
        from_port: 'valid_output',
        to_port: 'data_input'
      },
      {
        connection_id: 'conn_2',
        from_node_id: 'agent_1',
        to_node_id: 'processor_1',
        from_port: 'result_output',
        to_port: 'data_input'
      }
    ]
  }
];

/**
 * Test fixtures for pytest
 * Used by: conftest.py and test files
 */
export const testFixtures = {
  sampleDesignData: {
    name: 'Sample Design',
    description: 'A sample design for testing',
    version: '1.0.0',
    canvas_settings: {
      width: 1200,
      height: 800,
      zoom: 1.0,
      grid_enabled: true
    }
  },
  
  sampleNodeData: {
    node_id: 'test_node_1',
    component_type: 'ai_agent',
    position: { x: 100, y: 200 },
    size: { width: 150, height: 100 },
    properties: {
      model: 'gpt-3.5-turbo',
      temperature: 0.7,
      max_tokens: 1000
    },
    visual_style: {
      color: '#4CAF50',
      border_color: '#2E7D32'
    }
  },
  
  sampleConnectionData: {
    connection_id: 'test_conn_1',
    from_node_id: 'node_1',
    to_node_id: 'node_2',
    from_port: 'output',
    to_port: 'input',
    connection_type: 'data_flow',
    properties: {
      data_type: 'text',
      buffer_size: 1000
    },
    visual_style: {
      color: '#2196F3',
      width: 2
    }
  },
  
  mockAiResponse: {
    content: 'Mock AI response',
    usage: {
      prompt_tokens: 10,
      completion_tokens: 20,
      total_tokens: 30
    },
    model: 'mock-model',
    finish_reason: 'stop'
  }
};

/**
 * API endpoint fallback data mapping
 * Used by: ApiClient.js, infrastructureManager.js
 */
export const apiFallbackData = {
  '/api/settings': { settings: settingsMock },
  '/api/metrics': metricsMock,
  '/api/infrastructure/services': servicesMock,
  '/api/infrastructure/status': {
    message: 'System operational',
    timestamp: new Date().toISOString(),
    health: 'healthy',
    services: Object.fromEntries(servicesMock.map(s => [s.name, s.status]))
  },
  '/api/users': usersMock,
  '/api/agents': agentsMock,
  '/api/components': componentsMock,
  '/api/designs/templates': designTemplatesMock,
  '/api/auth/me': {
    id: 'usr_001',
    username: 'demo_user',
    email: 'demo@enhanced-csp.com',
    full_name: 'Demo User',
    role: 'admin',
    permissions: ['read', 'write', 'delete', 'admin']
  }
};

/**
 * Generate dynamic mock data for time-sensitive endpoints
 */
export const generateDynamicMocks = {
  stats: generateStatsMock,
  
  currentTime: () => new Date().toISOString(),
  
  systemHealth: () => ({
    status: Math.random() > 0.1 ? 'healthy' : 'warning',
    uptime: Math.floor(Date.now() / 1000) - Math.floor(Math.random() * 86400),
    last_check: new Date().toISOString()
  }),
  
  realtimeMetrics: () => ({
    ...metricsMock,
    system: {
      ...metricsMock.system,
      cpu_usage: Math.random() * 30 + 20, // 20-50%
      memory_usage: Math.random() * 40 + 40, // 40-80%
      network_throughput: Math.random() * 200 + 50 // 50-250 MB/s
    },
    timestamp: new Date().toISOString()
  })
};

/**
 * Helper function to get fallback data by endpoint
 * @param {string} endpoint - API endpoint path
 * @returns {any} Fallback data for the endpoint
 */
export const getFallbackData = (endpoint) => {
  // Remove query parameters and normalize endpoint
  const normalizedEndpoint = endpoint.split('?')[0];
  
  // Check for exact match first
  if (apiFallbackData[normalizedEndpoint]) {
    return apiFallbackData[normalizedEndpoint];
  }
  
  // Check for pattern matches
  if (normalizedEndpoint.includes('/designs/')) {
    return designTemplatesMock[0]; // Return first template as default
  }
  
  if (normalizedEndpoint.includes('/agents/')) {
    return agentsMock[0]; // Return first agent as default
  }
  
  if (normalizedEndpoint.includes('/users/')) {
    return usersMock[0]; // Return first user as default
  }
  
  // Return null if no fallback data available
  console.warn(`No fallback data available for endpoint: ${endpoint}`);
  return null;
};

/**
 * Export all mock data collections for easy importing
 */
export default {
  settings: settingsMock,
  stats: generateStatsMock,
  metrics: metricsMock,
  services: servicesMock,
  users: usersMock,
  agents: agentsMock,
  components: componentsMock,
  designTemplates: designTemplatesMock,
  testFixtures: testFixtures,
  apiFallbackData: apiFallbackData,
  generateDynamicMocks: generateDynamicMocks,
  getFallbackData: getFallbackData
};