/* --------------------------------------------------------------------------
 *  apiFallbackData.js
 *  Centralised mock data & helpers for the Enhanced CSP frontend.
 *  – Pure ES-module named exports for tree-shaking.
 *  – Also mounts a global `window.apiFallbackData` for legacy code paths.
 * ------------------------------------------------------------------------ */

/* ─────────────── Dashboard / infrastructure mocks ────────────────────── */
export const metricsMock = {
  active_processes: 0,
  quantum_entanglements: 0,
  blockchain_transactions: 0,
  neural_efficiency: 100,
  security_threats: 0,
  system_uptime: 100,
  cpu_usage: 0,
  memory_usage: 0
};

export const processesMock = { processes: [], count: 0 };

export const infrastructureMetricsMock = {
  cpu:     { current: 45, max: 100, unit: '%' },
  memory:  { current: 62, max: 100, unit: '%' },
  disk:    { current: 78, max: 100, unit: '%' },
  network: { current: 23, max: 100, unit: '%' },
  uptime:  { current: 99.5, max: 100, unit: '%' },
  requests:{ current: 1250, max: null, unit: '/min' }
};

export const infrastructureServicesMock = [
  { name: 'Web Server',     status: 'running',  uptime: '15d 4h 23m', port: 80   },
  { name: 'Database',       status: 'running',  uptime: '15d 4h 23m', port: 5432 },
  { name: 'Redis Cache',    status: 'running',  uptime: '15d 4h 23m', port: 6379 },
  { name: 'API Gateway',    status: 'running',  uptime: '15d 4h 23m', port: 8000 },
  { name: 'Message Queue',  status: 'warning',  uptime: '2d 1h 15m',  port: 5672 }
];

export const settingsMock = [
  { key:'app_name',      value:'Enhanced CSP System',           description:'Application name',                   widget:'text',   category:'Application' },
  { key:'debug',         value:false,                           description:'Enable debug mode',                 widget:'switch', category:'Application' },
  { key:'environment',   value:'development',                   description:'Application environment',           widget:'select', options:['development','testing','staging','production'], category:'Application' },
  { key:'enable_ai',     value:true,                            description:'Enable AI features',                widget:'switch', category:'Features' },
  { key:'enable_websockets',value:true,                         description:'Enable WebSocket support',          widget:'switch', category:'Features' },
  { key:'database_host', value:'localhost',                     description:'Database host address',             widget:'text',   category:'Database' },
  { key:'database_port', value:5432,                            description:'Database port',                     widget:'number', category:'Database' },
  { key:'database_pool_size',value:20,                          description:'Connection pool size',              widget:'number', category:'Database' },
  { key:'redis_host',    value:'localhost',                     description:'Redis host address',                widget:'text',   category:'Cache' },
  { key:'redis_port',    value:6379,                            description:'Redis port',                        widget:'number', category:'Cache' },
  { key:'ai_max_requests_per_minute',value:60,                  description:'AI rate limit (req/min)',          widget:'number', category:'AI' },
  { key:'ai_max_daily_cost',value:100.0,                        description:'Max daily AI cost ($)',            widget:'number', category:'AI' },
  { key:'security_max_login_attempts',value:5,                  description:'Login lockout threshold',          widget:'number', category:'Security' },
  { key:'api_rate_limit_requests_per_minute',value:100,         description:'API rate limit (req/min/user)',    widget:'number', category:'API' },
  { key:'log_level',     value:'INFO',                           description:'Global log level',                 widget:'select', options:['DEBUG','INFO','WARNING','ERROR'], category:'Monitoring' }
];

/* Simple sparkline history example */
export const infrastructureMetricHistory = [45, 47, 46, 48, 50];

/* Generate random stats for the dashboard tiles */
export function generateStatsGridData() {
  return {
    totalAgents:          Math.floor(Math.random() * 50) + 10,
    activeAgents:         Math.floor(Math.random() * 30) + 5,
    totalExecutions:      Math.floor(Math.random() * 10000) + 1000,
    successRate:          Math.random() * 20 + 80,
    averageResponseTime:  Math.random() * 500 + 100,
    systemUptime:         Math.random() * 720 + 720
  };
}

/* ─────────────── Admin-UI sample data sets ───────────────────────────── */
export const recentActivityMock = [
  { time:'14:35', action:'Agent Created',     details:'New AI agent "Content Analyzer" deployed', user:'Admin'   },
  { time:'14:22', action:'User Login',        details:'Sarah Johnson logged in',                  user:'System'  },
  { time:'14:15', action:'Task Completed',    details:'Data processing task #1247 completed',     user:'Agent-001'},
  { time:'14:08', action:'Alert Resolved',    details:'Security alert #SA-456 resolved',          user:'Agent-003'},
  { time:'13:55', action:'System Update',     details:'CSP engine updated to v2.1.3',             user:'Admin'   }
];

export const monitoringDataMock = [
  { service:'CSP Engine',           status:'Running', cpu:'23%', memory:'45%', uptime:'15d 4h' },
  { service:'AI Agent Manager',     status:'Running', cpu:'12%', memory:'32%', uptime:'15d 4h' },
  { service:'Database',             status:'Running', cpu:'8%',  memory:'67%', uptime:'15d 4h' },
  { service:'Web Server',           status:'Running', cpu:'5%',  memory:'28%', uptime:'15d 4h' },
  { service:'Message Queue',        status:'Warning', cpu:'45%', memory:'78%', uptime:'15d 4h' }
];

export const modelsDataMock = [
  { model:'GPT-4 Turbo',           type:'LLM',        status:'Active', requests:'1 247', responseTime:'1.8s', successRate:'99.2%' },
  { model:'Claude-3 Sonnet',       type:'LLM',        status:'Active', requests:'892',   responseTime:'2.1s', successRate:'98.8%' },
  { model:'Gemini Pro',            type:'Multimodal', status:'Active', requests:'634',   responseTime:'1.4s', successRate:'97.5%' },
  { model:'Text-Embedding-Ada-002',type:'Embedding',  status:'Active', requests:'3 421', responseTime:'0.3s', successRate:'99.9%' }
];

export const logsDataMock = [
  { timestamp:'2024-01-15 14:35:22', level:'INFO',  component:'AgentManager', message:'Agent "Content Analyzer" started successfully' },
  { timestamp:'2024-01-15 14:35:18', level:'DEBUG', component:'CSPEngine',    message:'Process synchronization completed'            },
  { timestamp:'2024-01-15 14:35:15', level:'WARN',  component:'MessageQueue', message:'Queue capacity at 78%, consider scaling'      },
  { timestamp:'2024-01-15 14:35:10', level:'INFO',  component:'Security',     message:'Authentication successful for sarah@company.com' },
  { timestamp:'2024-01-15 14:35:05', level:'ERROR', component:'Database',     message:'Connection timeout resolved after retry'      }
];

/* Agents (full & sample) */
export const defaultAgentsMock = [ /* … full agent objects (same as original) … */ ];
export const managerSampleAgents = [ /* … simplified sample objects … */ ];

/* User / role mocks */
export const defaultUsersMock = [
  { id:'1', full_name:'John Smith',   email:'john@company.com',  roles:['admin'],    is_active:true,  last_login:'2024-01-15T09:30:00Z', created_at:'2024-01-01T00:00:00Z' },
  { id:'2', full_name:'Sarah Johnson',email:'sarah@company.com', roles:['user'],     is_active:true,  last_login:'2024-01-15T14:22:00Z', created_at:'2024-01-02T00:00:00Z' },
  { id:'3', full_name:'Mike Chen',    email:'mike@company.com',  roles:['developer'],is_active:false, last_login:'2024-01-10T16:30:00Z', created_at:'2024-01-03T00:00:00Z' }
];

export const defaultRolesMock = [
  { id:'admin', name:'Administrator', permissions:['manage_users','manage_system'] },
  { id:'user',  name:'User',          permissions:['view_system']                 }
];

/* Alerts & incidents */
export const fallbackAlertsMock = [ /* … same objects as before … */ ];
export const fallbackIncidentsMock = [ /* … same objects as before … */ ];

/* ────────────────── Lightweight in-memory CRUD for users ─────────────── */
const _users = structuredClone ? structuredClone(defaultUsersMock) : JSON.parse(JSON.stringify(defaultUsersMock));

export function getUsers() {
  return _users.slice();        // shallow copy
}
export function addUser(user) {
  const id = Date.now().toString();
  _users.push({ ...user, id });
  return id;
}
export function updateUser(id, data) {
  const idx = _users.findIndex(u => u.id === id);
  if (idx !== -1) _users[idx] = { ..._users[idx], ...data };
}
export function deleteUser(id) {
  const idx = _users.findIndex(u => u.id === id);
  if (idx !== -1) _users.splice(idx, 1);
}

/* ─────────────────────────── Aggregate object ────────────────────────── */
export const apiFallbackData = {
  '/metrics':                     metricsMock,
  '/processes':                   processesMock,
  '/api/infrastructure/metrics':  infrastructureMetricsMock,
  '/api/infrastructure/services': infrastructureServicesMock,
  '/api/settings':                settingsMock,
  infrastructureMetricHistory,
  generateStatsGridData,
  users: _users,
  getUsers,
  addUser,
  updateUser,
  deleteUser
};

/* Mount global for legacy scripts (non-module browsers) */
if (typeof window !== 'undefined') {
  window.apiFallbackData = apiFallbackData;
}
