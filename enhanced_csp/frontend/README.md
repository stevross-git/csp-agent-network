# 🚀 Enhanced CSP System - Frontend Setup

## Quick Start

1. **Configure Azure AD** (if not done already):
   - Follow the reconfiguration guide
   - Set up as Single-page application (SPA)
   - Add redirect URIs for localhost:3000

2. **Start the test server**:
   ```bash
   cd /home/mate/PAIN/csp-agent-network/csp-agent-network-1/enhanced_csp/frontend
   python3 test-server.py
   ```

3. **Access the application**:
   - Login: http://localhost:3000/pages/login.html
   - Dashboard: http://localhost:3000/csp_admin_portal.html

## Configuration



## File Structure

```
frontend/
├── config/
│   └── azureConfig.js       # Azure AD configuration
├── services/
│   └── authService.js       # Authentication service
├── pages/
│   └── login.html          # Login page with Azure AD
├── test-server.py          # Development server
├── package.json            # Dependencies
└── README.md              # This file
```

## Next Steps

1. Test Azure AD authentication
2. Create user groups in Azure AD
3. Implement role-based access control
4. Deploy to production
