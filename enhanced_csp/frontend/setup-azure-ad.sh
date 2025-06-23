#!/bin/bash
# Enhanced CSP System - Azure AD Setup and Test Script

echo "🚀 Enhanced CSP System - Azure AD Authentication Setup"
echo "======================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/mate/PAIN/csp-agent-network/csp-agent-network-1/enhanced_csp"
FRONTEND_DIR="$PROJECT_DIR/frontend"
PAGES_DIR="$FRONTEND_DIR/pages"

echo -e "${BLUE}Project Directory: ${NC}$PROJECT_DIR"
echo -e "${BLUE}Frontend Directory: ${NC}$FRONTEND_DIR"
echo -e "${BLUE}Pages Directory: ${NC}$PAGES_DIR"
echo ""

# Step 1: Create directory structure if it doesn't exist
echo -e "${YELLOW}Step 1: Setting up directory structure...${NC}"
mkdir -p "$FRONTEND_DIR"/{config,services,components,assets}
mkdir -p "$PAGES_DIR"
echo -e "${GREEN}✅ Directory structure ready${NC}"

# Step 2: Create Azure AD configuration file
echo -e "${YELLOW}Step 2: Creating Azure AD configuration...${NC}"
cat > "$FRONTEND_DIR/config/azureConfig.js" << 'EOF'
// Azure AD Configuration for Enhanced CSP System
// Your actual Azure AD app registration details

const azureConfig = {
    development: {
        clientId: "53537e30-ae6b-48f7-9c7c-4db20fc27850",
        tenantId: "622a5fe0-fac1-4213-9cf7-d5f6defdf4c4",
        redirectUri: "http://localhost:3000",
        postLogoutRedirectUri: "http://localhost:3000",
        environment: "development"
    },
    production: {
        clientId: "53537e30-ae6b-48f7-9c7c-4db20fc27850",
        tenantId: "622a5fe0-fac1-4213-9cf7-d5f6defdf4c4",
        redirectUri: "https://your-production-domain.com",
        postLogoutRedirectUri: "https://your-production-domain.com",
        environment: "production"
    }
};

// Get current environment configuration
function getAzureConfig() {
    const env = window.location.hostname === 'localhost' ? 'development' : 'production';
    return azureConfig[env];
}

// MSAL Configuration
export const msalConfig = {
    auth: {
        clientId: getAzureConfig().clientId,
        authority: `https://login.microsoftonline.com/${getAzureConfig().tenantId}`,
        redirectUri: getAzureConfig().redirectUri,
        postLogoutRedirectUri: getAzureConfig().postLogoutRedirectUri,
        navigateToLoginRequestUrl: false
    },
    cache: {
        cacheLocation: "sessionStorage",
        storeAuthStateInCookie: false,
    }
};

// Login request configuration
export const loginRequest = {
    scopes: ["User.Read", "User.ReadBasic.All", "Group.Read.All"],
    prompt: "select_account"
};

// User roles configuration
export const USER_ROLES = {
    SUPER_ADMIN: 'super_admin',
    ADMIN: 'admin',
    DEVELOPER: 'developer',
    ANALYST: 'analyst',
    USER: 'user'
};

export default {
    msalConfig,
    loginRequest,
    USER_ROLES,
    getAzureConfig
};
EOF
echo -e "${GREEN}✅ Azure AD configuration created${NC}"

# Step 3: Create authentication service
echo -e "${YELLOW}Step 3: Creating authentication service...${NC}"
cat > "$FRONTEND_DIR/services/authService.js" << 'EOF'
// Authentication Service for Enhanced CSP System
import { msalConfig, loginRequest, USER_ROLES } from '../config/azureConfig.js';

class AuthService {
    constructor() {
        this.msalInstance = null;
        this.account = null;
        this.userRole = null;
        this.initializeAuth();
    }

    async initializeAuth() {
        if (typeof msal !== 'undefined') {
            this.msalInstance = new msal.PublicClientApplication(msalConfig);
            await this.msalInstance.initialize();
            this.account = this.msalInstance.getActiveAccount();
            
            if (this.account) {
                await this.setUserRole();
            }
        }
    }

    async login() {
        if (!this.msalInstance) {
            throw new Error('MSAL not initialized');
        }

        try {
            const loginResponse = await this.msalInstance.loginPopup(loginRequest);
            this.account = loginResponse.account;
            this.msalInstance.setActiveAccount(this.account);
            await this.setUserRole();
            return loginResponse;
        } catch (error) {
            console.error('Login failed:', error);
            throw error;
        }
    }

    async logout() {
        if (this.msalInstance) {
            await this.msalInstance.logoutPopup();
            this.account = null;
            this.userRole = null;
        }
    }

    async setUserRole() {
        // Get user role from Azure AD groups or email domain
        const email = this.account?.username || '';
        
        // Map based on email domain (customize as needed)
        if (email.includes('admin@')) this.userRole = USER_ROLES.ADMIN;
        else if (email.includes('dev@')) this.userRole = USER_ROLES.DEVELOPER;
        else this.userRole = USER_ROLES.USER;
    }

    isAuthenticated() {
        return this.account !== null;
    }

    getUserInfo() {
        return this.account;
    }

    getUserRole() {
        return this.userRole || USER_ROLES.USER;
    }
}

export const authService = new AuthService();
EOF
echo -e "${GREEN}✅ Authentication service created${NC}"

# Step 4: Create updated login.html (backing up original if exists)
echo -e "${YELLOW}Step 4: Setting up login page...${NC}"
if [ -f "$PAGES_DIR/login.html" ]; then
    cp "$PAGES_DIR/login.html" "$PAGES_DIR/login.html.backup"
    echo -e "${BLUE}📁 Original login.html backed up as login.html.backup${NC}"
fi

# The HTML content will be created separately since it's quite large

# Step 5: Create a simple HTTP server script for testing
echo -e "${YELLOW}Step 5: Creating test server...${NC}"
cat > "$FRONTEND_DIR/test-server.py" << 'EOF'
#!/usr/bin/env python3
# Simple HTTP server for testing Enhanced CSP System
import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 3000
DIRECTORY = Path(__file__).parent

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"🚀 Enhanced CSP Test Server")
        print(f"📡 Serving at http://localhost:{PORT}")
        print(f"📁 Directory: {DIRECTORY}")
        print(f"🔐 Login page: http://localhost:{PORT}/pages/login.html")
        print(f"📊 Dashboard: http://localhost:{PORT}/csp_admin_portal.html")
        print("Press Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
            sys.exit(0)
EOF

chmod +x "$FRONTEND_DIR/test-server.py"
echo -e "${GREEN}✅ Test server script created${NC}"

# Step 6: Create package.json for npm dependencies (optional)
echo -e "${YELLOW}Step 6: Creating package.json...${NC}"
cat > "$FRONTEND_DIR/package.json" << 'EOF'
{
  "name": "enhanced-csp-frontend",
  "version": "2.1.0",
  "description": "Enhanced CSP System Frontend with Azure AD Authentication",
  "main": "pages/login.html",
  "scripts": {
    "start": "python3 test-server.py",
    "dev": "python3 test-server.py",
    "test": "echo \"No tests yet\" && exit 0"
  },
  "dependencies": {
    "@azure/msal-browser": "^2.38.3"
  },
  "devDependencies": {
    "live-server": "^1.2.2"
  },
  "keywords": ["csp", "azure-ad", "authentication", "ai", "dashboard"],
  "author": "CSP Development Team",
  "license": "MIT"
}
EOF
echo -e "${GREEN}✅ Package.json created${NC}"

# Step 7: Create environment setup script
echo -e "${YELLOW}Step 7: Creating environment setup...${NC}"
cat > "$FRONTEND_DIR/.env.example" << 'EOF'
# Enhanced CSP System Environment Configuration
# Copy this file to .env and update with your values

# Azure AD Configuration
AZURE_CLIENT_ID=53537e30-ae6b-48f7-9c7c-4db20fc27850
AZURE_TENANT_ID=622a5fe0-fac1-4213-9cf7-d5f6defdf4c4
AZURE_REDIRECT_URI=http://localhost:3000
AZURE_POST_LOGOUT_REDIRECT_URI=http://localhost:3000

# Application Settings
APP_ENV=development
APP_PORT=3000
APP_HOST=localhost

# Security Settings
SESSION_TIMEOUT=28800  # 8 hours in seconds
ENABLE_DEMO_MODE=true

# API Configuration
API_BASE_URL=http://localhost:8080/api
API_TIMEOUT=30000

# Logging
LOG_LEVEL=debug
ENABLE_CONSOLE_LOGS=true
EOF
echo -e "${GREEN}✅ Environment configuration created${NC}"

# Step 8: Create README for setup
echo -e "${YELLOW}Step 8: Creating setup documentation...${NC}"
cat > "$FRONTEND_DIR/README.md" << 'EOF'
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

Your Azure AD app details are already configured:
- Client ID: 53537e30-ae6b-48f7-9c7c-4db20fc27850
- Tenant ID: 622a5fe0-fac1-4213-9cf7-d5f6defdf4c4

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
EOF
echo -e "${GREEN}✅ Setup documentation created${NC}"

# Step 9: Provide setup summary
echo ""
echo -e "${GREEN}🎉 Setup Complete! Here's what was created:${NC}"
echo -e "${BLUE}📁 Directory Structure:${NC}"
echo "   ├── config/azureConfig.js (Azure AD configuration)"
echo "   ├── services/authService.js (Authentication service)"
echo "   ├── test-server.py (Development server)"
echo "   ├── package.json (Dependencies)"
echo "   ├── .env.example (Environment template)"
echo "   └── README.md (Setup guide)"
echo ""

echo -e "${YELLOW}🔧 Next Steps:${NC}"
echo "1. Reconfigure Azure AD as SPA (follow the reconfiguration guide)"
echo "2. Copy the updated login.html to pages/login.html"
echo "3. Start the test server:"
echo "   cd $FRONTEND_DIR"
echo "   python3 test-server.py"
echo "4. Open http://localhost:3000/pages/login.html"
echo "5. Test Azure AD authentication"
echo ""

echo -e "${RED}⚠️  Important:${NC}"
echo "- Make sure to reconfigure Azure AD as SPA (not Web app)"
echo "- Remove the client secret from Azure AD"
echo "- Add localhost:3000 as a redirect URI"
echo "- Grant admin consent for API permissions"
echo ""

echo -e "${GREEN}🚀 Your Enhanced CSP System is ready for Azure AD authentication!${NC}"

# Make the script executable
chmod +x "$0"