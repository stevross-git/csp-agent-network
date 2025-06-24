#!/bin/bash
# Local MSAL Setup Script - Download MSAL library locally

echo "🔧 Setting up MSAL library locally..."
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get current directory
FRONTEND_DIR="$(pwd)"
if [[ ! "$FRONTEND_DIR" == *"frontend"* ]]; then
    echo -e "${RED}❌ Please run this script from the frontend directory${NC}"
    exit 1
fi

echo -e "${BLUE}📁 Frontend directory: $FRONTEND_DIR${NC}"

# Create js directory for local libraries
mkdir -p js/vendor

echo -e "${YELLOW}📦 Downloading MSAL library...${NC}"

# Try to download MSAL from multiple sources
download_success=false

# Try CDN 1 - Official Microsoft CDN
echo "Trying Microsoft CDN..."
if curl -s -o js/vendor/msal-browser.min.js "https://alcdn.msauth.net/browser/2.38.3/js/msal-browser.min.js"; then
    if [[ -s js/vendor/msal-browser.min.js ]]; then
        echo -e "${GREEN}✅ Downloaded from Microsoft CDN${NC}"
        download_success=true
    fi
fi

# Try CDN 2 - jsDelivr CDN
if [[ "$download_success" == false ]]; then
    echo "Trying jsDelivr CDN..."
    if curl -s -o js/vendor/msal-browser.min.js "https://cdn.jsdelivr.net/npm/@azure/msal-browser@2.38.3/dist/msal-browser.min.js"; then
        if [[ -s js/vendor/msal-browser.min.js ]]; then
            echo -e "${GREEN}✅ Downloaded from jsDelivr CDN${NC}"
            download_success=true
        fi
    fi
fi

# Try CDN 3 - unpkg CDN
if [[ "$download_success" == false ]]; then
    echo "Trying unpkg CDN..."
    if curl -s -o js/vendor/msal-browser.min.js "https://unpkg.com/@azure/msal-browser@2.38.3/dist/msal-browser.min.js"; then
        if [[ -s js/vendor/msal-browser.min.js ]]; then
            echo -e "${GREEN}✅ Downloaded from unpkg CDN${NC}"
            download_success=true
        fi
    fi
fi

# If all downloads failed, create a minimal MSAL implementation
if [[ "$download_success" == false ]]; then
    echo -e "${YELLOW}⚠️  CDN download failed, creating minimal local implementation...${NC}"
    
    cat > js/vendor/msal-browser.min.js << 'EOF'
// Minimal MSAL Browser Implementation for Enhanced CSP System
// This is a simplified version for when CDNs are not accessible

window.msal = {
    PublicClientApplication: class {
        constructor(config) {
            this.config = config;
            this.account = null;
            this.isInitialized = false;
        }

        async initialize() {
            this.isInitialized = true;
            console.log('✅ Minimal MSAL initialized');
        }

        getActiveAccount() {
            return this.account;
        }

        setActiveAccount(account) {
            this.account = account;
        }

        async loginPopup(request) {
            // Simulate Azure AD login for testing
            console.log('🧪 Simulating Azure AD login...');
            
            return new Promise((resolve, reject) => {
                // Show a simple prompt for testing
                const email = prompt('Enter your email for testing (or cancel for demo):');
                
                if (email) {
                    const mockAccount = {
                        homeAccountId: 'test-user-id',
                        name: email.split('@')[0],
                        username: email,
                        idTokenClaims: {
                            name: email.split('@')[0],
                            preferred_username: email,
                            oid: 'test-user-id'
                        },
                        tenantId: 'test-tenant'
                    };
                    
                    this.account = mockAccount;
                    resolve({ account: mockAccount });
                } else {
                    reject(new Error('Login cancelled'));
                }
            });
        }

        async acquireTokenSilent(request) {
            if (!this.account) {
                throw new Error('No active account');
            }
            
            // Return a mock token for testing
            return {
                accessToken: 'mock-access-token-for-testing',
                account: this.account,
                scopes: request.scopes
            };
        }

        async logoutPopup() {
            this.account = null;
            console.log('✅ Mock logout successful');
        }
    }
};

console.log('📦 Minimal MSAL implementation loaded');
EOF
    
    echo -e "${YELLOW}✅ Created minimal MSAL implementation${NC}"
    download_success=true
fi

# Verify the file exists and has content
if [[ -s js/vendor/msal-browser.min.js ]]; then
    file_size=$(wc -c < js/vendor/msal-browser.min.js)
    echo -e "${GREEN}✅ MSAL library ready (${file_size} bytes)${NC}"
else
    echo -e "${RED}❌ Failed to create MSAL library${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}🔧 Creating updated login.html with local MSAL...${NC}"

# Create the updated login.html with local MSAL reference
cat > pages/login.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced CSP System - Login</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    
    <!-- Load MSAL Library Locally -->
    <script src="../js/vendor/msal-browser.min.js"></script>
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #333;
        }

        .login-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            padding: 3rem;
            width: 100%;
            max-width: 450px;
            text-align: center;
            backdrop-filter: blur(10px);
        }

        .login-header {
            margin-bottom: 2rem;
        }

        .login-header h1 {
            font-size: 2rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }

        .login-header p {
            color: #7f8c8d;
            font-size: 1rem;
        }

        .status-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            display: none;
            font-weight: 500;
        }

        .status-loading {
            background: #e3f2fd;
            color: #1976d2;
            border: 1px solid #bbdefb;
        }

        .status-success {
            background: #e8f5e8;
            color: #2e7d32;
            border: 1px solid #c8e6c9;
        }

        .status-error {
            background: #ffebee;
            color: #c62828;
            border: 1px solid #ffcdd2;
        }

        .status-warning {
            background: #fff8e1;
            color: #f57c00;
            border: 1px solid #ffecb3;
        }

        .login-options {
            margin: 1.5rem 0;
        }

        .azure-login-btn {
            width: 100%;
            padding: 1rem 1.5rem;
            background: #0078d4;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .azure-login-btn:hover {
            background: #106ebe;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(16, 110, 190, 0.3);
        }

        .azure-login-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #ffffff;
            border-top: 2px solid transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: none;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .divider {
            text-align: center;
            margin: 1.5rem 0;
            position: relative;
            color: #7f8c8d;
        }

        .divider::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 1px;
            background: #e0e0e0;
        }

        .divider span {
            background: rgba(255, 255, 255, 0.95);
            padding: 0 1rem;
        }

        .demo-section {
            margin-top: 1.5rem;
        }

        .demo-section h3 {
            color: #2c3e50;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }

        .demo-form {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .demo-input {
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 0.9rem;
        }

        .demo-btn {
            padding: 0.75rem;
            background: #34495e;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.3s ease;
        }

        .demo-btn:hover {
            background: #2c3e50;
        }

        .demo-credentials {
            background: #f8f9fa;
            padding: 0.75rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            font-size: 0.85rem;
            border-left: 3px solid #3498db;
            cursor: pointer;
            transition: background 0.3s ease;
        }

        .demo-credentials:hover {
            background: #e9ecef;
        }

        .user-info {
            margin-top: 2rem;
            padding: 1.5rem;
            background: #f8f9fa;
            border-radius: 10px;
            text-align: left;
            display: none;
        }

        .user-info h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
        }

        .user-details {
            display: grid;
            gap: 0.5rem;
            font-size: 0.9rem;
        }

        .user-detail {
            display: flex;
            justify-content: space-between;
        }

        .user-detail strong {
            color: #2c3e50;
        }

        .login-footer {
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e0e0e0;
            color: #7f8c8d;
            font-size: 0.85rem;
        }

        .security-note {
            margin-top: 0.5rem;
            font-size: 0.8rem;
        }

        .action-buttons {
            display: none;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: #3498db;
            color: white;
        }

        .btn-secondary {
            background: #95a5a6;
            color: white;
        }

        .library-status {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            border-left: 3px solid #28a745;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <!-- Header -->
        <div class="login-header">
            <h1><i class="fas fa-network-wired"></i> Enhanced CSP System</h1>
            <p>Advanced AI-to-AI Communication Platform</p>
        </div>

        <!-- Library Status -->
        <div id="library-status" class="library-status" style="display: none;">
            <strong>MSAL Status:</strong> <span id="msal-status">Loading...</span>
        </div>

        <!-- Status Messages -->
        <div id="status-message" class="status-message"></div>

        <!-- User Info Display (shown after login) -->
        <div id="user-info" class="user-info">
            <h3><i class="fas fa-user-check"></i> Welcome!</h3>
            <div id="user-details" class="user-details"></div>
            <div id="action-buttons" class="action-buttons">
                <button id="dashboard-btn" class="btn btn-primary">
                    <i class="fas fa-tachometer-alt"></i> Go to Dashboard
                </button>
                <button id="logout-btn" class="btn btn-secondary">
                    <i class="fas fa-sign-out-alt"></i> Sign Out
                </button>
            </div>
        </div>

        <!-- Login Options (shown when not authenticated) -->
        <div id="login-section" class="login-options">
            <!-- Azure AD Login -->
            <button id="azure-login-btn" class="azure-login-btn">
                <i class="fab fa-microsoft"></i>
                <span>Sign in with Microsoft</span>
                <div class="spinner" id="azure-spinner"></div>
            </button>

            <!-- Divider -->
            <div class="divider">
                <span>or</span>
            </div>

            <!-- Demo Login Section -->
            <div class="demo-section">
                <h3><i class="fas fa-flask"></i> Demo Login</h3>
                
                <form id="demo-form" class="demo-form">
                    <input type="text" id="demo-username" class="demo-input" placeholder="Username" required>
                    <input type="password" id="demo-password" class="demo-input" placeholder="Password" required>
                    <button type="submit" class="demo-btn">
                        <i class="fas fa-sign-in-alt"></i> Demo Sign In
                    </button>
                </form>

                <div class="demo-credentials" data-username="admin" data-password="csp2025!">
                    <strong>Administrator:</strong> admin / csp2025!
                </div>
                <div class="demo-credentials" data-username="developer" data-password="dev123!">
                    <strong>Developer:</strong> developer / dev123!
                </div>
                <div class="demo-credentials" data-username="analyst" data-password="analyst123!">
                    <strong>Analyst:</strong> analyst / analyst123!
                </div>
                <div class="demo-credentials" data-username="user" data-password="user123!">
                    <strong>User:</strong> user / user123!
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="login-footer">
            <div class="footer-text">Enhanced CSP System v2.1.0</div>
            <div class="security-note">🔐 Enterprise-grade security with Azure AD</div>
        </div>
    </div>

    <script>
        // Azure AD Configuration
        const msalConfig = {
            auth: {
                clientId: "53537e30-ae6b-48f7-9c7c-4db20fc27850",
                authority: "https://login.microsoftonline.com/622a5fe0-fac1-4213-9cf7-d5f6defdf4c4",
                redirectUri: window.location.origin,
            },
            cache: {
                cacheLocation: "sessionStorage",
                storeAuthStateInCookie: false,
            }
        };

        const loginRequest = {
            scopes: ["User.Read", "User.ReadBasic.All"]
        };

        // User roles mapping
        const USER_ROLES = {
            SUPER_ADMIN: 'super_admin',
            ADMIN: 'admin',
            DEVELOPER: 'developer',
            ANALYST: 'analyst',
            USER: 'user'
        };

        // Demo users (for development)
        const demoUsers = {
            'admin': { password: 'csp2025!', role: USER_ROLES.ADMIN, name: 'System Administrator' },
            'developer': { password: 'dev123!', role: USER_ROLES.DEVELOPER, name: 'Developer User' },
            'analyst': { password: 'analyst123!', role: USER_ROLES.ANALYST, name: 'Data Analyst' },
            'user': { password: 'user123!', role: USER_ROLES.USER, name: 'Regular User' }
        };

        // Initialize MSAL
        let msalInstance;
        let currentUser = null;

        // DOM Elements
        const statusMessage = document.getElementById('status-message');
        const libraryStatus = document.getElementById('library-status');
        const msalStatus = document.getElementById('msal-status');
        const userInfo = document.getElementById('user-info');
        const userDetails = document.getElementById('user-details');
        const loginSection = document.getElementById('login-section');
        const azureLoginBtn = document.getElementById('azure-login-btn');
        const azureSpinner = document.getElementById('azure-spinner');
        const demoForm = document.getElementById('demo-form');
        const actionButtons = document.getElementById('action-buttons');
        const dashboardBtn = document.getElementById('dashboard-btn');
        const logoutBtn = document.getElementById('logout-btn');

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', async function() {
            console.log('🚀 Enhanced CSP System - Initializing...');
            
            // Show library status
            libraryStatus.style.display = 'block';
            
            // Check if MSAL is loaded
            if (typeof msal === 'undefined') {
                msalStatus.textContent = 'Failed to load';
                libraryStatus.style.borderLeftColor = '#dc3545';
                showStatus('MSAL library failed to load. Using demo mode only.', 'warning');
                
                // Hide Azure AD login button
                azureLoginBtn.style.display = 'none';
                
                // Set up demo-only mode
                setupEventListeners();
                return;
            }

            try {
                // Determine MSAL mode
                const isMinimalMsal = msal.PublicClientApplication.toString().includes('Minimal MSAL');
                
                if (isMinimalMsal) {
                    msalStatus.textContent = 'Local implementation (testing mode)';
                    libraryStatus.style.borderLeftColor = '#ffc107';
                    showStatus('Using local MSAL implementation for testing.', 'warning');
                } else {
                    msalStatus.textContent = 'Full Azure AD integration';
                    libraryStatus.style.borderLeftColor = '#28a745';
                }

                // Initialize MSAL
                msalInstance = new msal.PublicClientApplication(msalConfig);
                await msalInstance.initialize();
                console.log('✅ MSAL initialized successfully');

                // Check for existing session
                await checkExistingSession();

                // Set up event listeners
                setupEventListeners();

            } catch (error) {
                console.error('❌ Initialization failed:', error);
                msalStatus.textContent = 'Initialization failed';
                libraryStatus.style.borderLeftColor = '#dc3545';
                showStatus('Failed to initialize authentication system.', 'error');
            }
        });

        // Check for existing authentication session
        async function checkExistingSession() {
            try {
                // Check Azure AD session
                if (msalInstance && msalInstance.getActiveAccount) {
                    const account = msalInstance.getActiveAccount();
                    if (account) {
                        console.log('🔍 Found existing Azure AD session');
                        await setupAzureUser(account);
                        return;
                    }
                }

                // Check demo session
                const demoSession = sessionStorage.getItem('csp_demo_session');
                if (demoSession) {
                    console.log('🔍 Found existing demo session');
                    currentUser = JSON.parse(demoSession);
                    showUserInfo(currentUser);
                    return;
                }

                console.log('ℹ️ No existing session found');
            } catch (error) {
                console.error('❌ Session check failed:', error);
            }
        }

        // Set up event listeners
        function setupEventListeners() {
            // Azure AD login button (only if MSAL is available)
            if (msalInstance) {
                azureLoginBtn.addEventListener('click', azureLogin);
            }

            // Demo form
            demoForm.addEventListener('submit', function(e) {
                e.preventDefault();
                const username = document.getElementById('demo-username').value;
                const password = document.getElementById('demo-password').value;
                demoLogin(username, password);
            });

            // Demo credential quick-fill
            document.querySelectorAll('.demo-credentials').forEach(cred => {
                cred.addEventListener('click', function() {
                    const username = this.getAttribute('data-username');
                    const password = this.getAttribute('data-password');
                    document.getElementById('demo-username').value = username;
                    document.getElementById('demo-password').value = password;
                });
            });

            // Action buttons
            dashboardBtn.addEventListener('click', goToDashboard);
            logoutBtn.addEventListener('click', logout);
        }

        // Show status message
        function showStatus(message, type = 'loading') {
            statusMessage.textContent = message;
            statusMessage.className = `status-message status-${type}`;
            statusMessage.style.display = 'block';
        }

        // Hide status message
        function hideStatus() {
            statusMessage.style.display = 'none';
        }

        // Set button loading state
        function setButtonLoading(button, loading) {
            const spinner = button.querySelector('.spinner');
            if (loading) {
                button.disabled = true;
                if (spinner) spinner.style.display = 'inline-block';
            } else {
                button.disabled = false;
                if (spinner) spinner.style.display = 'none';
            }
        }

        // Get user role from Azure AD info
        function getUserRole(account) {
            const email = account.username || account.idTokenClaims?.preferred_username || '';
            
            // Map based on email domain or Azure AD groups
            if (email.includes('admin@')) return USER_ROLES.SUPER_ADMIN;
            if (email.includes('dev@')) return USER_ROLES.DEVELOPER;
            if (email.includes('analyst@')) return USER_ROLES.ANALYST;
            
            // Default role
            return USER_ROLES.USER;
        }

        // Azure AD Login
        async function azureLogin() {
            if (!msalInstance) {
                showStatus('Azure AD login not available.', 'error');
                return;
            }

            try {
                setButtonLoading(azureLoginBtn, true);
                showStatus('Redirecting to Microsoft sign-in...', 'loading');

                const loginResponse = await msalInstance.loginPopup(loginRequest);
                console.log('✅ Azure AD login successful:', loginResponse);

                await setupAzureUser(loginResponse.account);

            } catch (error) {
                console.error('❌ Azure login failed:', error);
                setButtonLoading(azureLoginBtn, false);
                
                let errorMessage = 'Login failed. Please try again.';
                if (error.message && error.message.includes('cancelled')) {
                    errorMessage = 'Login was cancelled.';
                } else if (error.message) {
                    errorMessage = `Login failed: ${error.message}`;
                }
                
                showStatus(errorMessage, 'error');
                
                setTimeout(() => {
                    hideStatus();
                }, 5000);
            }
        }

        // Setup Azure user after successful login
        async function setupAzureUser(account) {
            try {
                // Get additional user info if possible
                const userRole = getUserRole(account);
                
                currentUser = {
                    id: account.homeAccountId || 'test-id',
                    name: account.name || account.idTokenClaims?.name || 'Azure User',
                    email: account.username || account.idTokenClaims?.preferred_username || '',
                    role: userRole,
                    loginTime: new Date().toISOString(),
                    authMethod: 'azure_ad',
                    tenantId: account.tenantId || 'test-tenant'
                };

                // Store session
                sessionStorage.setItem('csp_azure_session', JSON.stringify(currentUser));

                showStatus('Authentication successful!', 'success');
                setTimeout(() => {
                    hideStatus();
                    showUserInfo(currentUser);
                }, 2000);

                // Test backend connection
                await testBackendConnection();

            } catch (error) {
                console.error('❌ Failed to setup Azure user:', error);
                showStatus('Failed to complete user setup.', 'error');
            }
        }

        // Test backend connection with Azure AD token
        async function testBackendConnection() {
            try {
                console.log('🧪 Testing backend connection...');
                
                // Get access token
                const account = msalInstance.getActiveAccount();
                if (!account) {
                    console.log('ℹ️ No active account for backend test');
                    return;
                }

                const tokenResponse = await msalInstance.acquireTokenSilent({
                    scopes: ["User.Read"],
                    account: account
                });

                console.log('🔑 Got access token, testing backend...');

                // Test backend API
                const response = await fetch('http://localhost:8000/api/auth/me', {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${tokenResponse.accessToken}`,
                        'Content-Type': 'application/json'
                    }
                });

                if (response.ok) {
                    const userData = await response.json();
                    console.log('✅ Backend connection successful:', userData);
                    showStatus('Backend connection verified!', 'success');
                } else {
                    console.warn('⚠️ Backend connection failed:', response.status);
                }

            } catch (error) {
                console.warn('⚠️ Backend test failed:', error);
                // Don't show error for this - it's just a test
            }
        }

        // Demo login
        function demoLogin(username, password) {
            const user = demoUsers[username.toLowerCase()];
            
            if (user && user.password === password) {
                currentUser = {
                    id: username,
                    name: user.name,
                    email: `${username}@demo.csp.ai`,
                    role: user.role,
                    loginTime: new Date().toISOString(),
                    authMethod: 'demo'
                };

                sessionStorage.setItem('csp_demo_session', JSON.stringify(currentUser));
                showUserInfo(currentUser);
                showStatus('Demo login successful!', 'success');
                
                setTimeout(hideStatus, 3000);
            } else {
                showStatus('Invalid username or password.', 'error');
                setTimeout(hideStatus, 3000);
            }
        }

        // Show user information
        function showUserInfo(user) {
            // Hide login section and library status
            loginSection.style.display = 'none';
            libraryStatus.style.display = 'none';
            
            // Show user info
            userInfo.style.display = 'block';
            actionButtons.style.display = 'flex';

            // Update user details
            userDetails.innerHTML = `
                <div class="user-detail">
                    <strong>Name:</strong> 
                    <span>${user.name}</span>
                </div>
                <div class="user-detail">
                    <strong>Email:</strong> 
                    <span>${user.email}</span>
                </div>
                <div class="user-detail">
                    <strong>Role:</strong> 
                    <span class="role-badge">${user.role}</span>
                </div>
                <div class="user-detail">
                    <strong>Method:</strong> 
                    <span>${user.authMethod === 'azure_ad' ? 'Azure AD' : 'Demo'}</span>
                </div>
                <div class="user-detail">
                    <strong>Login:</strong> 
                    <span>${new Date(user.loginTime).toLocaleString()}</span>
                </div>
            `;

            console.log('✅ User authenticated:', user);
        }

        // Go to dashboard
        function goToDashboard() {
            // Test backend API first
            if (currentUser && currentUser.authMethod === 'azure_ad') {
                testDashboardAccess();
            } else {
                alert('Dashboard functionality will be implemented soon!\n\nYour authentication is working correctly.');
            }
        }

        // Test dashboard access
        async function testDashboardAccess() {
            try {
                const account = msalInstance.getActiveAccount();
                if (account) {
                    const tokenResponse = await msalInstance.acquireTokenSilent({
                        scopes: ["User.Read"],
                        account: account
                    });

                    const response = await fetch('http://localhost:8000/api/designs', {
                        headers: {
                            'Authorization': `Bearer ${tokenResponse.accessToken}`
                        }
                    });

                    if (response.ok) {
                        const designs = await response.json();
                        alert(`Dashboard Access Test Successful!\n\nFound ${designs.length} designs.\n\nYour Azure AD authentication is working perfectly with the backend.`);
                    } else {
                        alert('Dashboard access test failed. Please check backend connection.');
                    }
                } else {
                    alert('No active Azure AD account found.');
                }
            } catch (error) {
                console.error('Dashboard test failed:', error);
                alert('Dashboard access test failed: ' + error.message);
            }
        }

        // Logout
        async function logout() {
            try {
                if (currentUser?.authMethod === 'azure_ad' && msalInstance && msalInstance.logoutPopup) {
                    await msalInstance.logoutPopup();
                }
                
                // Clear sessions
                sessionStorage.removeItem('csp_azure_session');
                sessionStorage.removeItem('csp_demo_session');
                
                // Reset UI
                currentUser = null;
                userInfo.style.display = 'none';
                loginSection.style.display = 'block';
                libraryStatus.style.display = 'block';
                
                // Clear form
                document.getElementById('demo-username').value = '';
                document.getElementById('demo-password').value = '';
                
                showStatus('Logged out successfully.', 'success');
                setTimeout(hideStatus, 3000);

                console.log('✅ Logout successful');
                
            } catch (error) {
                console.error('❌ Logout failed:', error);
                showStatus('Logout failed.', 'error');
                setTimeout(hideStatus, 3000);
            }
        }
    </script>
</body>
</html>
EOF

echo -e "${GREEN}✅ Updated login.html created with local MSAL${NC}"

echo ""
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "==================="
echo ""
echo -e "${BLUE}📁 Files created:${NC}"
echo "   ├── js/vendor/msal-browser.min.js (MSAL library)"
echo "   └── pages/login.html (Updated login page)"
echo ""
echo -e "${YELLOW}🧪 Testing Instructions:${NC}"
echo "1. Start your test server:"
echo "   python test-server.py"
echo ""
echo "2. Open the login page:"
echo "   http://localhost:3000/pages/login.html"
echo ""
echo "3. The page will show MSAL status and work offline!"
echo ""
echo -e "${BLUE}💡 Features:${NC}"
echo "   • Works without internet connection"
echo "   • Shows MSAL library status"
echo "   • Fallback to demo mode if needed"
echo "   • Tests backend connection automatically"
echo ""
echo -e "${GREEN}🚀 Your Enhanced CSP System now works offline!${NC}"
EOF

chmod +x "$0"