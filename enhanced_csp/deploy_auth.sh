#!/bin/bash

# ==============================================================================
# CSP Agent Network - Authentication Wrapper Deployment Script
# ==============================================================================
# This script automatically applies authentication wrapper to all frontend pages

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/home/mate/PAIN/csp-agent-network/csp-agent-network-1"
ENHANCED_CSP_DIR="$PROJECT_ROOT/enhanced_csp"
FRONTEND_DIR="$ENHANCED_CSP_DIR/frontend"
PAGES_DIR="$FRONTEND_DIR/pages"
JS_DIR="$FRONTEND_DIR/js"
CSS_DIR="$FRONTEND_DIR/css"

echo -e "${BLUE}🔐 CSP Agent Network - Authentication Wrapper Deployment${NC}"
echo "=================================================================="
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if directories exist
check_directories() {
    print_status "Checking directory structure..."
    
    if [ ! -d "$PROJECT_ROOT" ]; then
        print_error "Project root directory not found: $PROJECT_ROOT"
        exit 1
    fi
    
    if [ ! -d "$PAGES_DIR" ]; then
        print_error "Frontend pages directory not found: $PAGES_DIR"
        print_status "Creating directory structure..."
        mkdir -p "$PAGES_DIR"
    fi
    
    if [ ! -d "$JS_DIR" ]; then
        print_status "Creating js directory: $JS_DIR"
        mkdir -p "$JS_DIR"
    fi
    
    if [ ! -d "$CSS_DIR" ]; then
        print_status "Creating css directory: $CSS_DIR"
        mkdir -p "$CSS_DIR"
    fi
    
    print_status "✅ Directory structure verified"
}

# Create the universal auth wrapper JavaScript file
create_auth_wrapper() {
    print_status "Creating universal authentication wrapper..."
    
    cat > "$JS_DIR/auth-wrapper.js" << 'EOF'
/**
 * CSP Agent Network - Universal Authentication Wrapper
 * Protects all frontend pages with authentication
 */

class CSPUniversalAuth {
    constructor(options = {}) {
        this.options = {
            loginPage: '/enhanced_csp/frontend/pages/login.html',
            securityPage: '/enhanced_csp/frontend/pages/security.html',
            sessionTimeout: 24 * 60 * 60 * 1000, // 24 hours
            checkInterval: 5 * 60 * 1000, // 5 minutes
            ...options
        };
        
        this.sessionTimer = null;
        this.initialized = false;
        
        console.log('🔐 CSP Universal Auth initializing...');
        this.init();
    }

    init() {
        // Skip auth check on login page
        if (this.isLoginPage()) {
            console.log('📄 Login page detected - skipping auth check');
            return;
        }
        
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.performAuthCheck());
        } else {
            this.performAuthCheck();
        }
        
        this.setupEventListeners();
        this.initialized = true;
    }

    isLoginPage() {
        return window.location.pathname.includes('login.html');
    }

    performAuthCheck() {
        console.log('🔍 Performing authentication check...');
        
        const session = this.getSession();
        
        if (!session || !this.isValidSession(session)) {
            console.log('❌ Authentication failed - redirecting to login');
            this.redirectToLogin();
            return false;
        }
        
        console.log('✅ Authentication successful:', session.username);
        this.setupAuthenticatedEnvironment(session);
        this.showPageContent();
        return true;
    }

    getSession() {
        try {
            const sessionData = localStorage.getItem('csp_session') || 
                               sessionStorage.getItem('csp_session');
            return sessionData ? JSON.parse(sessionData) : null;
        } catch (e) {
            console.error('❌ Invalid session data:', e);
            this.clearSession();
            return null;
        }
    }

    isValidSession(session) {
        if (!session?.loginTime) {
            console.log('❌ No login time in session');
            return false;
        }
        
        const loginTime = new Date(session.loginTime);
        const now = new Date();
        const age = now - loginTime;
        const maxAge = session.rememberMe ? (30 * 24 * 60 * 60 * 1000) : this.options.sessionTimeout;
        
        if (age > maxAge) {
            console.log('⏰ Session expired');
            return false;
        }
        
        return true;
    }

    setupAuthenticatedEnvironment(session) {
        // Add authentication header
        this.addAuthHeader(session);
        
        // Check page access permissions
        if (!this.checkPageAccess(session)) {
            return;
        }
        
        // Start session monitoring
        this.startSessionMonitoring(session);
        
        // Apply role-based UI modifications
        this.applyRoleBasedUI(session);
    }

    addAuthHeader(session) {
        // Remove existing header if present
        const existingHeader = document.getElementById('csp-auth-header');
        if (existingHeader) {
            existingHeader.remove();
        }
        
        const header = document.createElement('div');
        header.id = 'csp-auth-header';
        header.innerHTML = `
            <div style="position: fixed; top: 0; left: 0; right: 0; z-index: 99999; 
                        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                        color: white; padding: 0.75rem 1rem; font-size: 0.9rem;
                        display: flex; justify-content: space-between; align-items: center;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-weight: bold;">🛡️ CSP Agent Network</span>
                    <span style="background: rgba(255,255,255,0.25); padding: 0.25rem 0.6rem; 
                                border-radius: 15px; font-size: 0.75rem; font-weight: bold;">
                        ${(session.role || 'USER').toUpperCase()}
                    </span>
                    <span>👤 ${session.name || session.username}</span>
                    <span id="session-timer" style="opacity: 0.9; font-size: 0.8rem; font-family: monospace;"></span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <button onclick="window.location.href='${this.options.securityPage}'" 
                            style="background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3);
                                   color: white; padding: 0.3rem 0.8rem; border-radius: 4px; 
                                   cursor: pointer; font-size: 0.8rem; transition: all 0.2s;">
                        🛡️ Security
                    </button>
                    <button onclick="CSPAuth.logout()" 
                            style="background: rgba(220, 53, 69, 0.8); border: 1px solid rgba(220, 53, 69, 0.6);
                                   color: white; padding: 0.3rem 0.8rem; border-radius: 4px; 
                                   cursor: pointer; font-size: 0.8rem; transition: all 0.2s;">
                        🚪 Logout
                    </button>
                </div>
            </div>
        `;
        
        document.body.insertBefore(header, document.body.firstChild);
        
        // Adjust page content for header
        if (!document.body.style.paddingTop) {
            document.body.style.paddingTop = '65px';
        }
        
        this.startSessionTimer(session);
    }

    checkPageAccess(session) {
        const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        const userRole = session.role?.toLowerCase() || 'viewer';
        
        // Define page access rules (you can modify these)
        const accessRules = {
            'security.html': ['admin', 'security'],
            'settings.html': ['admin'],
            'users.html': ['admin'],
            'system.html': ['admin'],
            'logs.html': ['admin', 'security'],
            // Add more restrictive pages here
        };
        
        const requiredRoles = accessRules[currentPage];
        if (requiredRoles && !requiredRoles.includes(userRole)) {
            console.log(`🚫 Access denied to ${currentPage} for role ${userRole}`);
            this.showAccessDenied(currentPage, userRole, requiredRoles);
            return false;
        }
        
        return true;
    }

    showAccessDenied(page, userRole, requiredRoles) {
        document.body.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; 
                        min-height: 100vh; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 2rem;">
                <div style="text-align: center; background: white; padding: 3rem; 
                           border-radius: 20px; box-shadow: 0 20px 50px rgba(0,0,0,0.15);
                           max-width: 500px; width: 100%;">
                    <div style="font-size: 5rem; margin-bottom: 1rem; animation: pulse 2s infinite;">🚫</div>
                    <h1 style="color: #e74c3c; margin-bottom: 1rem; font-size: 2rem;">Access Denied</h1>
                    <p style="color: #666; margin-bottom: 1rem; font-size: 1.1rem;">
                        You don't have permission to access <strong>${page}</strong>
                    </p>
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                        <p style="margin: 0; color: #495057; font-size: 0.9rem;">
                            <strong>Your Role:</strong> ${userRole.toUpperCase()}<br>
                            <strong>Required Roles:</strong> ${requiredRoles.join(', ').toUpperCase()}
                        </p>
                    </div>
                    <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 2rem;">
                        <button onclick="history.back()" 
                                style="background: #6c757d; color: white; border: none; 
                                       padding: 0.75rem 1.5rem; border-radius: 8px; cursor: pointer;
                                       font-size: 1rem; transition: all 0.2s;">
                            ← Go Back
                        </button>
                        <button onclick="window.location.href='/enhanced_csp/frontend/pages/dashboard.html'" 
                                style="background: #007bff; color: white; border: none; 
                                       padding: 0.75rem 1.5rem; border-radius: 8px; cursor: pointer;
                                       font-size: 1rem; transition: all 0.2s;">
                            🏠 Dashboard
                        </button>
                    </div>
                </div>
            </div>
            <style>
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                    100% { transform: scale(1); }
                }
            </style>
        `;
    }

    applyRoleBasedUI(session) {
        const userRole = session.role?.toLowerCase() || 'viewer';
        
        // Add role-specific CSS classes
        document.documentElement.setAttribute('data-user-role', userRole);
        
        // Hide/show elements based on role
        document.querySelectorAll('[data-require-role]').forEach(element => {
            const requiredRoles = element.getAttribute('data-require-role').split(',');
            if (!requiredRoles.includes(userRole)) {
                element.style.display = 'none';
            }
        });
    }

    startSessionTimer(session) {
        const startTime = new Date(session.loginTime);
        
        this.sessionTimer = setInterval(() => {
            const now = new Date();
            const elapsed = Math.floor((now - startTime) / 1000);
            const hours = Math.floor(elapsed / 3600);
            const minutes = Math.floor((elapsed % 3600) / 60);
            
            const timer = document.getElementById('session-timer');
            if (timer) {
                timer.textContent = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }

    startSessionMonitoring(session) {
        // Periodic session validation
        setInterval(() => {
            if (!this.isValidSession(session)) {
                console.log('🔄 Periodic validation failed - session expired');
                this.handleSessionExpiry();
            }
        }, this.options.checkInterval);
        
        // Warn before expiry
        const maxAge = session.rememberMe ? (30 * 24 * 60 * 60 * 1000) : this.options.sessionTimeout;
        const warningTime = maxAge - (5 * 60 * 1000); // 5 minutes before expiry
        
        setTimeout(() => {
            if (this.isValidSession(session)) {
                this.showExpiryWarning();
            }
        }, warningTime);
    }

    showExpiryWarning() {
        if (confirm('⚠️ Your session will expire in 5 minutes. Would you like to extend it?')) {
            this.extendSession();
        }
    }

    extendSession() {
        const session = this.getSession();
        if (session) {
            session.loginTime = new Date().toISOString();
            
            if (session.rememberMe) {
                localStorage.setItem('csp_session', JSON.stringify(session));
            } else {
                sessionStorage.setItem('csp_session', JSON.stringify(session));
            }
            
            console.log('✅ Session extended');
        }
    }

    setupEventListeners() {
        // Monitor storage changes (other tabs)
        window.addEventListener('storage', (e) => {
            if (e.key === 'csp_session') {
                console.log('📱 Session changed in another tab');
                setTimeout(() => this.performAuthCheck(), 100);
            }
        });

        // Monitor page visibility
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.initialized) {
                this.performAuthCheck();
            }
        });

        // Security monitoring
        this.setupSecurityMonitoring();
    }

    setupSecurityMonitoring() {
        // Monitor for suspicious activity
        let suspiciousActivity = 0;
        
        // Monitor for rapid-fire requests (potential attack)
        let requestCount = 0;
        setInterval(() => {
            if (requestCount > 100) { // More than 100 requests per minute
                console.warn('🚨 Suspicious activity detected: High request frequency');
                suspiciousActivity++;
            }
            requestCount = 0;
        }, 60000);
        
        // Monitor original fetch to count requests
        const originalFetch = window.fetch;
        window.fetch = function(...args) {
            requestCount++;
            return originalFetch.apply(this, args);
        };
    }

    showPageContent() {
        // Make page content visible
        document.body.style.opacity = '1';
        document.body.classList.add('authenticated');
        
        // Dispatch custom event for page-specific initialization
        document.dispatchEvent(new CustomEvent('cspAuthReady', {
            detail: { session: this.getSession() }
        }));
    }

    handleSessionExpiry() {
        this.clearSession();
        alert('🔒 Your session has expired. Please log in again.');
        this.redirectToLogin();
    }

    logout() {
        if (confirm('🔒 Are you sure you want to logout?')) {
            console.log('🚪 User logout initiated');
            this.clearSession();
            window.location.href = this.options.loginPage;
        }
    }

    clearSession() {
        localStorage.removeItem('csp_session');
        sessionStorage.removeItem('csp_session');
        
        if (this.sessionTimer) {
            clearInterval(this.sessionTimer);
            this.sessionTimer = null;
        }
        
        console.log('🗑️ Session cleared');
    }

    redirectToLogin() {
        // Store current page for redirect after login
        sessionStorage.setItem('csp_redirect_after_login', window.location.href);
        window.location.href = this.options.loginPage;
    }
}

// Initialize authentication when script loads
console.log('🔐 CSP Auth Wrapper script loaded');

// Wait for DOM to initialize auth
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.CSPAuth = new CSPUniversalAuth();
    });
} else {
    window.CSPAuth = new CSPUniversalAuth();
}
EOF

    print_status "✅ Authentication wrapper created: $JS_DIR/auth-wrapper.js"
}

# Create the auth wrapper CSS file
create_auth_css() {
    print_status "Creating authentication styles..."
    
    cat > "$CSS_DIR/auth-wrapper.css" << 'EOF'
/* CSP Authentication Wrapper Styles */

/* Hide content until authenticated */
body {
    opacity: 0;
    transition: opacity 0.3s ease-in-out;
}

body.authenticated {
    opacity: 1;
}

/* Auth header styles */
#csp-auth-header button:hover {
    background: rgba(255, 255, 255, 0.35) !important;
    transform: translateY(-1px);
}

/* Role-based visibility */
[data-require-role] {
    transition: opacity 0.2s ease-in-out;
}

/* Role-specific styles */
html[data-user-role="admin"] .admin-only { display: block !important; }
html[data-user-role="designer"] .designer-only { display: block !important; }
html[data-user-role="viewer"] .viewer-only { display: block !important; }

.admin-only, .designer-only, .viewer-only {
    display: none;
}

/* Loading overlay */
.csp-auth-loading {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(30, 60, 114, 0.95);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 99999;
    color: white;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.csp-auth-loading .spinner {
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top: 3px solid white;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin-right: 1rem;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
EOF

    print_status "✅ Authentication styles created: $CSS_DIR/auth-wrapper.css"
}

# Function to backup a file
backup_file() {
    local file="$1"
    local backup="${file}.backup.$(date +%Y%m%d_%H%M%S)"
    
    if [ -f "$file" ]; then
        cp "$file" "$backup"
        print_status "📋 Backed up: $(basename "$file") -> $(basename "$backup")"
    fi
}

# Function to apply auth wrapper to a single HTML file
apply_auth_to_file() {
    local file="$1"
    local filename=$(basename "$file")
    
    # Skip login.html
    if [[ "$filename" == "login.html" ]]; then
        print_status "⏭️  Skipping login.html (no auth needed)"
        return
    fi
    
    print_status "🔧 Processing: $filename"
    
    # Backup the original file
    backup_file "$file"
    
    # Check if already has auth wrapper
    if grep -q "auth-wrapper.js" "$file"; then
        print_warning "⚠️  $filename already has auth wrapper - skipping"
        return
    fi
    
    # Create temporary file for modifications
    local temp_file=$(mktemp)
    
    # Process the file
    while IFS= read -r line; do
        echo "$line" >> "$temp_file"
        
        # Add auth scripts before closing head tag
        if [[ "$line" == *"</head>"* ]]; then
            cat >> "$temp_file" << 'EOF'
    <!-- CSP Authentication Wrapper -->
    <script src="../js/auth-wrapper.js"></script>
    <link rel="stylesheet" href="../css/auth-wrapper.css">
EOF
        fi
        
        # Add auth completion script before closing body tag
        if [[ "$line" == *"</body>"* ]]; then
            cat >> "$temp_file" << 'EOF'
    <!-- Authentication Ready Handler -->
    <script>
        document.addEventListener('cspAuthReady', (event) => {
            console.log('🎉 Page authenticated and ready');
            // Page-specific initialization can go here
        });
    </script>
EOF
        fi
    done < "$file"
    
    # Replace original file with modified version
    mv "$temp_file" "$file"
    
    print_status "✅ Updated: $filename"
}

# Main function to apply auth to all pages
apply_auth_to_all_pages() {
    print_status "Applying authentication wrapper to all HTML pages..."
    
    local count=0
    
    # Process all HTML files in pages directory
    for file in "$PAGES_DIR"/*.html; do
        if [ -f "$file" ]; then
            apply_auth_to_file "$file"
            ((count++))
        fi
    done
    
    print_status "✅ Processed $count HTML files"
}

# Function to update login.html with redirect handling
update_login_page() {
    print_status "Updating login.html with redirect handling..."
    
    local login_file="$PAGES_DIR/login.html"
    
    if [ ! -f "$login_file" ]; then
        print_warning "⚠️  login.html not found - creating basic version"
        create_basic_login_page
        return
    fi
    
    # Check if redirect handling already exists
    if grep -q "csp_redirect_after_login" "$login_file"; then
        print_status "✅ login.html already has redirect handling"
        return
    fi
    
    # Backup original
    backup_file "$login_file"
    
    # Add redirect handling script before closing body tag
    sed -i '/<\/body>/i \    <!-- Redirect Handling for Auth Wrapper -->\n    <script>\n        function handleRedirectAfterLogin(sessionData) {\n            const redirectUrl = sessionStorage.getItem("csp_redirect_after_login") || "dashboard.html";\n            sessionStorage.removeItem("csp_redirect_after_login");\n            console.log("🔄 Redirecting to:", redirectUrl);\n            setTimeout(() => {\n                window.location.href = redirectUrl;\n            }, 1000);\n        }\n        \n        // Update existing login success handler to use redirect\n        const originalRedirect = window.redirectToAdminPortal;\n        if (originalRedirect) {\n            window.redirectToAdminPortal = function(sessionData) {\n                handleRedirectAfterLogin(sessionData);\n            };\n        }\n    </script>' "$login_file"
    
    print_status "✅ Updated login.html with redirect handling"
}

# Create a basic login page if it doesn't exist
create_basic_login_page() {
    local login_file="$PAGES_DIR/login.html"
    
    cat > "$login_file" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSP Agent Network - Login</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-container {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            max-width: 400px;
            width: 90%;
        }
        .login-title {
            text-align: center;
            color: #1e3c72;
            margin-bottom: 2rem;
        }
        .form-group {
            margin-bottom: 1rem;
        }
        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            color: #333;
        }
        .form-input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 1rem;
        }
        .login-btn {
            width: 100%;
            background: #1e3c72;
            color: white;
            border: none;
            padding: 0.75rem;
            border-radius: 6px;
            font-size: 1rem;
            cursor: pointer;
            margin-top: 1rem;
        }
        .login-btn:hover {
            background: #2a5298;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h1 class="login-title">🔐 CSP Agent Network</h1>
        <form id="login-form">
            <div class="form-group">
                <label class="form-label" for="username">Username</label>
                <input type="text" id="username" class="form-input" required>
            </div>
            <div class="form-group">
                <label class="form-label" for="password">Password</label>
                <input type="password" id="password" class="form-input" required>
            </div>
            <button type="submit" class="login-btn">Login</button>
        </form>
    </div>

    <script>
        // Demo users (replace with your actual authentication)
        const users = {
            'admin': { password: 'csp2025!', role: 'admin', name: 'Administrator' },
            'developer': { password: 'dev123!', role: 'designer', name: 'Developer' },
            'user': { password: 'user123!', role: 'viewer', name: 'User' }
        };

        document.getElementById('login-form').addEventListener('submit', (e) => {
            e.preventDefault();
            
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            const user = users[username.toLowerCase()];
            
            if (user && user.password === password) {
                const sessionData = {
                    username: username,
                    role: user.role,
                    name: user.name,
                    loginTime: new Date().toISOString(),
                    rememberMe: false
                };
                
                sessionStorage.setItem('csp_session', JSON.stringify(sessionData));
                
                handleRedirectAfterLogin(sessionData);
            } else {
                alert('Invalid credentials');
            }
        });

        function handleRedirectAfterLogin(sessionData) {
            const redirectUrl = sessionStorage.getItem('csp_redirect_after_login') || 'dashboard.html';
            sessionStorage.removeItem('csp_redirect_after_login');
            console.log('🔄 Redirecting to:', redirectUrl);
            window.location.href = redirectUrl;
        }
    </script>
</body>
</html>
EOF

    print_status "✅ Created basic login page: $login_file"
}

# Function to create a test page for verification
create_test_page() {
    print_status "Creating test page for verification..."
    
    cat > "$PAGES_DIR/test-auth.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSP Agent Network - Auth Test</title>
</head>
<body>
    <div style="padding: 2rem; font-family: Arial, sans-serif;">
        <h1>🧪 Authentication Test Page</h1>
        <p>If you can see this content, authentication is working!</p>
        
        <div style="background: #f0f0f0; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h3>Session Information</h3>
            <div id="session-info">Loading...</div>
        </div>
        
        <div style="margin: 1rem 0;">
            <h3>Role-Based Content Test</h3>
            <div data-require-role="admin" class="admin-only" style="padding: 1rem; background: #ffe6e6; border-radius: 4px; margin: 0.5rem 0;">
                🔴 <strong>Admin Only:</strong> This content is only visible to administrators
            </div>
            <div data-require-role="designer" class="designer-only" style="padding: 1rem; background: #e6f3ff; border-radius: 4px; margin: 0.5rem 0;">
                🔵 <strong>Designer Only:</strong> This content is only visible to designers
            </div>
            <div data-require-role="viewer" class="viewer-only" style="padding: 1rem; background: #e6ffe6; border-radius: 4px; margin: 0.5rem 0;">
                🟢 <strong>Viewer Only:</strong> This content is only visible to viewers
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('cspAuthReady', (event) => {
            const session = event.detail.session;
            
            document.getElementById('session-info').innerHTML = `
                <strong>Username:</strong> ${session.username}<br>
                <strong>Role:</strong> ${session.role}<br>
                <strong>Name:</strong> ${session.name}<br>
                <strong>Login Time:</strong> ${new Date(session.loginTime).toLocaleString()}
            `;
            
            console.log('🎉 Test page authenticated successfully!', session);
        });
    </script>
</body>
</html>
EOF

    print_status "✅ Created test page: $PAGES_DIR/test-auth.html"
}

# Function to show deployment summary
show_summary() {
    echo ""
    echo -e "${GREEN}=================================================================="
    echo -e "🎉 CSP Authentication Wrapper Deployment Complete!"
    echo -e "==================================================================${NC}"
    echo ""
    echo -e "${BLUE}Files Created:${NC}"
    echo "  📄 $JS_DIR/auth-wrapper.js"
    echo "  🎨 $CSS_DIR/auth-wrapper.css"
    echo "  🧪 $PAGES_DIR/test-auth.html"
    echo ""
    echo -e "${BLUE}Pages Modified:${NC}"
    local count=$(find "$PAGES_DIR" -name "*.html" -not -name "login.html" | wc -l)
    echo "  🔐 $count HTML pages now have authentication wrapper"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "  1. 🌐 Start your web server"
    echo "  2. 🧪 Visit test-auth.html to verify authentication"
    echo "  3. 🔐 Try accessing other pages without logging in"
    echo "  4. 👤 Test with different user roles (admin, designer, viewer)"
    echo ""
    echo -e "${BLUE}Default Test Credentials:${NC}"
    echo "  👑 admin / csp2025! (Administrator)"
    echo "  🛠️  developer / dev123! (Designer)" 
    echo "  👁️  user / user123! (Viewer)"
    echo ""
    echo -e "${YELLOW}⚠️  Security Notes:${NC}"
    echo "  - Change default passwords in production"
    echo "  - Configure SSL/HTTPS for production use"
    echo "  - Review role permissions in auth-wrapper.js"
    echo "  - Monitor authentication logs in security.html"
    echo ""
    echo -e "${GREEN}✅ All frontend pages are now protected!${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}Starting CSP Authentication Wrapper deployment...${NC}"
    echo ""
    
    check_directories
    create_auth_wrapper
    create_auth_css
    apply_auth_to_all_pages
    update_login_page
    create_test_page
    
    show_summary
}

# Run main function
main "$@"