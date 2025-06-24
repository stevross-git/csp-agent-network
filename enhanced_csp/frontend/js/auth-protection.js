/**
 * Enhanced CSP System - Universal Local Authentication Protection
 * Simplified auth system without MSAL - Local authentication only
 */

class CSPAuthProtection {
    constructor(options = {}) {
        this.options = {
            loginPage: '/pages/login.html',
            sessionTimeout: 8 * 60 * 60 * 1000, // 8 hours
            apiBaseUrl: "http://localhost:8000",
            enableDebug: true,
            ...options
        };

        this.currentUser = null;
        this.sessionTimer = null;
        this.isInitialized = false;
        this.authCheckInProgress = false;

        // User roles and permissions
        this.USER_ROLES = {
            SUPER_ADMIN: 'super_admin',
            ADMIN: 'admin', 
            DEVELOPER: 'developer',
            ANALYST: 'analyst',
            USER: 'user'
        };

        this.PAGE_PERMISSIONS = {
            // Admin pages
            'settings.html': ['super_admin', 'admin'],
            'users.html': ['super_admin', 'admin'],
            'security.html': ['super_admin', 'admin'],
            'system.html': ['super_admin', 'admin'],
            'logs.html': ['super_admin', 'admin'],
            
            // Developer pages
            'ai-models.html': ['super_admin', 'admin', 'developer'],
            'api-explorer.html': ['super_admin', 'admin', 'developer'],
            'containers.html': ['super_admin', 'admin', 'developer'],
            
            // General access pages (all authenticated users)
            'dashboard.html': ['super_admin', 'admin', 'developer', 'analyst', 'user'],
            'profile.html': ['super_admin', 'admin', 'developer', 'analyst', 'user'],
            'notifications.html': ['super_admin', 'admin', 'developer', 'analyst', 'user'],
            'chat.html': ['super_admin', 'admin', 'developer', 'analyst', 'user']
        };

        // Initialize authentication
        this.init();
    }

    async init() {
        this.log('🔐 Initializing CSP Authentication Protection...');
        
        // Skip auth on login page
        if (this.isLoginPage()) {
            this.log('📄 Login page detected - skipping auth check');
            return;
        }

        try {
            this.authCheckInProgress = true;
            
            // Check for existing session
            const isAuthenticated = await this.checkExistingSession();
            
            if (!isAuthenticated) {
                this.log('❌ No valid session found - redirecting to login');
                this.redirectToLogin();
                return;
            }

            // User is authenticated, set up the environment
            await this.setupAuthenticatedEnvironment();
            
            this.isInitialized = true;
            this.authCheckInProgress = false;
            
            // Dispatch ready event
            this.dispatchAuthEvent('cspAuthReady', { user: this.currentUser });
            
        } catch (error) {
            this.log('❌ Auth initialization failed:', error);
            this.authCheckInProgress = false;
            this.handleAuthError(error);
        }
    }

    async checkExistingSession() {
        try {
            const token = this.getStoredToken();
            if (!token) {
                this.log('No stored token found');
                return false;
            }

            // Validate token with backend
            const response = await fetch(`${this.options.apiBaseUrl}/api/auth/validate`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const userData = await response.json();
                this.currentUser = {
                    ...userData.user,
                    token: token,
                    auth_method: 'local'
                };
                this.log('✅ Valid session found for user:', this.currentUser.email);
                return true;
            } else {
                this.log('❌ Token validation failed');
                this.clearStoredTokens();
                return false;
            }

        } catch (error) {
            this.log('❌ Session check failed:', error);
            this.clearStoredTokens();
            return false;
        }
    }

    async setupAuthenticatedEnvironment() {
        this.log('🎉 Setting up authenticated environment...');

        // Add user role to document
        document.documentElement.setAttribute('data-user-role', this.currentUser.role);
        document.documentElement.setAttribute('data-user-id', this.currentUser.id);

        // Check page-specific access
        if (!this.checkPageAccess()) {
            return; // Access denied page will be shown
        }

        // Add authentication header
        this.addAuthHeader();

        // Start session monitoring
        this.startSessionMonitoring();

        // Set up logout handlers
        this.setupLogoutHandlers();

        this.log('🎉 Authenticated environment setup complete');
    }

    checkPageAccess() {
        const currentPage = this.getCurrentPageName();
        const userRole = this.currentUser.role;

        const allowedRoles = this.PAGE_PERMISSIONS[currentPage];
        
        // If page has no specific restrictions, allow access
        if (!allowedRoles) {
            this.log(`📄 Page ${currentPage} has no restrictions - access granted`);
            return true;
        }

        // Check if user role is allowed
        if (allowedRoles.includes(userRole)) {
            this.log(`✅ Access granted to ${currentPage} for role: ${userRole}`);
            return true;
        }

        // Access denied
        this.log(`❌ Access denied to ${currentPage} for role: ${userRole}`);
        this.showAccessDenied(currentPage, userRole, allowedRoles);
        return false;
    }

    addAuthHeader() {
        // Remove existing header
        const existingHeader = document.getElementById('csp-auth-header');
        if (existingHeader) existingHeader.remove();

        const header = document.createElement('div');
        header.id = 'csp-auth-header';
        header.innerHTML = `
            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                        color: white; padding: 0.75rem 1.5rem; font-size: 0.9rem;
                        display: flex; justify-content: space-between; align-items: center;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); z-index: 1000;
                        position: fixed; top: 0; left: 0; right: 0;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-weight: 600;">🔐 CSP System</span>
                    <span style="opacity: 0.8;">Welcome, ${this.currentUser.full_name || this.currentUser.email}</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 0.25rem 0.5rem; 
                                border-radius: 12px; font-size: 0.8rem;">
                        ${this.currentUser.role.replace('_', ' ').toUpperCase()}
                    </span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <button onclick="CSPAuth.showProfile()" 
                            style="background: rgba(255,255,255,0.1); border: none;
                                   color: white; padding: 0.4rem 0.8rem; border-radius: 4px; 
                                   cursor: pointer; font-size: 0.8rem;">
                        👤 Profile
                    </button>
                    <button onclick="CSPAuth.logout()" 
                            style="background: rgba(220, 53, 69, 0.8); border: none;
                                   color: white; padding: 0.4rem 0.8rem; border-radius: 4px; 
                                   cursor: pointer; font-size: 0.8rem;">
                        🚪 Logout
                    </button>
                </div>
            </div>
        `;

        document.body.insertAdjacentElement('afterbegin', header);
        
        // Adjust page content for header
        if (!document.body.style.paddingTop) {
            document.body.style.paddingTop = '60px';
        }
    }

    showAccessDenied(page, userRole, requiredRoles) {
        document.body.innerHTML = `
            <div style="display: flex; justify-content: center; align-items: center; 
                        height: 100vh; background: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
                <div style="text-align: center; background: white; padding: 3rem; 
                            border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); max-width: 500px;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🚫</div>
                    <h2 style="color: #dc3545; margin-bottom: 1rem;">Access Denied</h2>
                    <p style="color: #6c757d; margin-bottom: 2rem;">
                        You don't have permission to access <strong>${page}</strong><br>
                        Your role: <strong>${userRole}</strong><br>
                        Required roles: <strong>${requiredRoles.join(', ')}</strong>
                    </p>
                    <div style="display: flex; gap: 1rem; justify-content: center;">
                        <button onclick="window.history.back()" 
                                style="background: #6c757d; color: white; border: none; 
                                       padding: 0.75rem 1.5rem; border-radius: 6px; cursor: pointer;">
                            ← Go Back
                        </button>
                        <button onclick="window.location.href='/pages/dashboard.html'" 
                                style="background: #007bff; color: white; border: none; 
                                       padding: 0.75rem 1.5rem; border-radius: 6px; cursor: pointer;">
                            📊 Dashboard
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    startSessionMonitoring() {
        // Clear existing timer
        if (this.sessionTimer) {
            clearInterval(this.sessionTimer);
        }

        // Check session every 5 minutes
        this.sessionTimer = setInterval(async () => {
            const isValid = await this.validateCurrentSession();
            if (!isValid) {
                this.log('❌ Session expired during monitoring');
                this.handleSessionExpiry();
            }
        }, 5 * 60 * 1000);
    }

    async validateCurrentSession() {
        try {
            const token = this.getStoredToken();
            if (!token) return false;

            const response = await fetch(`${this.options.apiBaseUrl}/api/auth/validate`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            return response.ok;
        } catch (error) {
            this.log('❌ Session validation error:', error);
            return false;
        }
    }

    setupLogoutHandlers() {
        // Handle page visibility change
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                // Page became visible, validate session
                this.validateCurrentSession().then(isValid => {
                    if (!isValid) {
                        this.handleSessionExpiry();
                    }
                });
            }
        });

        // Handle beforeunload
        window.addEventListener('beforeunload', () => {
            if (this.sessionTimer) {
                clearInterval(this.sessionTimer);
            }
        });
    }

    async logout() {
        try {
            this.log('🚪 Logging out...');

            const token = this.getStoredToken();
            if (token) {
                // Inform backend about logout
                try {
                    await fetch(`${this.options.apiBaseUrl}/api/auth/logout`, {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${token}`,
                            'Content-Type': 'application/json'
                        }
                    });
                } catch (error) {
                    this.log('❌ Backend logout error:', error);
                }
            }

            // Clear local state
            this.clearStoredTokens();
            this.currentUser = null;
            this.isInitialized = false;
            
            if (this.sessionTimer) {
                clearInterval(this.sessionTimer);
                this.sessionTimer = null;
            }

            // Redirect to login
            this.redirectToLogin();

        } catch (error) {
            this.log('❌ Logout error:', error);
            // Even if there's an error, clear local state and redirect
            this.clearStoredTokens();
            this.redirectToLogin();
        }
    }

    showProfile() {
        // Simple profile modal
        const modal = document.createElement('div');
        modal.innerHTML = `
            <div style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; 
                        background: rgba(0,0,0,0.5); z-index: 10000; display: flex; 
                        justify-content: center; align-items: center;">
                <div style="background: white; padding: 2rem; border-radius: 12px; 
                            box-shadow: 0 4px 12px rgba(0,0,0,0.3); max-width: 400px; width: 90%;">
                    <h3 style="margin: 0 0 1rem 0;">👤 User Profile</h3>
                    <div style="margin-bottom: 1rem;">
                        <strong>Name:</strong> ${this.currentUser.full_name || 'N/A'}<br>
                        <strong>Email:</strong> ${this.currentUser.email}<br>
                        <strong>Role:</strong> ${this.currentUser.role}<br>
                        <strong>Auth Method:</strong> Local Authentication
                    </div>
                    <button onclick="this.parentElement.parentElement.remove()"
                            style="background: #007bff; color: white; border: none; 
                                   padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer;">
                        Close
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    // Utility methods
    isLoginPage() {
        const path = window.location.pathname;
        return path.includes('login.html') || path.endsWith('/login') || path === '/pages/login.html';
    }

    getCurrentPageName() {
        return window.location.pathname.split('/').pop() || 'index.html';
    }

    getStoredToken() {
        return localStorage.getItem('local_access_token') || sessionStorage.getItem('local_access_token');
    }

    clearStoredTokens() {
        localStorage.removeItem('local_access_token');
        localStorage.removeItem('local_refresh_token');
        sessionStorage.removeItem('local_access_token');
        sessionStorage.removeItem('local_refresh_token');
    }

    redirectToLogin() {
        const currentUrl = encodeURIComponent(window.location.pathname + window.location.search);
        window.location.href = `${this.options.loginPage}?redirect=${currentUrl}`;
    }

    handleSessionExpiry() {
        this.log('⏰ Session expired');
        this.dispatchAuthEvent('cspSessionExpired');
        
        // Show notification
        if (window.toast) {
            window.toast.warning('Session Expired', 'Please log in again.');
        } else {
            alert('Your session has expired. Please log in again.');
        }
        
        setTimeout(() => this.redirectToLogin(), 2000);
    }

    handleAuthError(error) {
        this.log('❌ Authentication error:', error);
        this.dispatchAuthEvent('cspAuthError', error);
        this.redirectToLogin();
    }

    dispatchAuthEvent(eventName, detail = null) {
        const event = new CustomEvent(eventName, { detail });
        document.dispatchEvent(event);
    }

    log(...args) {
        if (this.options.enableDebug) {
            console.log('[CSPAuth]', ...args);
        }
    }

    // Static methods for global access
    static getInstance() {
        return window.CSPAuth;
    }

    static getCurrentUser() {
        return window.CSPAuth?.currentUser || null;
    }

    static hasRole(role) {
        const user = CSPAuthProtection.getCurrentUser();
        return user?.role === role;
    }

    static hasAnyRole(roles) {
        const user = CSPAuthProtection.getCurrentUser();
        return roles.includes(user?.role);
    }

    static async makeAuthenticatedRequest(url, options = {}) {
        const auth = CSPAuthProtection.getInstance();
        if (!auth || !auth.currentUser) {
            throw new Error('User not authenticated');
        }

        const token = auth.getStoredToken();
        const headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
            ...options.headers
        };

        return fetch(url, { ...options, headers });
    }
}

// Initialize authentication when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Create global instance
    window.CSPAuth = new CSPAuthProtection();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CSPAuthProtection;
}