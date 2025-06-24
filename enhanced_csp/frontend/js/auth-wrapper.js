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
        this.initializationPromise = null;
        this.currentUser = null;
        
        console.log('🔐 CSP Universal Auth initializing...');
        this.init();
    }

    async init() {
        // Prevent multiple initialization attempts
        if (this.initializationPromise) {
            return this.initializationPromise;
        }

        this.initializationPromise = this._doInit();
        return this.initializationPromise;
    }

    async _doInit() {
        try {
            // Skip auth check on login page
            if (this.isLoginPage()) {
                console.log('📄 Login page detected - skipping auth check');
                this.initialized = true;
                return;
            }
            
            // Wait for DOM to be ready
            if (document.readyState === 'loading') {
                await new Promise(resolve => {
                    document.addEventListener('DOMContentLoaded', resolve);
                });
            }
            
            // Perform authentication check
            const isAuthenticated = await this.performAuthCheck();
            
            if (isAuthenticated) {
                this.setupAuthenticatedEnvironment(this.currentUser);
                this.showPageContent();
                this.setupEventListeners();
            }
            
            this.initialized = true;
            
            // Make available globally after successful initialization
            window.CSPAuth = this;
            window.unifiedAuth = this;
            
            console.log('✅ CSP Universal Auth initialized successfully');
        } catch (error) {
            console.error('❌ CSP Universal Auth initialization failed:', error);
            this.initialized = true; // Mark as initialized to prevent hanging
            throw error;
        }
    }

    async waitForInit() {
        if (this.initialized) return;
        if (this.initializationPromise) {
            await this.initializationPromise;
        }
    }

    isLoginPage() {
        return window.location.pathname.includes('login.html');
    }

    async performAuthCheck() {
        console.log('🔍 Performing authentication check...');
        
        try {
            const session = await this.getSession();
            
            if (!session || !this.isValidSession(session)) {
                console.log('❌ Authentication failed - redirecting to login');
                this.redirectToLogin();
                return false;
            }
            
            this.currentUser = session;
            console.log('✅ Authentication successful:', session.username || session.email);
            return true;
        } catch (error) {
            console.error('❌ Auth check failed:', error);
            this.redirectToLogin();
            return false;
        }
    }

    async getSession() {
        try {
            // Check for Azure AD session first
            if (typeof msal !== 'undefined') {
                const msalInstance = new msal.PublicClientApplication({
                    auth: {
                        clientId: "53537e30-ae6b-48f7-9c7c-4db20fc27850",
                        authority: "https://login.microsoftonline.com/622a5fe0-fac1-4213-9cf7-d5f6defdf4c4",
                        redirectUri: window.location.origin,
                    },
                    cache: {
                        cacheLocation: "sessionStorage",
                        storeAuthStateInCookie: false,
                    }
                });
                
                await msalInstance.initialize();
                const account = msalInstance.getActiveAccount();
                
                if (account) {
                    return {
                        id: account.homeAccountId,
                        name: account.name,
                        email: account.username,
                        username: account.username,
                        role: this.mapEmailToRole(account.username),
                        authMethod: 'azure_ad',
                        loginTime: new Date().toISOString()
                    };
                }
            }

            // Check stored Azure session
            const azureSession = sessionStorage.getItem('csp_azure_session');
            if (azureSession) {
                return JSON.parse(azureSession);
            }

            // Check demo session
            const demoSession = sessionStorage.getItem('csp_demo_session');
            if (demoSession) {
                return JSON.parse(demoSession);
            }

            // Check legacy session storage
            const legacySession = localStorage.getItem('csp_session') || 
                                sessionStorage.getItem('csp_session');
            if (legacySession) {
                return JSON.parse(legacySession);
            }

            return null;
        } catch (error) {
            console.error('Error getting session:', error);
            return null;
        }
    }

    mapEmailToRole(email) {
        if (email.includes('admin')) return 'admin';
        if (email.includes('dev') || email.includes('developer')) return 'developer';
        if (email.includes('analyst')) return 'analyst';
        return 'user';
    }

    isValidSession(session) {
        if (!session || !session.username && !session.email) {
            return false;
        }

        // Check session timeout (if loginTime exists)
        if (session.loginTime) {
            const loginTime = new Date(session.loginTime).getTime();
            const now = Date.now();
            const sessionAge = now - loginTime;
            
            if (sessionAge > this.options.sessionTimeout) {
                console.log('⏰ Session expired due to timeout');
                return false;
            }
        }

        return true;
    }

    setupAuthenticatedEnvironment(user) {
        console.log('🎉 Setting up authenticated environment for:', user.username || user.email);
        
        // Add user role to HTML element for CSS styling
        document.documentElement.setAttribute('data-user-role', user.role || 'user');
        
        // Add authentication class to body
        document.body.classList.add('authenticated');
        
        // Setup session monitoring
        this.setupSessionMonitoring();
        
        // Add user info header
        this.addUserInfoHeader(user);
        
        // Setup role-based visibility
        this.setupRoleBasedVisibility(user.role || 'user');
    }

    addUserInfoHeader(user) {
        // Remove existing header if present
        const existingHeader = document.getElementById('csp-auth-header');
        if (existingHeader) {
            existingHeader.remove();
        }

        const roleEmoji = {
            'admin': '👑',
            'developer': '🛠️',
            'analyst': '📊',
            'user': '👤'
        };

        const header = document.createElement('div');
        header.id = 'csp-auth-header';
        header.style.cssText = `
            position: fixed;
            top: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0 0 0 8px;
            z-index: 10000;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            backdrop-filter: blur(10px);
            font-family: 'Segoe UI', sans-serif;
        `;

        header.innerHTML = `
            <span>${roleEmoji[user.role] || '👤'} ${user.name || user.username || user.email}</span>
            <span style="color: #ccc;">|</span>
            <span style="color: #4CAF50;">${(user.role || 'user').toUpperCase()}</span>
            <button id="csp-logout-btn" style="
                background: #f44336;
                color: white;
                border: none;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.8rem;
                transition: background 0.2s;
            ">Sign Out</button>
        `;

        document.body.appendChild(header);

        // Add logout functionality
        const logoutBtn = document.getElementById('csp-logout-btn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.logout());
            logoutBtn.addEventListener('mouseenter', (e) => {
                e.target.style.background = '#d32f2f';
            });
            logoutBtn.addEventListener('mouseleave', (e) => {
                e.target.style.background = '#f44336';
            });
        }
    }

    setupRoleBasedVisibility(userRole) {
        const roleHierarchy = {
            'user': 1,
            'analyst': 2,
            'developer': 3,
            'admin': 4,
            'super_admin': 5
        };

        const userLevel = roleHierarchy[userRole] || 1;

        // Hide elements that require higher roles
        document.querySelectorAll('[data-require-role]').forEach(element => {
            const requiredRole = element.getAttribute('data-require-role');
            const requiredLevel = roleHierarchy[requiredRole] || 1;
            
            if (userLevel < requiredLevel) {
                element.style.display = 'none';
            }
        });

        // Show elements for specific roles
        document.querySelectorAll(`[data-show-for-role]`).forEach(element => {
            const showForRole = element.getAttribute('data-show-for-role');
            if (showForRole !== userRole) {
                element.style.display = 'none';
            }
        });
    }

    setupSessionMonitoring() {
        // Clear existing timer
        if (this.sessionTimer) {
            clearInterval(this.sessionTimer);
        }

        // Check session validity periodically
        this.sessionTimer = setInterval(async () => {
            const session = await this.getSession();
            if (!session || !this.isValidSession(session)) {
                console.warn('⚠️ Session validation failed during monitoring');
                this.handleSessionExpiry();
            }
        }, this.options.checkInterval);
    }

    setupEventListeners() {
        // Handle window focus to recheck session
        window.addEventListener('focus', async () => {
            const session = await this.getSession();
            if (!session || !this.isValidSession(session)) {
                this.handleSessionExpiry();
            }
        });

        // Handle beforeunload to clean up
        window.addEventListener('beforeunload', () => {
            if (this.sessionTimer) {
                clearInterval(this.sessionTimer);
            }
        });
    }

    showPageContent() {
        // Make page content visible
        document.body.style.opacity = '1';
        
        // Dispatch custom event indicating auth is ready
        document.dispatchEvent(new CustomEvent('cspAuthReady', {
            detail: { 
                user: this.currentUser,
                authSystem: 'CSPUniversalAuth'
            }
        }));
    }

    handleSessionExpiry() {
        console.warn('⚠️ Session expired or invalid. Please log in again.');
        this.redirectToLogin();
    }

    async logout() {
        if (confirm('🔒 Are you sure you want to logout?')) {
            console.log('🚪 User logout initiated');
            await this.clearSession();
            window.location.href = this.options.loginPage;
        }
    }

    async clearSession() {
        // Clear all session storage
        localStorage.removeItem('csp_session');
        sessionStorage.removeItem('csp_session');
        sessionStorage.removeItem('csp_azure_session');
        sessionStorage.removeItem('csp_demo_session');
        
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

    // Public API methods
    async isAuthenticated() {
        await this.waitForInit();
        return this.currentUser !== null;
    }

    async getCurrentUser() {
        await this.waitForInit();
        return this.currentUser;
    }

    async hasRole(requiredRole) {
        await this.waitForInit();
        if (!this.currentUser) return false;
        
        const roleHierarchy = {
            'user': 1,
            'analyst': 2,
            'developer': 3,
            'admin': 4,
            'super_admin': 5
        };

        const userLevel = roleHierarchy[this.currentUser.role] || 1;
        const requiredLevel = roleHierarchy[requiredRole] || 1;
        
        return userLevel >= requiredLevel;
    }

    async hasPermission(permission) {
        await this.waitForInit();
        if (!this.currentUser) return false;
        
        const rolePermissions = {
            'admin': ['*'],
            'developer': ['system.view', 'ai.manage', 'quantum.view'],
            'analyst': ['system.view', 'reports.view'],
            'user': ['system.view']
        };
        
        const userPermissions = rolePermissions[this.currentUser.role] || [];
        return userPermissions.includes('*') || userPermissions.includes(permission);
    }

    // Static method to get initialized instance
    static async getInstance() {
        if (!window.CSPAuth) {
            window.CSPAuth = new CSPUniversalAuth();
        }
        await window.CSPAuth.waitForInit();
        return window.CSPAuth;
    }
}

// Initialize authentication when script loads
console.log('🔐 CSP Auth Wrapper script loaded');

// Auto-initialize but don't block if there are errors
(async () => {
    try {
        const auth = new CSPUniversalAuth();
        await auth.waitForInit();
    } catch (error) {
        console.error('❌ Auth wrapper failed to initialize:', error);
        // Allow page to continue loading even if auth fails
        document.body.style.opacity = '1';
    }
})();

// Helper functions available globally
window.requireAuth = async () => {
    const auth = await CSPUniversalAuth.getInstance();
    return auth.isAuthenticated();
};

window.requireRole = async (role) => {
    const auth = await CSPUniversalAuth.getInstance();
    return auth.hasRole(role);
};

window.getCurrentUser = async () => {
    const auth = await CSPUniversalAuth.getInstance();
    return auth.getCurrentUser();
};

window.hasPermission = async (permission) => {
    const auth = await CSPUniversalAuth.getInstance();
    return auth.hasPermission(permission);
};