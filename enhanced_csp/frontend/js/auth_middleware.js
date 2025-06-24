// js/auth-middleware.js - Global Authentication Middleware
/**
 * Enhanced CSP System - Global Authentication Middleware
 * Protects all pages and ensures Azure AD authentication
 */

class CSPAuthMiddleware {
    constructor() {
        this.currentUser = null;
        this.isInitialized = false;
        this.initializationPromise = null;
        this.publicPages = [
            '/pages/login.html',
            '/login.html',
            '/pages/signup.html',
            '/signup.html'
        ];
        this.msalInstance = null;
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
        console.log('🔐 CSP Auth Middleware - Initializing...');
        
        try {
            // Initialize MSAL if available
            if (typeof msal !== 'undefined') {
                this.msalInstance = new msal.PublicClientApplication({
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
                
                await this.msalInstance.initialize();
                console.log('✅ MSAL initialized in middleware');
            } else {
                console.warn('⚠️ MSAL library not available');
            }

            this.isInitialized = true;
            
            // Make available globally after successful initialization
            window.authMiddleware = this;
            window.unifiedAuth = this;
            
            await this.checkAuthentication();
            
            console.log('✅ CSP Auth Middleware - Initialization complete');
        } catch (error) {
            console.error('❌ Auth Middleware initialization failed:', error);
            this.isInitialized = true; // Mark as initialized to prevent hanging
            throw error;
        }
    }

    async waitForInit() {
        if (this.isInitialized) return;
        if (this.initializationPromise) {
            await this.initializationPromise;
        }
    }

    async checkAuthentication() {
        try {
            // Check for existing Azure AD session
            if (this.msalInstance) {
                const account = this.msalInstance.getActiveAccount();
                if (account) {
                    this.currentUser = await this.createUserFromAccount(account);
                    console.log('✅ Azure AD session found:', this.currentUser.email);
                    return true;
                }
            }

            // Check for stored session
            const storedSession = sessionStorage.getItem('csp_azure_session');
            if (storedSession) {
                try {
                    this.currentUser = JSON.parse(storedSession);
                    console.log('✅ Stored session found:', this.currentUser.email);
                    return true;
                } catch (e) {
                    console.error('Invalid stored session data');
                    sessionStorage.removeItem('csp_azure_session');
                }
            }

            // Check demo session as fallback
            const demoSession = sessionStorage.getItem('csp_demo_session');
            if (demoSession) {
                try {
                    this.currentUser = JSON.parse(demoSession);
                    console.log('✅ Demo session found:', this.currentUser.email);
                    return true;
                } catch (e) {
                    console.error('Invalid demo session data');
                    sessionStorage.removeItem('csp_demo_session');
                }
            }

            console.log('ℹ️ No authentication session found');
            return false;
        } catch (error) {
            console.error('❌ Authentication check failed:', error);
            return false;
        }
    }

    async createUserFromAccount(account) {
        return {
            id: account.homeAccountId,
            name: account.name || account.idTokenClaims?.name,
            email: account.username,
            role: this.mapEmailToRole(account.username),
            loginTime: new Date().toISOString(),
            authMethod: 'azure_ad',
            tenantId: account.tenantId
        };
    }

    mapEmailToRole(email) {
        if (email.includes('admin@')) return 'admin';
        if (email.includes('dev@')) return 'developer';
        if (email.includes('analyst@')) return 'analyst';
        return 'user';
    }

    isAuthenticated() {
        return this.currentUser !== null;
    }

    getCurrentUser() {
        return this.currentUser;
    }

    async requireAuthentication() {
        await this.waitForInit();

        const isAuth = await this.checkAuthentication();
        
        if (!isAuth && !this.isPublicPage()) {
            console.log('🔒 Authentication required, redirecting to login...');
            this.redirectToLogin();
            return false;
        }

        return true;
    }

    isPublicPage() {
        const currentPath = window.location.pathname;
        return this.publicPages.some(page => currentPath.includes(page));
    }

    redirectToLogin() {
        const currentUrl = encodeURIComponent(window.location.href);
        window.location.href = `/pages/login.html?redirect=${currentUrl}`;
    }

    createUserInfoDisplay() {
        if (!this.currentUser) return '';

        return `
            <div class="user-info-bar" style="
                position: fixed;
                top: 0;
                right: 0;
                background: rgba(0,0,0,0.9);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 0 0 0 8px;
                z-index: 1000;
                font-size: 0.85rem;
                display: flex;
                align-items: center;
                gap: 1rem;
                backdrop-filter: blur(10px);
            ">
                <span>👤 ${this.currentUser.name}</span>
                <span>🏷️ ${this.currentUser.role}</span>
                <button onclick="authMiddleware.logout()" style="
                    background: #dc3545;
                    color: white;
                    border: none;
                    padding: 0.25rem 0.5rem;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 0.8rem;
                ">Sign Out</button>
            </div>
        `;
    }

    addUserInfoToPage() {
        if (this.isAuthenticated() && !this.isPublicPage()) {
            const userInfoHtml = this.createUserInfoDisplay();
            document.body.insertAdjacentHTML('afterbegin', userInfoHtml);
        }
    }

    // Page-specific role checking
    hasRole(requiredRole) {
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

    requireRole(requiredRole) {
        if (!this.hasRole(requiredRole)) {
            console.warn(`🚫 Access denied. Required role: ${requiredRole}, User role: ${this.currentUser?.role}`);
            return false;
        }
        return true;
    }

    // API call wrapper with authentication
    async apiCall(url, options = {}) {
        await this.waitForInit();
        
        if (!this.isAuthenticated()) {
            throw new Error('Authentication required for API calls');
        }

        try {
            // Add authentication headers if available
            const headers = {
                'Content-Type': 'application/json',
                ...options.headers
            };

            if (this.msalInstance && this.currentUser) {
                try {
                    const account = this.msalInstance.getActiveAccount();
                    if (account) {
                        const tokenResponse = await this.msalInstance.acquireTokenSilent({
                            scopes: ["User.Read"],
                            account: account
                        });
                        headers['Authorization'] = `Bearer ${tokenResponse.accessToken}`;
                    }
                } catch (tokenError) {
                    console.warn('Could not acquire token for API call:', tokenError);
                }
            }

            const response = await fetch(url, {
                ...options,
                headers
            });

            if (response.status === 401) {
                console.warn('⚠️ API call returned 401 - session may have expired');
                await this.checkAuthentication();
                if (!this.isAuthenticated()) {
                    this.redirectToLogin();
                    throw new Error('Session expired');
                }
            }

            return response;
        } catch (error) {
            console.error('❌ API call failed:', error);
            throw error;
        }
    }

    async logout() {
        await this.waitForInit();
        
        if (confirm('🔒 Are you sure you want to logout?')) {
            console.log('🚪 User logout initiated');
            
            // Clear stored sessions
            sessionStorage.removeItem('csp_azure_session');
            sessionStorage.removeItem('csp_demo_session');
            localStorage.removeItem('csp_session');
            
            // Azure AD logout
            if (this.msalInstance && this.currentUser?.authMethod === 'azure_ad') {
                try {
                    await this.msalInstance.logoutRedirect({
                        postLogoutRedirectUri: window.location.origin
                    });
                } catch (error) {
                    console.error('Azure logout failed:', error);
                    // Fallback to simple redirect
                    window.location.href = '/pages/login.html';
                }
            } else {
                // Simple logout for demo/fallback auth
                this.currentUser = null;
                window.location.href = '/pages/login.html';
            }
        }
    }

    // Static method to get initialized instance
    static async getInstance() {
        if (!window.authMiddleware) {
            window.authMiddleware = new CSPAuthMiddleware();
        }
        await window.authMiddleware.waitForInit();
        return window.authMiddleware;
    }
}

// Initialize on script load
console.log('🔐 CSP Auth Middleware script loaded');

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', async () => {
        console.log('🔄 CSP Auth Middleware - Checking page protection...');
        
        try {
            const authMiddleware = await CSPAuthMiddleware.getInstance();
            const isAuthenticated = await authMiddleware.requireAuthentication();
            
            if (isAuthenticated) {
                // Add user info to page
                authMiddleware.addUserInfoToPage();
                
                // Set up periodic session check
                setInterval(async () => {
                    const stillAuth = await authMiddleware.checkAuthentication();
                    if (!stillAuth && !authMiddleware.isPublicPage()) {
                        console.warn('⚠️ Session expired, redirecting to login...');
                        authMiddleware.redirectToLogin();
                    }
                }, 30000); // Check every 30 seconds
                
                // Dispatch custom event when authentication is ready
                document.dispatchEvent(new CustomEvent('cspAuthReady', {
                    detail: { user: authMiddleware.getCurrentUser() }
                }));
            }
        } catch (error) {
            console.error('❌ Auth middleware initialization failed:', error);
        }
    });
} else {
    // If DOM is already loaded, initialize immediately
    CSPAuthMiddleware.getInstance().then(authMiddleware => {
        authMiddleware.requireAuthentication().then(isAuthenticated => {
            if (isAuthenticated) {
                authMiddleware.addUserInfoToPage();
                document.dispatchEvent(new CustomEvent('cspAuthReady', {
                    detail: { user: authMiddleware.getCurrentUser() }
                }));
            }
        });
    }).catch(error => {
        console.error('❌ Auth middleware initialization failed:', error);
    });
}

// Helper functions for pages to use
window.requireRole = async (role) => {
    const authMiddleware = await CSPAuthMiddleware.getInstance();
    return authMiddleware.requireRole(role);
};

window.getCurrentUser = async () => {
    const authMiddleware = await CSPAuthMiddleware.getInstance();
    return authMiddleware.getCurrentUser();
};

window.isAuthenticated = async () => {
    const authMiddleware = await CSPAuthMiddleware.getInstance();
    return authMiddleware.isAuthenticated();
};

window.makeAuthenticatedApiCall = async (url, options) => {
    const authMiddleware = await CSPAuthMiddleware.getInstance();
    return authMiddleware.apiCall(url, options);
};

console.log('🔐 CSP Authentication Middleware loaded');