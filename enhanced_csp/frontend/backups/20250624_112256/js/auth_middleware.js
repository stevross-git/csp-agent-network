// js/auth-middleware.js - Global Authentication Middleware
/**
 * Enhanced CSP System - Global Authentication Middleware
 * Protects all pages and ensures Azure AD authentication
 */

class CSPAuthMiddleware {
    constructor() {
        this.currentUser = null;
        this.isInitialized = false;
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
        console.log('🔐 CSP Auth Middleware - Initializing...');
        
        // Initialize MSAL if available
        if (typeof msal !== 'undefined') {
            try {
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
            } catch (error) {
                console.error('❌ MSAL initialization failed:', error);
            }
        }

        this.isInitialized = true;
        await this.checkAuthentication();
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
                this.currentUser = JSON.parse(storedSession);
                console.log('✅ Stored session found:', this.currentUser.email);
                return true;
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
        if (!this.isInitialized) {
            await this.init();
        }

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

    async logout() {
        try {
            if (this.msalInstance && this.currentUser?.authMethod === 'azure_ad') {
                await this.msalInstance.logoutPopup();
            }
            
            // Clear all sessions
            sessionStorage.removeItem('csp_azure_session');
            sessionStorage.removeItem('csp_demo_session');
            this.currentUser = null;
            
            console.log('✅ Logout successful');
            this.redirectToLogin();
        } catch (error) {
            console.error('❌ Logout failed:', error);
        }
    }

    async getAccessToken() {
        if (!this.msalInstance || !this.currentUser) {
            throw new Error('No active authentication session');
        }

        try {
            const account = this.msalInstance.getActiveAccount();
            const tokenResponse = await this.msalInstance.acquireTokenSilent({
                scopes: ["User.Read"],
                account: account
            });
            
            return tokenResponse.accessToken;
        } catch (error) {
            if (error.name === 'InteractionRequiredAuthError') {
                // Try popup token acquisition
                const tokenResponse = await this.msalInstance.acquireTokenPopup({
                    scopes: ["User.Read"]
                });
                return tokenResponse.accessToken;
            }
            throw error;
        }
    }

    async apiCall(url, options = {}) {
        try {
            const token = await this.getAccessToken();
            
            const headers = {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
                ...options.headers
            };

            const response = await fetch(url, {
                ...options,
                headers
            });

            if (response.status === 401) {
                console.warn('⚠️ API call unauthorized, requiring re-authentication');
                await this.logout();
                return null;
            }

            return response;
        } catch (error) {
            console.error('❌ API call failed:', error);
            throw error;
        }
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
            this.showAccessDenied(requiredRole);
            return false;
        }
        return true;
    }

    showAccessDenied(requiredRole) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;
        
        modal.innerHTML = `
            <div style="
                background: white;
                border-radius: 12px;
                padding: 2rem;
                max-width: 400px;
                text-align: center;
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            ">
                <h2 style="color: #dc3545; margin-bottom: 1rem;">🚫 Access Denied</h2>
                <p style="margin-bottom: 1.5rem;">
                    This page requires <strong>${requiredRole}</strong> role or higher.<br>
                    Your current role: <strong>${this.currentUser?.role || 'none'}</strong>
                </p>
                <button onclick="history.back()" style="
                    background: #6c757d;
                    color: white;
                    border: none;
                    padding: 0.75rem 1.5rem;
                    border-radius: 6px;
                    cursor: pointer;
                    margin-right: 0.5rem;
                ">Go Back</button>
                <button onclick="authMiddleware.logout()" style="
                    background: #dc3545;
                    color: white;
                    border: none;
                    padding: 0.75rem 1.5rem;
                    border-radius: 6px;
                    cursor: pointer;
                ">Sign Out</button>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
}

// Global instance
window.authMiddleware = new CSPAuthMiddleware();

// Auto-protect pages when DOM loads
document.addEventListener('DOMContentLoaded', async function() {
    console.log('🔄 CSP Auth Middleware - Checking page protection...');
    
    const isAuthenticated = await window.authMiddleware.requireAuthentication();
    
    if (isAuthenticated) {
        // Add user info to page
        window.authMiddleware.addUserInfoToPage();
        
        // Set up periodic session check
        setInterval(async () => {
            const stillAuth = await window.authMiddleware.checkAuthentication();
            if (!stillAuth && !window.authMiddleware.isPublicPage()) {
                console.warn('⚠️ Session expired, redirecting to login...');
                window.authMiddleware.redirectToLogin();
            }
        }, 30000); // Check every 30 seconds
    }
});

// Helper functions for pages to use
window.requireRole = (role) => window.authMiddleware.requireRole(role);
window.getCurrentUser = () => window.authMiddleware.getCurrentUser();
window.isAuthenticated = () => window.authMiddleware.isAuthenticated();
window.makeAuthenticatedApiCall = (url, options) => window.authMiddleware.apiCall(url, options);

console.log('🔐 CSP Authentication Middleware loaded');