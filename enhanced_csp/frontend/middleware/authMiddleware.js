// middleware/authMiddleware.js - Enhanced Authentication Middleware
/**
 * Enhanced CSP System - Comprehensive Authentication Middleware
 * Handles Azure AD, session management, and role-based access control
 */

import { msalConfig, loginRequest, tokenRequest } from '../config/azureConfig.js';
import { getUserRole, hasPermission } from '../config/roles.js';

class AuthMiddleware {
    constructor() {
        this.msalInstance = null;
        this.currentAccount = null;
        this.userProfile = null;
        this.isInitialized = false;
        this.initializationPromise = null;
        this.sessionCheckInterval = null;
        
        console.log('🔐 AuthMiddleware: Starting initialization...');
        this.initialize();
    }

    /**
     * Initialize the authentication middleware
     */
    async initialize() {
        // Prevent multiple initialization attempts
        if (this.initializationPromise) {
            return this.initializationPromise;
        }

        this.initializationPromise = this._doInitialize();
        return this.initializationPromise;
    }

    async _doInitialize() {
        try {
            console.log('🔄 AuthMiddleware: Initializing MSAL...');
            
            // Check if MSAL is available
            if (typeof msal === 'undefined') {
                console.warn('⚠️ MSAL library not available');
                this.isInitialized = true;
                return false;
            }

            // Initialize MSAL
            this.msalInstance = new msal.PublicClientApplication(msalConfig);
            await this.msalInstance.initialize();
            
            // Get current account
            this.currentAccount = this.msalInstance.getActiveAccount();
            
            if (this.currentAccount) {
                await this.loadUserProfile();
                this.setupSessionMonitoring();
                console.log('✅ AuthMiddleware: Existing session restored');
            }
            
            this.isInitialized = true;
            
            // Make available globally after successful initialization
            window.authMiddleware = this;
            window.unifiedAuth = this;
            
            console.log('✅ AuthMiddleware: Initialization complete');
            return true;
        } catch (error) {
            console.error('❌ AuthMiddleware initialization failed:', error);
            this.isInitialized = true; // Mark as initialized to prevent hanging
            throw error;
        }
    }

    /**
     * Wait for initialization to complete
     */
    async waitForInitialization() {
        if (this.isInitialized) return;
        if (this.initializationPromise) {
            await this.initializationPromise;
        }
    }

    /**
     * Check if user is authenticated
     */
    isAuthenticated() {
        return this.currentAccount !== null && this.userProfile !== null;
    }

    /**
     * Get current account
     */
    getCurrentAccount() {
        return this.currentAccount;
    }

    /**
     * Get current user profile
     */
    getCurrentUser() {
        return this.userProfile;
    }

    /**
     * Login with popup
     */
    async loginPopup() {
        await this.waitForInitialization();
        
        if (!this.msalInstance) {
            throw new Error('MSAL not available');
        }

        try {
            const loginResponse = await this.msalInstance.loginPopup(loginRequest);
            this.currentAccount = loginResponse.account;
            this.msalInstance.setActiveAccount(this.currentAccount);
            
            await this.loadUserProfile();
            this.setupSessionMonitoring();
            this.storeSessionInfo();
            
            return loginResponse;
        } catch (error) {
            console.error('Login failed:', error);
            throw error;
        }
    }

    /**
     * Login with redirect
     */
    async loginRedirect() {
        await this.waitForInitialization();
        
        if (!this.msalInstance) {
            throw new Error('MSAL not available');
        }

        try {
            await this.msalInstance.loginRedirect(loginRequest);
        } catch (error) {
            console.error('Login redirect failed:', error);
            throw error;
        }
    }

    /**
     * Handle redirect response
     */
    async handleRedirectResponse() {
        await this.waitForInitialization();
        
        if (!this.msalInstance) {
            return null;
        }

        try {
            const response = await this.msalInstance.handleRedirectPromise();
            
            if (response) {
                this.currentAccount = response.account;
                this.msalInstance.setActiveAccount(this.currentAccount);
                await this.loadUserProfile();
                this.setupSessionMonitoring();
                this.storeSessionInfo();
            }
            
            return response;
        } catch (error) {
            console.error('Handle redirect failed:', error);
            throw error;
        }
    }

    /**
     * Load user profile from Microsoft Graph
     */
    async loadUserProfile() {
        if (!this.currentAccount) {
            throw new Error('No active account');
        }

        try {
            // Get access token
            const accessToken = await this.getValidToken(['User.Read']);
            
            // Fetch user info from Microsoft Graph
            const userInfo = await this.fetchUserInfo(accessToken.accessToken);
            
            // Create user profile
            this.userProfile = {
                id: this.currentAccount.homeAccountId,
                name: userInfo.displayName || this.currentAccount.name,
                email: userInfo.mail || this.currentAccount.username,
                role: getUserRole(userInfo),
                department: userInfo.department,
                jobTitle: userInfo.jobTitle,
                authMethod: 'azure_ad',
                loginTime: new Date().toISOString(),
                lastActivity: new Date().toISOString()
            };
            
            console.log('✅ User profile loaded:', this.userProfile.email);
        } catch (error) {
            console.error('Failed to load user profile:', error);
            
            // Fallback profile from account info
            this.userProfile = {
                id: this.currentAccount.homeAccountId,
                name: this.currentAccount.name,
                email: this.currentAccount.username,
                role: 'user',
                authMethod: 'azure_ad',
                loginTime: new Date().toISOString(),
                lastActivity: new Date().toISOString()
            };
        }
    }

    /**
     * Fetch user info from Microsoft Graph
     */
    async fetchUserInfo(accessToken) {
        const response = await fetch('https://graph.microsoft.com/v1.0/me?$select=id,displayName,mail,userPrincipalName,jobTitle,department', {
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`Graph API request failed: ${response.status}`);
        }

        return response.json();
    }

    /**
     * Get valid access token
     */
    async getValidToken(scopes = ['User.Read']) {
        await this.waitForInitialization();
        
        if (!this.currentAccount || !this.msalInstance) {
            throw new Error('No active account or MSAL instance');
        }

        try {
            // Try silent token acquisition first
            const tokenResponse = await this.msalInstance.acquireTokenSilent({
                scopes: scopes,
                account: this.currentAccount
            });

            return tokenResponse;
        } catch (error) {
            if (error.name === 'InteractionRequiredAuthError') {
                // Fall back to popup interaction
                const tokenResponse = await this.msalInstance.acquireTokenPopup({
                    scopes: scopes,
                    account: this.currentAccount
                });
                return tokenResponse;
            }
            throw error;
        }
    }

    /**
     * Make authenticated API call
     */
    async makeAuthenticatedRequest(url, options = {}) {
        await this.waitForInitialization();
        
        if (!this.isAuthenticated()) {
            throw new Error('User not authenticated');
        }

        try {
            // Get access token
            const tokenResponse = await this.getValidToken(['User.Read']);
            
            // Prepare headers
            const headers = {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${tokenResponse.accessToken}`,
                ...options.headers
            };

            // Make request
            const response = await fetch(url, {
                ...options,
                headers
            });

            if (response.status === 401) {
                // Token might be invalid, try to refresh
                console.log('Received 401, attempting token refresh...');
                const refreshedToken = await this.getValidToken();
                
                const retryHeaders = {
                    ...headers,
                    'Authorization': `Bearer ${refreshedToken.accessToken}`
                };

                return fetch(url, {
                    ...options,
                    headers: retryHeaders
                });
            }

            return response;
        } catch (error) {
            console.error('Authenticated request failed:', error);
            throw error;
        }
    }

    /**
     * Check if user has specific role
     */
    hasRole(requiredRole) {
        if (!this.userProfile) return false;
        
        const roleHierarchy = {
            'user': 1,
            'analyst': 2,
            'developer': 3,
            'admin': 4,
            'super_admin': 5
        };

        const userLevel = roleHierarchy[this.userProfile.role] || 1;
        const requiredLevel = roleHierarchy[requiredRole] || 1;
        
        return userLevel >= requiredLevel;
    }

    /**
     * Check if user has specific permission
     */
    hasPermission(permission) {
        if (!this.userProfile) return false;
        return hasPermission(this.userProfile.role, permission);
    }

    /**
     * Store session information
     */
    storeSessionInfo() {
        if (this.userProfile) {
            const sessionInfo = {
                account: this.currentAccount,
                userProfile: this.userProfile,
                timestamp: Date.now()
            };
            sessionStorage.setItem('csp_auth_session', JSON.stringify(sessionInfo));
        }
    }

    /**
     * Restore session information
     */
    restoreSessionInfo() {
        try {
            const sessionData = sessionStorage.getItem('csp_auth_session');
            if (sessionData) {
                const session = JSON.parse(sessionData);
                
                // Check if session is not too old (24 hours)
                if (Date.now() - session.timestamp < 24 * 60 * 60 * 1000) {
                    this.currentAccount = session.account;
                    this.userProfile = session.userProfile;
                    return true;
                }
            }
        } catch (error) {
            console.error('Failed to restore session:', error);
        }
        return false;
    }

    /**
     * Clear session information
     */
    clearSessionInfo() {
        sessionStorage.removeItem('csp_auth_session');
        localStorage.removeItem('csp_auth_session');
        
        if (this.sessionCheckInterval) {
            clearInterval(this.sessionCheckInterval);
            this.sessionCheckInterval = null;
        }
    }

    /**
     * Setup session monitoring
     */
    setupSessionMonitoring() {
        // Clear existing interval
        if (this.sessionCheckInterval) {
            clearInterval(this.sessionCheckInterval);
        }

        // Check session every 5 minutes
        this.sessionCheckInterval = setInterval(async () => {
            try {
                if (this.currentAccount && this.msalInstance) {
                    // Try to acquire token silently to check if session is still valid
                    await this.msalInstance.acquireTokenSilent({
                        scopes: ['User.Read'],
                        account: this.currentAccount
                    });
                    
                    // Update last activity
                    if (this.userProfile) {
                        this.userProfile.lastActivity = new Date().toISOString();
                        this.storeSessionInfo();
                    }
                }
            } catch (error) {
                console.warn('Session check failed:', error);
                // Session might be invalid, trigger logout
                this.handleSessionExpiry();
            }
        }, 5 * 60 * 1000); // 5 minutes
    }

    /**
     * Handle session expiry
     */
    handleSessionExpiry() {
        console.warn('⚠️ Session expired');
        this.clearSessionInfo();
        this.currentAccount = null;
        this.userProfile = null;
        
        // Dispatch event for UI components to handle
        window.dispatchEvent(new CustomEvent('sessionExpired'));
    }

    /**
     * Logout current user
     */
    async logout() {
        await this.waitForInitialization();
        
        try {
            this.clearSessionInfo();
            
            if (this.msalInstance && this.currentAccount) {
                await this.msalInstance.logoutRedirect({
                    account: this.currentAccount,
                    postLogoutRedirectUri: window.location.origin
                });
            }
            
            this.currentAccount = null;
            this.userProfile = null;
            
            // Clear global references
            if (window.unifiedAuth === this) {
                window.unifiedAuth = null;
            }
            
        } catch (error) {
            console.error('Logout failed:', error);
            throw error;
        }
    }

    /**
     * Validate session and redirect if necessary
     */
    validateSession(requiredPath = '/pages/login.html') {
        if (!this.isAuthenticated()) {
            console.log('User not authenticated, redirecting to login...');
            window.location.href = requiredPath;
            return false;
        }
        return true;
    }

    /**
     * Get initialized instance (static method)
     */
    static async getInstance() {
        if (!window.authMiddleware) {
            window.authMiddleware = new AuthMiddleware();
        }
        await window.authMiddleware.waitForInitialization();
        return window.authMiddleware;
    }
}

// Create and export singleton instance
const authMiddleware = new AuthMiddleware();

// Helper functions for easy access
export const getAuthMiddleware = async () => {
    await authMiddleware.waitForInitialization();
    return authMiddleware;
};

export const withAuth = async (callback) => {
    const auth = await getAuthMiddleware();
    return callback(auth);
};

export const requireAuth = async () => {
    const auth = await getAuthMiddleware();
    return auth.validateSession();
};

export const getCurrentUser = async () => {
    const auth = await getAuthMiddleware();
    return auth.getCurrentUser();
};

export const hasRole = async (role) => {
    const auth = await getAuthMiddleware();
    return auth.hasRole(role);
};

export const hasPermission = async (permission) => {
    const auth = await getAuthMiddleware();
    return auth.hasPermission(permission);
};

export const makeAuthenticatedRequest = async (url, options) => {
    const auth = await getAuthMiddleware();
    return auth.makeAuthenticatedRequest(url, options);
};

// Make available globally
window.getAuthMiddleware = getAuthMiddleware;
window.withAuth = withAuth;
window.requireAuth = requireAuth;

export { authMiddleware };
export default authMiddleware;