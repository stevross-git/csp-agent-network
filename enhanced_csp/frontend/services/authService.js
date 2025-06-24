// Authentication Service for Enhanced CSP System
import { msalConfig, loginRequest, tokenRequest, USER_ROLES } from '../config/azureConfig.js';
import { getUserRole } from '../config/roles.js';

class AuthService {
    constructor() {
        this.msalInstance = null;
        this.account = null;
        this.userRole = null;
        this.isInitialized = false;
        this.initializationPromise = null;
        this.initializeAuth();
    }

    async initializeAuth() {
        // Prevent multiple initialization attempts
        if (this.initializationPromise) {
            return this.initializationPromise;
        }

        this.initializationPromise = this._doInitialize();
        return this.initializationPromise;
    }

    async _doInitialize() {
        try {
            console.log('🔐 AuthService: Starting initialization...');
            
            // Check if MSAL is available
            if (typeof msal === 'undefined') {
                console.warn('⚠️ MSAL library not loaded, using fallback authentication');
                this.isInitialized = true;
                return false;
            }

            // Initialize MSAL
            this.msalInstance = new msal.PublicClientApplication(msalConfig);
            await this.msalInstance.initialize();
            
            // Get active account
            this.account = this.msalInstance.getActiveAccount();
            
            if (this.account) {
                await this.setUserRole();
                console.log('✅ AuthService: Existing session restored');
            }
            
            this.isInitialized = true;
            
            // Make available globally after successful initialization
            window.unifiedAuth = this;
            window.authService = this;
            
            console.log('✅ AuthService: Initialization complete');
            return true;
        } catch (error) {
            console.error('❌ AuthService initialization failed:', error);
            this.isInitialized = true; // Mark as initialized even on error to prevent hanging
            throw error;
        }
    }

    // Wait for initialization to complete
    async waitForInit() {
        if (this.isInitialized) return;
        if (this.initializationPromise) {
            await this.initializationPromise;
        }
    }

    async login() {
        await this.waitForInit();
        
        if (!this.msalInstance) {
            throw new Error('MSAL not available');
        }

        try {
            const loginResponse = await this.msalInstance.loginPopup(loginRequest);
            this.account = loginResponse.account;
            this.msalInstance.setActiveAccount(this.account);
            await this.setUserRole();
            
            // Store session info
            this.storeSessionInfo();
            
            return loginResponse;
        } catch (error) {
            console.error('Login failed:', error);
            throw error;
        }
    }

    async loginRedirect() {
        await this.waitForInit();
        
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

    async logout() {
        await this.waitForInit();
        
        try {
            // Clear stored session
            this.clearSessionInfo();
            
            if (this.msalInstance && this.account) {
                await this.msalInstance.logoutPopup({
                    account: this.account,
                    postLogoutRedirectUri: msalConfig.auth.postLogoutRedirectUri
                });
            }
            
            this.account = null;
            this.userRole = null;
            
            // Clear global references
            if (window.unifiedAuth === this) {
                window.unifiedAuth = null;
            }
            
        } catch (error) {
            console.error('Logout failed:', error);
            throw error;
        }
    }

    async getAccessToken() {
        await this.waitForInit();
        
        if (!this.account) {
            throw new Error('No active account');
        }

        if (!this.msalInstance) {
            throw new Error('MSAL not available');
        }

        try {
            const tokenResponse = await this.msalInstance.acquireTokenSilent({
                ...tokenRequest,
                account: this.account
            });
            return tokenResponse.accessToken;
        } catch (error) {
            if (error.name === 'InteractionRequiredAuthError') {
                const tokenResponse = await this.msalInstance.acquireTokenPopup(tokenRequest);
                return tokenResponse.accessToken;
            }
            throw error;
        }
    }

    async setUserRole() {
        if (!this.account) return;

        try {
            // Get additional user info from Microsoft Graph
            const accessToken = await this.getAccessToken();
            const userInfo = await this.getUserInfo(accessToken);
            this.userRole = getUserRole(userInfo);
        } catch (error) {
            console.error('Failed to set user role:', error);
            this.userRole = USER_ROLES.USER; // Default role
        }
    }

    async getUserInfo(accessToken) {
        const response = await fetch('https://graph.microsoft.com/v1.0/me?$select=id,displayName,mail,userPrincipalName,jobTitle,department', {
            headers: {
                'Authorization': `Bearer ${accessToken}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch user info');
        }
        
        return response.json();
    }

    // Session management
    storeSessionInfo() {
        if (this.account) {
            const sessionInfo = {
                account: this.account,
                userRole: this.userRole,
                timestamp: Date.now()
            };
            sessionStorage.setItem('csp_auth_session', JSON.stringify(sessionInfo));
        }
    }

    clearSessionInfo() {
        sessionStorage.removeItem('csp_auth_session');
        localStorage.removeItem('csp_auth_session');
    }

    restoreSessionInfo() {
        try {
            const sessionData = sessionStorage.getItem('csp_auth_session');
            if (sessionData) {
                const session = JSON.parse(sessionData);
                // Check if session is not too old (24 hours)
                if (Date.now() - session.timestamp < 24 * 60 * 60 * 1000) {
                    this.account = session.account;
                    this.userRole = session.userRole;
                    return true;
                }
            }
        } catch (error) {
            console.error('Failed to restore session:', error);
        }
        return false;
    }

    // Public API methods
    isAuthenticated() {
        return this.account !== null;
    }

    getCurrentUser() {
        return this.account;
    }

    getUserRole() {
        return this.userRole || USER_ROLES.USER;
    }

    hasRole(requiredRole) {
        const userRole = this.getUserRole();
        const roleHierarchy = {
            [USER_ROLES.USER]: 1,
            [USER_ROLES.ANALYST]: 2,
            [USER_ROLES.DEVELOPER]: 3,
            [USER_ROLES.ADMIN]: 4,
            [USER_ROLES.SUPER_ADMIN]: 5
        };
        
        const userLevel = roleHierarchy[userRole] || 1;
        const requiredLevel = roleHierarchy[requiredRole] || 1;
        
        return userLevel >= requiredLevel;
    }

    hasPermission(permission) {
        const rolePermissions = {
            [USER_ROLES.SUPER_ADMIN]: ['*'],
            [USER_ROLES.ADMIN]: ['system.admin', 'user.manage', 'ai.manage', 'quantum.manage', 'blockchain.manage'],
            [USER_ROLES.DEVELOPER]: ['system.view', 'ai.manage', 'quantum.view', 'blockchain.view'],
            [USER_ROLES.ANALYST]: ['system.view', 'ai.view', 'quantum.view', 'reports.view'],
            [USER_ROLES.USER]: ['system.view']
        };
        
        const userPermissions = rolePermissions[this.getUserRole()] || [];
        return userPermissions.includes('*') || userPermissions.includes(permission);
    }

    // Utility method to safely get auth service
    static async getInstance() {
        if (!window.authService) {
            window.authService = new AuthService();
        }
        await window.authService.waitForInit();
        return window.authService;
    }
}

// Create singleton instance
const authService = new AuthService();

// Helper function for getting initialized auth service
export const getAuthService = async () => {
    await authService.waitForInit();
    return authService;
};

// Helper function for safe auth operations
export const withAuth = async (callback) => {
    const auth = await getAuthService();
    return callback(auth);
};

export { authService };
export default authService;