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
