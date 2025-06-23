// services/sessionService.js
import { authService } from './authService.js';

class SessionService {
    constructor() {
        this.sessionData = null;
        this.initializeSession();
    }

    async initializeSession() {
        if (authService.isAuthenticated()) {
            await this.createSession();
        }
    }

    async createSession() {
        const userInfo = authService.getUserInfo();
        const role = authService.getUserRole();
        
        this.sessionData = {
            userId: userInfo.homeAccountId,
            username: userInfo.username,
            name: userInfo.name,
            email: userInfo.username,
            role: role,
            loginTime: new Date().toISOString(),
            lastActivity: new Date().toISOString(),
            permissions: ROLE_PERMISSIONS[role] || []
        };

        // Store session data
        sessionStorage.setItem('csp_azure_session', JSON.stringify(this.sessionData));
        
        // Set up activity tracking
        this.setupActivityTracking();
    }

    setupActivityTracking() {
        const updateActivity = () => {
            if (this.sessionData) {
                this.sessionData.lastActivity = new Date().toISOString();
                sessionStorage.setItem('csp_azure_session', JSON.stringify(this.sessionData));
            }
        };

        // Track user activity
        ['click', 'keypress', 'scroll', 'mousemove'].forEach(event => {
            document.addEventListener(event, updateActivity, { passive: true });
        });
    }

    getSession() {
        const stored = sessionStorage.getItem('csp_azure_session');
        return stored ? JSON.parse(stored) : this.sessionData;
    }

    clearSession() {
        this.sessionData = null;
        sessionStorage.removeItem('csp_azure_session');
    }

    isSessionValid() {
        const session = this.getSession();
        if (!session) return false;

        // Check if session is expired (e.g., 8 hours)
        const loginTime = new Date(session.loginTime);
        const now = new Date();
        const sessionDuration = now - loginTime;
        const maxDuration = 8 * 60 * 60 * 1000; // 8 hours

        return sessionDuration < maxDuration;
    }
}

export const sessionService = new SessionService();