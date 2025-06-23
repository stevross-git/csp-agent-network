// components/ProtectedRoute.js
import { authService } from '../services/authService.js';

export class ProtectedRoute {
    constructor(requiredPermissions = []) {
        this.requiredPermissions = requiredPermissions;
    }

    async checkAccess() {
        if (!authService.isAuthenticated()) {
            return { allowed: false, reason: 'Not authenticated' };
        }

        if (this.requiredPermissions.length === 0) {
            return { allowed: true };
        }

        const hasAllPermissions = this.requiredPermissions.every(
            permission => authService.hasPermission(permission)
        );

        if (!hasAllPermissions) {
            return { 
                allowed: false, 
                reason: 'Insufficient permissions',
                required: this.requiredPermissions,
                userRole: authService.getUserRole()
            };
        }

        return { allowed: true };
    }

    async renderWithAuth(renderFunction, fallbackFunction = null) {
        const access = await this.checkAccess();
        
        if (access.allowed) {
            return renderFunction();
        } else {
            if (fallbackFunction) {
                return fallbackFunction(access);
            } else {
                return this.renderUnauthorized(access);
            }
        }
    }

    renderUnauthorized(access) {
        return `
            <div class="unauthorized-container">
                <h2>🚫 Access Denied</h2>
                <p>${access.reason}</p>
                ${access.required ? `<p>Required permissions: ${access.required.join(', ')}</p>` : ''}
                ${access.userRole ? `<p>Your role: ${access.userRole}</p>` : ''}
                <button onclick="authService.logout()">Logout</button>
            </div>
        `;
    }
}
