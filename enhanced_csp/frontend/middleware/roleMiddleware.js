/**
 * Enhanced CSP System - Role-based Access Control Middleware
 * Client-side role and permission validation
 */

import { authMiddleware } from './authMiddleware';

class RoleMiddleware {
    constructor() {
        this.roleHierarchy = {
            'Administrator': ['Security Officer', 'Developer', 'Analyst', 'User'],
            'Security Officer': ['Analyst', 'User'],
            'Developer': ['User'],
            'Analyst': ['User'],
            'User': []
        };

        this.rolePermissions = {
            'Administrator': [
                'view_all_designs',
                'create_design',
                'edit_design',
                'delete_design',
                'execute_design',
                'manage_users',
                'manage_system',
                'view_metrics',
                'manage_components'
            ],
            'Security Officer': [
                'view_all_designs',
                'create_design',
                'edit_design',
                'execute_design',
                'view_metrics',
                'manage_security'
            ],
            'Developer': [
                'view_own_designs',
                'create_design',
                'edit_own_design',
                'execute_design',
                'manage_components'
            ],
            'Analyst': [
                'view_shared_designs',
                'create_design',
                'execute_design',
                'view_metrics'
            ],
            'User': [
                'view_shared_designs',
                'execute_design'
            ]
        };
    }

    /**
     * Get user roles from token claims
     */
    getUserRoles() {
        const claims = authMiddleware.getUserClaims();
        if (!claims) return [];

        // Extract roles from Azure AD claims
        const roles = [];
        
        // Check for roles in different claim types
        if (claims.roles) {
            roles.push(...claims.roles);
        }
        
        if (claims.groups) {
            // Map Azure AD groups to CSP roles
            const groupRoleMap = {
                'CSP-Administrators': 'Administrator',
                'CSP-Security-Officers': 'Security Officer',
                'CSP-Developers': 'Developer',
                'CSP-Analysts': 'Analyst',
                'CSP-Users': 'User'
            };

            claims.groups.forEach(groupId => {
                const role = groupRoleMap[groupId];
                if (role && !roles.includes(role)) {
                    roles.push(role);
                }
            });
        }

        // Default role if no specific roles found
        if (roles.length === 0) {
            roles.push('User');
        }

        return roles;
    }

    /**
     * Check if user has specific role
     */
    hasRole(requiredRole) {
        const userRoles = this.getUserRoles();
        
        // Direct role match
        if (userRoles.includes(requiredRole)) {
            return true;
        }

        // Check role hierarchy (higher roles include lower role permissions)
        return userRoles.some(userRole => {
            const subordinateRoles = this.roleHierarchy[userRole] || [];
            return subordinateRoles.includes(requiredRole);
        });
    }

    /**
     * Check if user has any of the specified roles
     */
    hasAnyRole(requiredRoles) {
        return requiredRoles.some(role => this.hasRole(role));
    }

    /**
     * Check if user has all specified roles
     */
    hasAllRoles(requiredRoles) {
        return requiredRoles.every(role => this.hasRole(role));
    }

    /**
     * Get user permissions based on roles
     */
    getUserPermissions() {
        const userRoles = this.getUserRoles();
        const permissions = new Set();

        userRoles.forEach(role => {
            const rolePermissions = this.rolePermissions[role] || [];
            rolePermissions.forEach(permission => permissions.add(permission));
        });

        return Array.from(permissions);
    }

    /**
     * Check if user has specific permission
     */
    hasPermission(requiredPermission) {
        const userPermissions = this.getUserPermissions();
        return userPermissions.includes(requiredPermission);
    }

    /**
     * Check if user has any of the specified permissions
     */
    hasAnyPermission(requiredPermissions) {
        return requiredPermissions.some(permission => this.hasPermission(permission));
    }

    /**
     * Check if user has all specified permissions
     */
    hasAllPermissions(requiredPermissions) {
        return requiredPermissions.every(permission => this.hasPermission(permission));
    }

    /**
     * Get highest user role (for display purposes)
     */
    getHighestRole() {
        const userRoles = this.getUserRoles();
        const roleOrder = ['Administrator', 'Security Officer', 'Developer', 'Analyst', 'User'];
        
        for (const role of roleOrder) {
            if (userRoles.includes(role)) {
                return role;
            }
        }
        
        return 'User';
    }

    /**
     * Validate route access based on role requirements
     */
    validateRouteAccess(routeConfig) {
        if (!authMiddleware.isAuthenticated()) {
            return {
                allowed: false,
                reason: 'authentication_required',
                redirect: '/login'
            };
        }

        // No role requirements - allow access
        if (!routeConfig.requiredRoles && !routeConfig.requiredPermissions) {
            return { allowed: true };
        }

        // Check role requirements
        if (routeConfig.requiredRoles) {
            const hasRequiredRole = routeConfig.requireAllRoles 
                ? this.hasAllRoles(routeConfig.requiredRoles)
                : this.hasAnyRole(routeConfig.requiredRoles);

            if (!hasRequiredRole) {
                return {
                    allowed: false,
                    reason: 'insufficient_role',
                    redirect: '/unauthorized',
                    required: routeConfig.requiredRoles,
                    current: this.getUserRoles()
                };
            }
        }

        // Check permission requirements
        if (routeConfig.requiredPermissions) {
            const hasRequiredPermission = routeConfig.requireAllPermissions
                ? this.hasAllPermissions(routeConfig.requiredPermissions)
                : this.hasAnyPermission(routeConfig.requiredPermissions);

            if (!hasRequiredPermission) {
                return {
                    allowed: false,
                    reason: 'insufficient_permission',
                    redirect: '/unauthorized',
                    required: routeConfig.requiredPermissions,
                    current: this.getUserPermissions()
                };
            }
        }

        return { allowed: true };
    }

    /**
     * Create role-based router guard
     */
    createRouteGuard(routeConfig) {
        return (to, from, next) => {
            const validation = this.validateRouteAccess(routeConfig);
            
            if (validation.allowed) {
                next();
            } else {
                console.warn(`Access denied to ${to.path}:`, validation.reason);
                next(validation.redirect);
            }
        };
    }
}

export const roleMiddleware = new RoleMiddleware();

export default roleMiddleware;