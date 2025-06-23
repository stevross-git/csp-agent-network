/**
 * Enhanced CSP System - Role Utility Functions
 * Helper functions for role and permission management
 */

/**
 * Check if user has specific role
 */
export const checkUserRole = (userProfile, requiredRole) => {
    if (!userProfile || !userProfile.roles) {
        return false;
    }

    return userProfile.roles.includes(requiredRole);
};

/**
 * Check if user has any of the specified roles
 */
export const checkUserRoles = (userProfile, requiredRoles, requireAll = false) => {
    if (!userProfile || !userProfile.roles || !Array.isArray(requiredRoles)) {
        return false;
    }

    if (requireAll) {
        return requiredRoles.every(role => userProfile.roles.includes(role));
    } else {
        return requiredRoles.some(role => userProfile.roles.includes(role));
    }
};

/**
 * Check if user has specific permission
 */
export const checkUserPermission = (userProfile, requiredPermission) => {
    if (!userProfile || !userProfile.permissions) {
        return false;
    }

    return userProfile.permissions.includes(requiredPermission);
};

/**
 * Check if user has any of the specified permissions
 */
export const checkUserPermissions = (userProfile, requiredPermissions, requireAll = false) => {
    if (!userProfile || !userProfile.permissions || !Array.isArray(requiredPermissions)) {
        return false;
    }

    if (requireAll) {
        return requiredPermissions.every(permission => userProfile.permissions.includes(permission));
    } else {
        return requiredPermissions.some(permission => userProfile.permissions.includes(permission));
    }
};

/**
 * Get user's highest role (for display purposes)
 */
export const getUserHighestRole = (userProfile) => {
    if (!userProfile || !userProfile.roles) {
        return 'Guest';
    }

    const roleHierarchy = ['Administrator', 'Security Officer', 'Developer', 'Analyst', 'User'];
    
    for (const role of roleHierarchy) {
        if (userProfile.roles.includes(role)) {
            return role;
        }
    }

    return 'User';
};

/**
 * Check if user can access resource
 */
export const canAccessResource = (userProfile, resourceConfig) => {
    // Check authentication
    if (!userProfile) {
        return false;
    }

    // Check if user is active
    if (userProfile.isActive === false) {
        return false;
    }

    // Check role requirements
    if (resourceConfig.requiredRoles) {
        const hasRole = checkUserRoles(
            userProfile, 
            resourceConfig.requiredRoles, 
            resourceConfig.requireAllRoles
        );
        if (!hasRole) {
            return false;
        }
    }

    // Check permission requirements
    if (resourceConfig.requiredPermissions) {
        const hasPermission = checkUserPermissions(
            userProfile, 
            resourceConfig.requiredPermissions, 
            resourceConfig.requireAllPermissions
        );
        if (!hasPermission) {
            return false;
        }
    }

    return true;
};

/**
 * Format role for display
 */
export const formatRole = (role) => {
    return role.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
};

/**
 * Get role color for badges
 */
export const getRoleColor = (role) => {
    const roleColors = {
        'Administrator': 'danger',
        'Security Officer': 'warning',
        'Developer': 'info',
        'Analyst': 'success',
        'User': 'secondary'
    };

    return roleColors[role] || 'secondary';
};

/**
 * Get permission description
 */
export const getPermissionDescription = (permission) => {
    const descriptions = {
        'view_all_designs': 'View all design projects',
        'create_design': 'Create new design projects',
        'edit_design': 'Edit existing designs',
        'delete_design': 'Delete design projects',
        'execute_design': 'Execute design workflows',
        'manage_users': 'Manage user accounts',
        'manage_system': 'System administration',
        'view_metrics': 'View system metrics',
        'manage_components': 'Manage system components',
        'manage_security': 'Security administration'
    };

    return descriptions[permission] || permission.replace(/_/g, ' ');
};

/**
 * Group permissions by category
 */
export const groupPermissionsByCategory = (permissions) => {
    const categories = {
        'Design Management': ['view_all_designs', 'create_design', 'edit_design', 'delete_design', 'execute_design'],
        'User Management': ['manage_users', 'view_metrics'],
        'System Administration': ['manage_system', 'manage_components', 'manage_security']
    };

    const grouped = {};

    Object.entries(categories).forEach(([category, categoryPerms]) => {
        const userPerms = permissions.filter(perm => categoryPerms.includes(perm));
        if (userPerms.length > 0) {
            grouped[category] = userPerms;
        }
    });

    return grouped;
};