// config/roles.js
export const USER_ROLES = {
    SUPER_ADMIN: 'super_admin',
    ADMIN: 'admin',
    DEVELOPER: 'developer',
    ANALYST: 'analyst',
    USER: 'user'
};

export const ROLE_PERMISSIONS = {
    [USER_ROLES.SUPER_ADMIN]: [
        'system.admin',
        'user.manage',
        'ai.manage',
        'quantum.manage',
        'blockchain.manage',
        'security.admin',
        'reports.admin',
        'settings.admin'
    ],
    [USER_ROLES.ADMIN]: [
        'system.view',
        'user.manage',
        'ai.manage',
        'quantum.view',
        'blockchain.view',
        'reports.view',
        'settings.view'
    ],
    [USER_ROLES.DEVELOPER]: [
        'system.view',
        'ai.manage',
        'quantum.view',
        'blockchain.view',
        'reports.view'
    ],
    [USER_ROLES.ANALYST]: [
        'system.view',
        'ai.view',
        'quantum.view',
        'reports.view'
    ],
    [USER_ROLES.USER]: [
        'system.view'
    ]
};

// Role assignment based on Azure AD groups or user attributes
export function getUserRole(userInfo) {
    try {
        const email = userInfo.mail || userInfo.upn || userInfo.username;
        const groups = userInfo.groups || [];
        
        // Check Azure AD groups (configure these in Azure AD)
        if (groups.includes('CSP-SuperAdmins')) return USER_ROLES.SUPER_ADMIN;
        if (groups.includes('CSP-Admins')) return USER_ROLES.ADMIN;
        if (groups.includes('CSP-Developers')) return USER_ROLES.DEVELOPER;
        if (groups.includes('CSP-Analysts')) return USER_ROLES.ANALYST;
        
        // Fallback to email domain checking
        if (email?.includes('admin@')) return USER_ROLES.ADMIN;
        if (email?.includes('dev@') || email?.includes('developer@')) return USER_ROLES.DEVELOPER;
        if (email?.includes('analyst@')) return USER_ROLES.ANALYST;
        
        // Default role
        return USER_ROLES.USER;
    } catch (error) {
        console.error('Error determining user role:', error);
        return USER_ROLES.USER;
    }
}

// Check if user has specific permission
export function hasPermission(userRole, permission) {
    const permissions = ROLE_PERMISSIONS[userRole] || [];
    return permissions.includes(permission);
}

// Check if user has any of the specified permissions
export function hasAnyPermission(userRole, permissionList) {
    const userPermissions = ROLE_PERMISSIONS[userRole] || [];
    return permissionList.some(permission => userPermissions.includes(permission));
}

// Check if user has all specified permissions
export function hasAllPermissions(userRole, permissionList) {
    const userPermissions = ROLE_PERMISSIONS[userRole] || [];
    return permissionList.every(permission => userPermissions.includes(permission));
}

// Get role hierarchy level (higher number = more permissions)
export function getRoleLevel(role) {
    const levels = {
        [USER_ROLES.USER]: 1,
        [USER_ROLES.ANALYST]: 2,
        [USER_ROLES.DEVELOPER]: 3,
        [USER_ROLES.ADMIN]: 4,
        [USER_ROLES.SUPER_ADMIN]: 5
    };
    return levels[role] || 1;
}

// Check if user role has sufficient level for required role
export function hasRoleLevel(userRole, requiredRole) {
    return getRoleLevel(userRole) >= getRoleLevel(requiredRole);
}

// Get all permissions for a role
export function getRolePermissions(role) {
    return ROLE_PERMISSIONS[role] || [];
}

// Role display utilities
export function formatRoleName(role) {
    return role.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
}

export function getRoleEmoji(role) {
    const emojis = {
        [USER_ROLES.SUPER_ADMIN]: '🔱',
        [USER_ROLES.ADMIN]: '👑',
        [USER_ROLES.DEVELOPER]: '🛠️',
        [USER_ROLES.ANALYST]: '📊',
        [USER_ROLES.USER]: '👤'
    };
    return emojis[role] || '👤';
}

export function getRoleColor(role) {
    const colors = {
        [USER_ROLES.SUPER_ADMIN]: '#8B5CF6',
        [USER_ROLES.ADMIN]: '#EF4444',
        [USER_ROLES.DEVELOPER]: '#3B82F6',
        [USER_ROLES.ANALYST]: '#10B981',
        [USER_ROLES.USER]: '#6B7280'
    };
    return colors[role] || '#6B7280';
}

// Make available globally for backward compatibility
if (typeof window !== 'undefined') {
    window.USER_ROLES = USER_ROLES;
    window.ROLE_PERMISSIONS = ROLE_PERMISSIONS;
    window.getUserRole = getUserRole;
    window.hasPermission = hasPermission;
}

export default {
    USER_ROLES,
    ROLE_PERMISSIONS,
    getUserRole,
    hasPermission,
    hasAnyPermission,
    hasAllPermissions,
    getRoleLevel,
    hasRoleLevel,
    getRolePermissions,
    formatRoleName,
    getRoleEmoji,
    getRoleColor
};