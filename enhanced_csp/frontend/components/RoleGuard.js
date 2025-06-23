/**
 * Enhanced CSP System - Role Guard Component
 * Fine-grained role-based access control
 */

import React from 'react';
import { Alert } from 'react-bootstrap';
import { Shield, AlertTriangle } from 'lucide-react';

import { useUserProfile } from '../hooks/useUserProfile';
import { checkUserRole, checkUserPermission } from '../utils/roleUtils';

const RoleGuard = ({ 
    children, 
    allowedRoles = [], 
    requiredPermissions = [],
    requireAll = false, // true: user must have ALL roles/permissions, false: user must have ANY
    fallback = null,
    showError = true 
}) => {
    const { userProfile, loading } = useUserProfile();

    if (loading) {
        return fallback || (
            <div className="d-flex align-items-center text-muted">
                <Shield className="me-2" size={16} />
                Checking permissions...
            </div>
        );
    }

    if (!userProfile) {
        return showError ? (
            <Alert variant="warning" className="d-flex align-items-center">
                <AlertTriangle className="me-2" size={16} />
                Unable to verify user permissions
            </Alert>
        ) : fallback;
    }

    // Check roles
    const roleCheck = allowedRoles.length === 0 || (
        requireAll 
            ? allowedRoles.every(role => checkUserRole(userProfile, role))
            : allowedRoles.some(role => checkUserRole(userProfile, role))
    );

    // Check permissions
    const permissionCheck = requiredPermissions.length === 0 || (
        requireAll
            ? requiredPermissions.every(permission => checkUserPermission(userProfile, permission))
            : requiredPermissions.some(permission => checkUserPermission(userProfile, permission))
    );

    const hasAccess = roleCheck && permissionCheck;

    if (!hasAccess) {
        if (showError) {
            return (
                <Alert variant="danger" className="d-flex align-items-center">
                    <Shield className="me-2" size={16} />
                    <div>
                        <strong>Access Denied:</strong> You don't have the required permissions to view this content.
                        {allowedRoles.length > 0 && (
                            <div className="mt-1">
                                <small>Required roles: {allowedRoles.join(', ')}</small>
                            </div>
                        )}
                        {requiredPermissions.length > 0 && (
                            <div className="mt-1">
                                <small>Required permissions: {requiredPermissions.join(', ')}</small>
                            </div>
                        )}
                    </div>
                </Alert>
            );
        }
        return fallback;
    }

    return children;
};

export default RoleGuard;