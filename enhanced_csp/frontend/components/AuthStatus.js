/**
 * Enhanced CSP System - Authentication Status Component
 * Real-time authentication status indicator
 */

import React, { useState, useEffect } from 'react';
import { Badge, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { 
    useMsal, 
    useIsAuthenticated,
    useMsalAuthentication 
} from '@azure/msal-react';
import { 
    CheckCircle, 
    XCircle, 
    AlertTriangle, 
    Clock,
    Shield
} from 'lucide-react';

import { useUserProfile } from '../hooks/useUserProfile';

const AuthStatus = () => {
    const isAuthenticated = useIsAuthenticated();
    const { accounts, instance } = useMsal();
    const { userProfile } = useUserProfile();
    const [tokenStatus, setTokenStatus] = useState('checking');
    const [lastCheck, setLastCheck] = useState(new Date());

    useEffect(() => {
        checkTokenStatus();
        
        // Check token status every 5 minutes
        const interval = setInterval(checkTokenStatus, 5 * 60 * 1000);
        return () => clearInterval(interval);
    }, [isAuthenticated, accounts]);

    const checkTokenStatus = async () => {
        if (!isAuthenticated || !accounts.length) {
            setTokenStatus('unauthenticated');
            return;
        }

        try {
            const request = {
                scopes: ["User.Read"],
                account: accounts[0],
                forceRefresh: false
            };

            await instance.acquireTokenSilent(request);
            setTokenStatus('valid');
            setLastCheck(new Date());
        } catch (error) {
            if (error.name === 'InteractionRequiredAuthError') {
                setTokenStatus('interaction_required');
            } else {
                setTokenStatus('error');
            }
            console.warn('Token validation failed:', error);
        }
    };

    const getStatusConfig = () => {
        switch (tokenStatus) {
            case 'valid':
                return {
                    variant: 'success',
                    icon: <CheckCircle size={14} />,
                    text: 'Authenticated',
                    tooltip: `Authentication valid. Last checked: ${lastCheck.toLocaleTimeString()}`
                };
            case 'interaction_required':
                return {
                    variant: 'warning',
                    icon: <AlertTriangle size={14} />,
                    text: 'Renewal Required',
                    tooltip: 'Token needs renewal. Click to refresh authentication.'
                };
            case 'error':
                return {
                    variant: 'danger',
                    icon: <XCircle size={14} />,
                    text: 'Auth Error',
                    tooltip: 'Authentication error occurred. Please sign in again.'
                };
            case 'checking':
                return {
                    variant: 'info',
                    icon: <Clock size={14} />,
                    text: 'Checking...',
                    tooltip: 'Verifying authentication status...'
                };
            default:
                return {
                    variant: 'secondary',
                    icon: <Shield size={14} />,
                    text: 'Not Authenticated',
                    tooltip: 'Please sign in to access the system'
                };
        }
    };

    const config = getStatusConfig();

    const handleStatusClick = async () => {
        if (tokenStatus === 'interaction_required') {
            try {
                const request = {
                    scopes: ["User.Read"],
                    account: accounts[0]
                };
                
                await instance.acquireTokenRedirect(request);
            } catch (error) {
                console.error('Token refresh failed:', error);
            }
        }
    };

    const tooltip = (
        <Tooltip>
            <div>
                <strong>{config.text}</strong>
                <br />
                {config.tooltip}
                {userProfile && (
                    <>
                        <br />
                        <small>User: {userProfile.displayName}</small>
                        <br />
                        <small>Role: {userProfile.roles?.[0] || 'User'}</small>
                    </>
                )}
            </div>
        </Tooltip>
    );

    return (
        <OverlayTrigger placement="bottom" overlay={tooltip}>
            <Badge 
                bg={config.variant} 
                className="d-flex align-items-center cursor-pointer auth-status-badge"
                onClick={handleStatusClick}
                style={{ cursor: tokenStatus === 'interaction_required' ? 'pointer' : 'default' }}
            >
                {config.icon}
                <span className="ms-1 d-none d-md-inline">{config.text}</span>
            </Badge>
        </OverlayTrigger>
    );
};

export default AuthStatus;