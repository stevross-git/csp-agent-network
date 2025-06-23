/**
 * Enhanced CSP System - Loading Spinner Component
 */

import React from 'react';
import { Spinner } from 'react-bootstrap';
import { Shield } from 'lucide-react';

const LoadingSpinner = ({ message = 'Loading...', size = 'lg' }) => {
    return (
        <div className="d-flex flex-column align-items-center justify-content-center" style={{ minHeight: '200px' }}>
            <Shield className="text-primary mb-3" size={48} />
            <Spinner animation="border" variant="primary" size={size} className="mb-3" />
            <div className="text-muted">{message}</div>
        </div>
    );
};

export default LoadingSpinner;