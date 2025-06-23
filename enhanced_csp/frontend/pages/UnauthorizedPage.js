/**
 * Enhanced CSP System - Unauthorized Access Page
 */

import React from 'react';
import { Container, Row, Col, Card, Button, Alert } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { Shield, ArrowLeft, Mail } from 'lucide-react';

import { useUserProfile } from '../hooks/useUserProfile';

const UnauthorizedPage = () => {
    const navigate = useNavigate();
    const { userProfile } = useUserProfile();

    return (
        <Container className="mt-5">
            <Row className="justify-content-center">
                <Col md={8} lg={6}>
                    <Card className="shadow text-center">
                        <Card.Body className="p-5">
                            <Shield size={64} className="text-warning mb-4" />
                            
                            <h3 className="text-warning mb-3">Access Denied</h3>
                            
                            <p className="mb-4">
                                You don't have the necessary permissions to access this resource.
                            </p>

                            {userProfile && (
                                <Alert variant="info" className="mb-4">
                                    <div><strong>Current User:</strong> {userProfile.displayName}</div>
                                    <div><strong>Current Role:</strong> {userProfile.roles?.[0] || 'User'}</div>
                                </Alert>
                            )}

                            <div className="d-grid gap-2">
                                <Button 
                                    variant="primary" 
                                    onClick={() => navigate('/dashboard')}
                                >
                                    <ArrowLeft className="me-2" size={16} />
                                    Return to Dashboard
                                </Button>
                                
                                <Button 
                                    variant="outline-secondary" 
                                    href="mailto:admin@yourcompany.com"
                                >
                                    <Mail className="me-2" size={16} />
                                    Request Access
                                </Button>
                            </div>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
        </Container>
    );
};

export default UnauthorizedPage;