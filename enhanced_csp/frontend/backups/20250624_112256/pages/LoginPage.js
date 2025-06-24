/**
 * Enhanced CSP System - Azure AD Login Page
 * Modern login interface with Azure AD integration
 */

import React from 'react';
import { Container, Row, Col, Card, Button, Alert } from 'react-bootstrap';
import { useMsal } from '@azure/msal-react';
import { Shield, LogIn } from 'lucide-react';

import { loginRequest } from '../config/authConfig';

const LoginPage = () => {
    const { instance } = useMsal();

    const handleLogin = () => {
        instance.loginRedirect(loginRequest).catch(e => {
            console.error('Login failed:', e);
        });
    };

    return (
        <Container className="mt-5">
            <Row className="justify-content-center">
                <Col md={6} lg={4}>
                    <Card className="shadow">
                        <Card.Body className="text-center p-5">
                            <div className="mb-4">
                                <Shield size={64} className="text-primary" />
                            </div>
                            
                            <h3 className="mb-3">Enhanced CSP System</h3>
                            <p className="text-muted mb-4">
                                Sign in with your Microsoft account to access the system
                            </p>

                            <Button 
                                variant="primary" 
                                size="lg" 
                                onClick={handleLogin}
                                className="w-100 mb-3"
                            >
                                <LogIn className="me-2" size={20} />
                                Sign in with Microsoft
                            </Button>

                            <Alert variant="info" className="mt-4">
                                <small>
                                    This application uses Azure Active Directory for secure authentication.
                                    Contact your administrator if you need access.
                                </small>
                            </Alert>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
        </Container>
    );
};

export default LoginPage;