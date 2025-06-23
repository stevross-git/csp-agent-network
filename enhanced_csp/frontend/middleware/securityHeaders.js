/**
 * Enhanced CSP System - Security Headers Middleware
 * Client-side security header management and CSP implementation
 */

class SecurityHeadersMiddleware {
    constructor() {
        this.cspPolicies = {
            'default-src': ["'self'"],
            'script-src': [
                "'self'",
                "'unsafe-inline'", // Required for React development
                "https://login.microsoftonline.com",
                "https://graph.microsoft.com",
                "https://cdnjs.cloudflare.com"
            ],
            'style-src': [
                "'self'",
                "'unsafe-inline'", // Required for Bootstrap and styled-components
                "https://fonts.googleapis.com",
                "https://cdnjs.cloudflare.com"
            ],
            'font-src': [
                "'self'",
                "https://fonts.gstatic.com",
                "https://cdnjs.cloudflare.com",
                "data:"
            ],
            'img-src': [
                "'self'",
                "https://graph.microsoft.com",
                "https://login.microsoftonline.com",
                "data:",
                "blob:"
            ],
            'connect-src': [
                "'self'",
                "https://login.microsoftonline.com",
                "https://graph.microsoft.com",
                "wss://localhost:*", // WebSocket connections
                process.env.REACT_APP_CSP_API_URL || "http://localhost:8000"
            ],
            'frame-src': [
                "'self'",
                "https://login.microsoftonline.com"
            ],
            'object-src': ["'none'"],
            'base-uri': ["'self'"],
            'form-action': ["'self'"],
            'upgrade-insecure-requests': []
        };

        this.securityHeaders = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=()'
        };
    }

    /**
     * Generate Content Security Policy string
     */
    generateCSP() {
        const policies = Object.entries(this.cspPolicies)
            .map(([directive, sources]) => {
                if (sources.length === 0) {
                    return directive;
                }
                return `${directive} ${sources.join(' ')}`;
            })
            .join('; ');

        return policies;
    }

    /**
     * Apply security headers to fetch requests
     */
    applySecurityHeaders(request) {
        const headers = new Headers(request.headers);

        // Add security headers
        Object.entries(this.securityHeaders).forEach(([header, value]) => {
            headers.set(header, value);
        });

        // Add CSP header
        headers.set('Content-Security-Policy', this.generateCSP());

        // Add request ID for tracking
        headers.set('X-Request-ID', this.generateRequestId());

        // Add timestamp
        headers.set('X-Request-Time', new Date().toISOString());

        return new Request(request, { headers });
    }

    /**
     * Create secure fetch wrapper
     */
    secureFetch(url, options = {}) {
        // Create request with security headers
        const request = new Request(url, options);
        const secureRequest = this.applySecurityHeaders(request);

        // Add additional security validations
        this.validateRequest(secureRequest);

        return fetch(secureRequest);
    }

    /**
     * Validate request security
     */
    validateRequest(request) {
        const url = new URL(request.url);

        // Ensure HTTPS in production
        if (process.env.NODE_ENV === 'production' && url.protocol !== 'https:') {
            console.warn('Non-HTTPS request in production:', url.href);
        }

        // Validate allowed domains
        const allowedDomains = [
            'localhost',
            'login.microsoftonline.com',
            'graph.microsoft.com',
            new URL(process.env.REACT_APP_CSP_API_URL || 'http://localhost:8000').hostname
        ];

        if (!allowedDomains.includes(url.hostname)) {
            console.warn('Request to non-whitelisted domain:', url.hostname);
        }

        // Check for suspicious patterns
        this.checkSuspiciousPatterns(request);
    }

    /**
     * Check for suspicious request patterns
     */
    checkSuspiciousPatterns(request) {
        const url = request.url.toLowerCase();
        const suspiciousPatterns = [
            /\.\.\//, // Path traversal
            /<script/i, // XSS attempts
            /javascript:/i, // JavaScript injection
            /vbscript:/i, // VBScript injection
            /on\w+\s*=/i, // Event handler injection
        ];

        suspiciousPatterns.forEach(pattern => {
            if (pattern.test(url)) {
                console.error('Suspicious request pattern detected:', pattern, url);
                throw new Error('Potentially malicious request blocked');
            }
        });
    }

    /**
     * Generate unique request ID
     */
    generateRequestId() {
        return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Sanitize user input
     */
    sanitizeInput(input) {
        if (typeof input !== 'string') {
            return input;
        }

        // Remove potential XSS vectors
        return input
            .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
            .replace(/javascript:/gi, '')
            .replace(/vbscript:/gi, '')
            .replace(/on\w+\s*=/gi, '')
            .replace(/style\s*=/gi, '')
            .trim();
    }

    /**
     * Validate and sanitize form data
     */
    sanitizeFormData(formData) {
        const sanitized = {};

        for (const [key, value] of Object.entries(formData)) {
            // Sanitize key
            const cleanKey = this.sanitizeInput(key);
            
            // Sanitize value
            let cleanValue = value;
            if (typeof value === 'string') {
                cleanValue = this.sanitizeInput(value);
            } else if (Array.isArray(value)) {
                cleanValue = value.map(item => 
                    typeof item === 'string' ? this.sanitizeInput(item) : item
                );
            }

            sanitized[cleanKey] = cleanValue;
        }

        return sanitized;
    }

    /**
     * Create secure WebSocket connection
     */
    createSecureWebSocket(url, protocols = []) {
        // Validate WebSocket URL
        const wsUrl = new URL(url);
        
        if (process.env.NODE_ENV === 'production' && wsUrl.protocol !== 'wss:') {
            throw new Error('Secure WebSocket (WSS) required in production');
        }

        // Create WebSocket with security considerations
        const ws = new WebSocket(url, protocols);

        // Add security event handlers
        ws.addEventListener('open', () => {
            console.log('Secure WebSocket connection established');
        });

        ws.addEventListener('error', (error) => {
            console.error('WebSocket security error:', error);
        });

        // Wrap send method with validation
        const originalSend = ws.send.bind(ws);
        ws.send = (data) => {
            try {
                // Validate data before sending
                if (typeof data === 'string') {
                    const sanitized = this.sanitizeInput(data);
                    originalSend(sanitized);
                } else {
                    originalSend(data);
                }
            } catch (error) {
                console.error('WebSocket send validation failed:', error);
                throw error;
            }
        };

        return ws;
    }

    /**
     * Set up global security policies
     */
    initializeGlobalSecurity() {
        // Override default fetch with secure version
        const originalFetch = window.fetch;
        window.fetch = (url, options = {}) => {
            return this.secureFetch(url, options);
        };

        // Set up CSP violation reporting
        document.addEventListener('securitypolicyviolation', (event) => {
            console.error('CSP Violation:', {
                directive: event.violatedDirective,
                blockedURI: event.blockedURI,
                lineNumber: event.lineNumber,
                sourceFile: event.sourceFile
            });

            // Report to monitoring service (if available)
            this.reportSecurityViolation(event);
        });

        // Set up error boundary for security errors
        window.addEventListener('error', (event) => {
            if (event.message.includes('security') || event.message.includes('CSP')) {
                console.error('Security-related error:', event);
                this.reportSecurityError(event);
            }
        });

        console.log('🛡️ Security middleware initialized');
    }

    /**
     * Report security violations
     */
    reportSecurityViolation(event) {
        // This would typically send to a monitoring service
        const violation = {
            type: 'csp_violation',
            directive: event.violatedDirective,
            blockedURI: event.blockedURI,
            documentURI: event.documentURI,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent
        };

        // In production, send to monitoring service
        if (process.env.NODE_ENV === 'production') {
            // Example: send to monitoring endpoint
            // this.secureFetch('/api/security/report', {
            //     method: 'POST',
            //     body: JSON.stringify(violation)
            // });
        }

        console.warn('Security violation reported:', violation);
    }

    /**
     * Report security errors
     */
    reportSecurityError(error) {
        const errorReport = {
            type: 'security_error',
            message: error.message,
            filename: error.filename,
            lineno: error.lineno,
            colno: error.colno,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent
        };

        console.error('Security error reported:', errorReport);
    }
}

export const securityHeaders = new SecurityHeadersMiddleware();

export default securityHeaders;