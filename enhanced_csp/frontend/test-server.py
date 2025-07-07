#!/usr/bin/env python3
"""
Enhanced CSP Frontend Test Server
================================
A simple HTTP server for serving frontend files during development.
Includes CORS support and proxy capabilities for API requests.
"""

import os
import sys
import json
import argparse
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import webbrowser
from pathlib import Path

class CSPFrontendHandler(SimpleHTTPRequestHandler):
    """Custom handler for serving frontend files with CORS and API proxy support"""
    
    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        self.directory = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=self.directory, **kwargs)
    
    def end_headers(self):
        """Add CORS headers to all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        self.send_header('Access-Control-Max-Age', '86400')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight CORS requests"""
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests with custom routing"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # API proxy - forward to backend
        if path.startswith('/api/'):
            self.proxy_to_backend()
            return
        
        # Health check endpoint
        if path == '/health':
            self.send_json_response({'status': 'ok', 'service': 'csp-frontend'})
            return
        
        # Config endpoint for frontend
        if path == '/config.js':
            self.serve_config()
            return
        
        # Default routing for SPA
        if path == '/' or path == '':
            path = '/pages/login.html'
        
        # Route common paths
        if path in ['/login', '/dashboard', '/admin']:
            if path == '/login':
                path = '/pages/login.html'
            elif path == '/dashboard':
                path = '/csp_admin_portal.html'
            elif path == '/admin':
                path = '/csp_admin_portal.html'
        
        # Store original path and update for file serving
        original_path = self.path
        self.path = path
        
        try:
            super().do_GET()
        except Exception as e:
            # If file not found, try serving index.html for SPA routing
            if "No such file or directory" in str(e):
                self.path = '/pages/login.html'
                try:
                    super().do_GET()
                except:
                    self.send_error(404, f"File not found: {original_path}")
            else:
                self.send_error(500, f"Server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path.startswith('/api/'):
            self.proxy_to_backend()
        else:
            self.send_error(405, "Method not allowed")
    
    def proxy_to_backend(self):
        """Proxy API requests to the backend server"""
        import urllib.request
        import urllib.error
        
        backend_url = f"http://localhost:8000{self.path}"
        
        try:
            # Read request body if present
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else None
            
            # Create request
            req = urllib.request.Request(backend_url, data=body)
            
            # Copy headers
            for header, value in self.headers.items():
                if header.lower() not in ['host', 'connection']:
                    req.add_header(header, value)
            
            # Make request
            response = urllib.request.urlopen(req, timeout=30)
            
            # Send response
            self.send_response(response.getcode())
            
            # Copy response headers
            for header, value in response.headers.items():
                if header.lower() not in ['connection', 'transfer-encoding']:
                    self.send_header(header, value)
            
            self.end_headers()
            
            # Copy response body
            self.wfile.write(response.read())
            
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {
                'error': 'Backend request failed',
                'status': e.code,
                'message': str(e)
            }
            self.wfile.write(json.dumps(error_response).encode())
        
        except Exception as e:
            self.send_error(502, f"Backend proxy error: {str(e)}")
    
    def serve_config(self):
        """Serve frontend configuration"""
        config = {
            'apiUrl': 'http://localhost:8000',
            'environment': 'development',
            'version': '2.0.0',
            'azure': {
                'clientId': '53537e30-ae6b-48f7-9c7c-4db20fc27850',
                'tenantId': '622a5fe0-fac1-4213-9cf7-d5f6defdf4c4',
                'redirectUri': 'http://localhost:3000'
            },
            'features': {
                'enableAuth': True,
                'enableMonitoring': True,
                'enableAI': True
            }
        }
        
        # Serve as JavaScript module
        js_content = f"window.CSP_CONFIG = {json.dumps(config, indent=2)};"
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/javascript')
        self.end_headers()
        self.wfile.write(js_content.encode())
    
    def send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        """Custom log formatting"""
        print(f"[{self.date_time_string()}] {format % args}")

def check_backend_connection():
    """Check if backend server is running"""
    import urllib.request
    import urllib.error
    
    try:
        urllib.request.urlopen('http://localhost:8000/health', timeout=5)
        return True
    except:
        return False

def print_startup_info(port, backend_available):
    """Print startup information"""
    print("\n" + "="*60)
    print("🚀 Enhanced CSP Frontend Development Server")
    print("="*60)
    print(f"Server running at: http://localhost:{port}")
    print(f"Environment: Development")
    print(f"Backend Status: {'✅ Connected' if backend_available else '❌ Not Available'}")
    print("\n📱 Available URLs:")
    print(f"  • Login Page:    http://localhost:{port}/pages/login.html")
    print(f"  • Admin Portal:  http://localhost:{port}/csp_admin_portal.html")
    print(f"  • Health Check:  http://localhost:{port}/health")
    print(f"  • Config:        http://localhost:{port}/config.js")
    
    if not backend_available:
        print("\n⚠️  Backend server not detected at localhost:8000")
        print("   API requests will fail until backend is started")
        print("   Run: cd ../backend && python main.py")
    
    print("\n🛠️  Development Features:")
    print("  • Auto CORS headers")
    print("  • API request proxying")
    print("  • SPA routing support")
    print("  • Azure AD integration")
    
    print(f"\n🔧 Press Ctrl+C to stop the server")
    print("="*60)

def main():
    """Main server function"""
    parser = argparse.ArgumentParser(description='CSP Frontend Development Server')
    parser.add_argument('--port', '-p', type=int, default=3000,
                       help='Port to serve on (default: 3000)')
    parser.add_argument('--host', default='localhost',
                       help='Host to bind to (default: localhost)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Don\'t open browser automatically')
    parser.add_argument('--backend-check', action='store_true', default=True,
                       help='Check backend availability (default: True)')
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / 'pages').exists():
        print("❌ Error: This script should be run from the frontend directory")
        print(f"Current directory: {current_dir}")
        print("Expected to find: ./pages/ directory")
        print("\nPlease run from: /home/mate/PAIN/csp-agent-network/csp-agent-network-1/enhanced_csp/frontend")
        sys.exit(1)
    
    # Check backend availability
    backend_available = False
    if args.backend_check:
        print("🔍 Checking backend connection...")
        backend_available = check_backend_connection()
    
    # Start server
    try:
        server = HTTPServer((args.host, args.port), CSPFrontendHandler)
        
        # Print startup info
        print_startup_info(args.port, backend_available)
        
        # Open browser if requested
        if not args.no_browser:
            def open_browser():
                import time
                time.sleep(1)  # Wait for server to start
                webbrowser.open(f'http://localhost:{args.port}/pages/login.html')
            
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
        
        # Start serving
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ Error: Port {args.port} is already in use")
            print(f"Try a different port: python test-server.py --port {args.port + 1}")
        else:
            print(f"\n❌ Error starting server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
