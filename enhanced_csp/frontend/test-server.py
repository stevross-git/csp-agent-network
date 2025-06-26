# Add this to your existing test-server.py or main server file

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from urllib.parse import urlparse, parse_qs

class EnhancedCSPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Handle API endpoints
        if path == '/api/pages/available':
            self.send_pages_data()
        elif path.startswith('/pages/') and path.endswith('.html'):
            self.serve_html_file(path)
        elif path.startswith('/js/') or path.startswith('/css/'):
            self.serve_static_file(path)
        else:
            self.serve_file(path)
    
    def send_pages_data(self):
        """Send available pages data for navigation"""
        pages_data = {
            "pages": [
                {"name": "dashboard", "title": "Dashboard", "category": "core"},
                {"name": "admin", "title": "Admin Portal", "category": "admin"},
                {"name": "designer", "title": "Visual Designer", "category": "core"},
                {"name": "monitoring", "title": "Monitoring", "category": "monitoring"},
                {"name": "ai-agents", "title": "AI Agents", "category": "ai"},
                {"name": "settings", "title": "Settings", "category": "admin"},
                {"name": "security", "title": "Security", "category": "admin"},
                {"name": "logs", "title": "System Logs", "category": "admin"},
                {"name": "users", "title": "User Management", "category": "admin"},
                {"name": "roles", "title": "Role Management", "category": "admin"}
            ],
            "categories": {
                "core": ["dashboard", "designer"],
                "admin": ["admin", "settings", "security", "logs", "users", "roles"],
                "monitoring": ["monitoring"],
                "ai": ["ai-agents"]
            },
            "icons": {
                "dashboard": "📊",
                "admin": "👨‍💼",
                "designer": "🎨",
                "monitoring": "📈",
                "ai-agents": "🤖",
                "settings": "⚙️",
                "security": "🔐",
                "logs": "📋",
                "users": "👥",
                "roles": "🏷️"
            }
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
        
        response = json.dumps(pages_data)
        self.wfile.write(response.encode('utf-8'))
    
    def serve_html_file(self, path):
        """Serve HTML files from pages directory"""
        file_path = f".{path}"
        if os.path.exists(file_path):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_404()
    
    def serve_static_file(self, path):
        """Serve static JS/CSS files"""
        file_path = f".{path}"
        if os.path.exists(file_path):
            # Determine content type
            if path.endswith('.js'):
                content_type = 'application/javascript'
            elif path.endswith('.css'):
                content_type = 'text/css'
            else:
                content_type = 'text/plain'
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_404()
    
    def serve_file(self, path):
        """Serve other files"""
        if path == '/':
            path = '/pages/admin.html'
        
        file_path = f".{path}"
        if os.path.exists(file_path):
            self.send_response(200)
            
            # Determine content type
            if path.endswith('.html'):
                content_type = 'text/html'
            elif path.endswith('.js'):
                content_type = 'application/javascript'
            elif path.endswith('.css'):
                content_type = 'text/css'
            elif path.endswith('.json'):
                content_type = 'application/json'
            else:
                content_type = 'text/plain'
            
            self.send_header('Content-Type', content_type)
            self.end_headers()
            
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_404()
    
    def send_404(self):
        """Send 404 error"""
        self.send_response(404)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        
        error_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>404 - File Not Found</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                h1 { color: #e74c3c; }
                p { color: #666; }
                a { color: #3498db; text-decoration: none; }
            </style>
        </head>
        <body>
            <h1>404 - File Not Found</h1>
            <p>The requested file could not be found.</p>
            <p><a href="/pages/admin.html">← Go to Admin Portal</a></p>
        </body>
        </html>
        """
        self.wfile.write(error_html.encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

def run_server(port=3000):
    """Run the development server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, EnhancedCSPHandler)
    print(f"🚀 Enhanced CSP Server running on http://localhost:{port}")
    print(f"📄 Admin Portal: http://localhost:{port}/pages/admin.html")
    print(f"🔧 API Endpoint: http://localhost:{port}/api/pages/available")
    print("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        httpd.server_close()

if __name__ == '__main__':
    run_server()