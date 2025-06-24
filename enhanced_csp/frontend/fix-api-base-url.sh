#!/bin/bash
# fix-api-base-url.sh - Fix API base URL in auth-protection.js

AUTH_FILE="/home/mate/PAIN/csp-agent-network/csp-agent-network-1/enhanced_csp/frontend/js/auth-protection.js"

echo "🔧 Fixing API Base URL in auth-protection.js"
echo "============================================"

# Create backup
cp "$AUTH_FILE" "$AUTH_FILE.backup.$(date +%Y%m%d_%H%M%S)"
echo "✅ Created backup of auth-protection.js"

# Fix the API base URL
sed -i 's/apiBaseUrl: window\.location\.origin,/apiBaseUrl: "http:\/\/localhost:8000",/' "$AUTH_FILE"

echo "✅ Updated API base URL to http://localhost:8000"

# Verify the change
echo ""
echo "🔍 Verification:"
grep -n "apiBaseUrl" "$AUTH_FILE"

echo ""
echo "🎯 The auth-protection.js will now call the correct backend URLs:"
echo "   ✅ http://localhost:8000/api/auth/validate"
echo "   ✅ http://localhost:8000/api/auth/local/login"
echo "   ✅ http://localhost:8000/api/auth/logout"
echo ""
echo "🔧 Next steps:"
echo "1. Restart your backend: python3 -m backend.main"
echo "2. Test the frontend: http://localhost:3000/pages/admin.html"
echo "3. Use credentials: admin@csp-system.com / AdminPass123!"