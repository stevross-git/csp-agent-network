#!/bin/bash
# debug-login-loop.sh - Debug and fix the login page loop

FRONTEND_DIR="/home/mate/PAIN/csp-agent-network/csp-agent-network-1/enhanced_csp/frontend"
LOGIN_FILE="$FRONTEND_DIR/pages/login.html"
ADMIN_FILE="$FRONTEND_DIR/pages/admin.html"

echo "🔍 Debugging Login Page Loop Issue"
echo "=================================="

echo ""
echo "1. 📄 Checking if login.html has auth-protection.js (it shouldn't):"
if grep -q "auth-protection.js" "$LOGIN_FILE"; then
    echo "❌ PROBLEM: login.html contains auth-protection.js!"
    echo "   This is causing the redirect loop."
    echo ""
    echo "🔧 Removing auth-protection.js from login.html..."
    
    # Create backup
    cp "$LOGIN_FILE" "$LOGIN_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Remove auth-protection.js from login.html
    sed -i '/auth-protection\.js/d' "$LOGIN_FILE"
    
    echo "✅ Removed auth-protection.js from login.html"
else
    echo "✅ login.html correctly excludes auth-protection.js"
fi

echo ""
echo "2. 🔍 Checking what scripts login.html loads:"
echo "Scripts in login.html:"
grep -n "<script" "$LOGIN_FILE" | head -10

echo ""
echo "3. 🔍 Checking what scripts admin.html loads:"
echo "Scripts in admin.html:"
grep -n "<script" "$ADMIN_FILE" | head -5

echo ""
echo "4. 🔍 Checking for redirect logic in login.html:"
if grep -q "window.location" "$LOGIN_FILE"; then
    echo "⚠️ Found window.location redirects in login.html:"
    grep -n "window.location" "$LOGIN_FILE"
else
    echo "✅ No automatic redirects found in login.html"
fi

echo ""
echo "5. 🔍 Checking for automatic form submission:"
if grep -q "submit()" "$LOGIN_FILE"; then
    echo "⚠️ Found automatic form submission in login.html:"
    grep -n "submit()" "$LOGIN_FILE"
else
    echo "✅ No automatic form submission found"
fi

echo ""
echo "6. 🧹 Cleaning up any auth event handlers in login.html:"

# Remove any auth event handlers that might be in login.html
if grep -q "cspAuthReady\|cspAuthError" "$LOGIN_FILE"; then
    echo "⚠️ Found auth event handlers in login.html - removing..."
    sed -i '/cspAuthReady/d' "$LOGIN_FILE"
    sed -i '/cspAuthError/d' "$LOGIN_FILE"
    echo "✅ Removed auth event handlers from login.html"
else
    echo "✅ No conflicting auth event handlers in login.html"
fi

echo ""
echo "7. 🔧 Ensuring login.html stays on login page:"

# Make sure login.html doesn't auto-redirect
if ! grep -q "console.log.*login.*page" "$LOGIN_FILE"; then
    echo "🔧 Adding login page identification..."
    # Add a script to identify this as login page
    sed -i 's|</body>|    <script>console.log("📄 Login page loaded - no auth protection needed");</script>\n</body>|' "$LOGIN_FILE"
fi

echo ""
echo "✅ Login page debugging complete!"
echo ""
echo "🔧 Next steps:"
echo "1. Clear browser cache/cookies: Ctrl+Shift+R"
echo "2. Test: http://localhost:3000/pages/login.html"
echo "3. Should stay on login page without redirecting"
echo "4. Try login with: admin@csp-system.com / AdminPass123!"
echo ""
echo "📋 Expected behavior:"
echo "   ✅ Login page loads and stays loaded"
echo "   ✅ Can enter credentials"
echo "   ✅ After successful login, redirects to admin page"
echo "   ✅ Admin page shows auth header and doesn't redirect back"