#!/bin/bash
# debug-auth-loop.sh - Debug authentication loop issues

FRONTEND_DIR="/home/mate/PAIN/csp-agent-network/csp-agent-network-1/enhanced_csp/frontend"
JS_DIR="$FRONTEND_DIR/js"

echo "🔍 Debugging Authentication Loop Issue"
echo "====================================="
echo ""

# Check if auth-protection.js exists
echo "1. 📁 Checking auth-protection.js file..."
if [ -f "$JS_DIR/auth-protection.js" ]; then
    echo "   ✅ auth-protection.js exists"
    file_size=$(stat -c%s "$JS_DIR/auth-protection.js")
    echo "   📊 File size: $file_size bytes"
    if [ $file_size -lt 1000 ]; then
        echo "   ⚠️  File seems too small - may be empty or incomplete"
    fi
else
    echo "   ❌ auth-protection.js MISSING!"
    echo "   🔧 This is likely the cause of the loop"
fi

echo ""

# Check if pages have the right script references
echo "2. 📄 Checking script references in pages..."
sample_page="$FRONTEND_DIR/pages/admin.html"
if [ -f "$sample_page" ]; then
    if grep -q "auth-protection.js" "$sample_page"; then
        echo "   ✅ admin.html references auth-protection.js"
    else
        echo "   ❌ admin.html missing auth-protection.js reference"
    fi
    
    if grep -q "auth-wrapper.js" "$sample_page"; then
        echo "   ⚠️  admin.html still has old auth-wrapper.js reference"
    fi
else
    echo "   ❌ admin.html not found"
fi

echo ""

# Check login page
echo "3. 🔑 Checking login page..."
login_page="$FRONTEND_DIR/pages/login.html"
if [ -f "$login_page" ]; then
    echo "   ✅ login.html exists"
    if grep -q "auth-protection.js" "$login_page"; then
        echo "   ⚠️  login.html has auth-protection.js (should NOT have it)"
    else
        echo "   ✅ login.html correctly excludes auth-protection.js"
    fi
else
    echo "   ❌ login.html not found"
fi

echo ""

# Check for conflicting auth systems
echo "4. 🔄 Checking for conflicting authentication systems..."
auth_wrapper_exists=false
for file in "$FRONTEND_DIR/pages"/*.html; do
    if [ -f "$file" ] && grep -q "auth-wrapper.js" "$file"; then
        filename=$(basename "$file")
        echo "   ⚠️  $filename still references old auth-wrapper.js"
        auth_wrapper_exists=true
    fi
done

if [ "$auth_wrapper_exists" = false ]; then
    echo "   ✅ No conflicting auth-wrapper.js references found"
fi

echo ""

# Check for MSAL references
echo "5. 🌐 Checking for old MSAL references..."
msal_found=false
for file in "$FRONTEND_DIR/pages"/*.html; do
    if [ -f "$file" ] && grep -q "msal-browser" "$file"; then
        filename=$(basename "$file")
        echo "   ⚠️  $filename still has MSAL references"
        msal_found=true
    fi
done

if [ "$msal_found" = false ]; then
    echo "   ✅ No MSAL references found"
fi

echo ""
echo "📋 Summary & Recommendations:"
echo "=============================="

if [ ! -f "$JS_DIR/auth-protection.js" ]; then
    echo "🚨 CRITICAL: auth-protection.js is missing!"
    echo "   This is definitely causing the login loop."
    echo "   Action: Create the auth-protection.js file immediately."
elif [ $(stat -c%s "$JS_DIR/auth-protection.js") -lt 1000 ]; then
    echo "🚨 CRITICAL: auth-protection.js appears to be empty or incomplete!"
    echo "   Action: Replace with the complete auth-protection.js code."
else
    echo "✅ auth-protection.js file looks good"
fi

if [ "$auth_wrapper_exists" = true ]; then
    echo "⚠️  Conflicting auth systems detected"
    echo "   Action: Remove auth-wrapper.js references from pages"
fi

if [ "$msal_found" = true ]; then
    echo "⚠️  Old MSAL references found"
    echo "   Action: Remove MSAL references from pages"
fi

echo ""
echo "🔧 Quick Fix Commands:"
echo "====================="

if [ ! -f "$JS_DIR/auth-protection.js" ] || [ $(stat -c%s "$JS_DIR/auth-protection.js") -lt 1000 ]; then
    echo "# 1. Create/fix auth-protection.js file:"
    echo "nano $JS_DIR/auth-protection.js"
    echo ""
fi

if [ "$auth_wrapper_exists" = true ]; then
    echo "# 2. Remove old auth-wrapper references:"
    echo "sed -i '/auth-wrapper\.js/d' $FRONTEND_DIR/pages/*.html"
    echo ""
fi

if [ "$msal_found" = true ]; then
    echo "# 3. Remove MSAL references:"
    echo "sed -i '/msal-browser/d' $FRONTEND_DIR/pages/*.html"
    echo ""
fi

echo "# 4. Test in browser with console open:"
echo "# Go to: http://localhost:3000/pages/admin.html"
echo "# Check browser console for error messages"