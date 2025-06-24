#!/bin/bash
# deploy-auth-protection.sh
# Adds local authentication protection to all HTML pages

PROJECT_ROOT="/home/mate/PAIN/csp-agent-network/csp-agent-network-1"
FRONTEND_DIR="$PROJECT_ROOT/enhanced_csp/frontend"
PAGES_DIR="$FRONTEND_DIR/pages"

echo "🔐 Deploying Local Authentication to All Pages..."
echo "=============================================="

# Function to add auth to a single page
add_auth_protection_to_page() {
    local file="$1"
    local filename=$(basename "$file")
    
    # Skip login page
    if [[ "$filename" == "login.html" ]]; then
        echo "⏭️  Skipping login.html"
        return
    fi
    
    echo "🔧 Processing: $filename"
    
    # Backup original
    cp "$file" "$file.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Check if already has auth-protection
    if grep -q "auth-protection.js" "$file"; then
        echo "⚠️  $filename already has auth-protection - updating..."
        # Remove old auth-protection and auth-wrapper references
        sed -i '/auth-protection\.js/d' "$file"
        sed -i '/auth-wrapper\.js/d' "$file"
        sed -i '/msal-browser/d' "$file"
        sed -i '/cspAuthReady/d' "$file"
        sed -i '/cspAuthError/d' "$file"
    fi
    
    # Remove any existing MSAL references
    sed -i '/msal-browser/d' "$file"
    sed -i '/auth-wrapper/d' "$file"
    
    # Create temp file
    temp_file=$(mktemp)
    
    # Process the file
    while IFS= read -r line; do
        echo "$line" >> "$temp_file"
        
        # Add auth script before closing head tag
        if [[ "$line" == *"</head>"* ]]; then
            cat >> "$temp_file" << 'EOF'
    <!-- CSP Local Authentication Protection -->
    <script src="../js/auth-protection.js"></script>
    <style>
        /* Authentication header styles */
        #csp-auth-header {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        /* Page content adjustment for auth header */
        body.auth-protected {
            padding-top: 60px;
        }
        
        /* Loading indicator */
        .auth-loading {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 9999;
            background: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .auth-spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
EOF
        fi
        
        # Add ready handler before closing body tag
        if [[ "$line" == *"</body>"* ]]; then
            cat >> "$temp_file" << 'EOF'
    <!-- Authentication Event Handlers -->
    <script>
        // Show loading indicator while auth initializes
        document.addEventListener('DOMContentLoaded', () => {
            if (!window.location.pathname.includes('login.html')) {
                const loader = document.createElement('div');
                loader.className = 'auth-loading';
                loader.innerHTML = `
                    <div class="auth-spinner"></div>
                    <div>🔐 Initializing Authentication...</div>
                `;
                document.body.appendChild(loader);
                
                // Remove loader after auth is ready or timeout
                const removeLoader = () => {
                    if (loader.parentNode) {
                        loader.parentNode.removeChild(loader);
                    }
                };
                
                // Remove on auth ready
                document.addEventListener('cspAuthReady', removeLoader);
                document.addEventListener('cspAuthError', removeLoader);
                
                // Remove after timeout
                setTimeout(removeLoader, 10000);
            }
        });
        
        // Authentication ready event
        document.addEventListener('cspAuthReady', (event) => {
            console.log('🎉 Page authenticated and ready:', event.detail);
            document.body.classList.add('auth-protected');
            
            // Page-specific initialization can go here
            if (typeof onAuthReady === 'function') {
                onAuthReady(event.detail);
            }
        });
        
        // Authentication error event
        document.addEventListener('cspAuthError', (event) => {
            console.error('❌ Authentication error:', event.detail);
            
            // Page-specific error handling
            if (typeof onAuthError === 'function') {
                onAuthError(event.detail);
            }
        });
        
        // Session expired event
        document.addEventListener('cspSessionExpired', (event) => {
            console.warn('⏰ Session expired:', event.detail);
            
            // Page-specific session expiry handling
            if (typeof onSessionExpired === 'function') {
                onSessionExpired(event.detail);
            }
        });
    </script>
EOF
        fi
    done < "$file"
    
    # Replace original with modified version
    mv "$temp_file" "$file"
    echo "✅ Updated: $filename"
}

# Function to process directory
process_directory() {
    local dir="$1"
    local dir_name=$(basename "$dir")
    
    echo ""
    echo "📁 Processing directory: $dir_name"
    
    local count=0
    for file in "$dir"/*.html; do
        if [ -f "$file" ]; then
            add_auth_protection_to_page "$file"
            ((count++))
        fi
    done
    
    echo "   Processed $count files in $dir_name"
    return $count
}

# Main execution
total_files=0

# Process pages directory
if [ -d "$PAGES_DIR" ]; then
    process_directory "$PAGES_DIR"
    total_files=$((total_files + $?))
else
    echo "⚠️  Pages directory not found: $PAGES_DIR"
fi

# Process root frontend directory
if [ -d "$FRONTEND_DIR" ]; then
    echo ""
    echo "📁 Processing root frontend directory"
    count=0
    for file in "$FRONTEND_DIR"/*.html; do
        if [ -f "$file" ]; then
            add_auth_protection_to_page "$file"
            ((count++))
        fi
    done
    echo "   Processed $count files in frontend root"
    total_files=$((total_files + count))
fi

# Process enhanced_csp root directory
ENHANCED_CSP_DIR="$PROJECT_ROOT/enhanced_csp"
if [ -d "$ENHANCED_CSP_DIR" ]; then
    echo ""
    echo "📁 Processing enhanced_csp root directory"
    count=0
    for file in "$ENHANCED_CSP_DIR"/*.html; do
        if [ -f "$file" ]; then
            add_auth_protection_to_page "$file"
            ((count++))
        fi
    done
    echo "   Processed $count files in enhanced_csp root"
    total_files=$((total_files + count))
fi

echo ""
echo "🎉 Local Authentication Deployment Complete!"
echo "=============================================="
echo "📊 Total files processed: $total_files"
echo ""
echo "✅ What was added to each page:"
echo "   • auth-protection.js script"
echo "   • Authentication loading indicator"
echo "   • Event handlers for auth ready/error/expired"
echo "   • CSS styles for auth header and loading"
echo ""
echo "🔧 Next Steps:"
echo "1. Ensure auth-protection.js is in frontend/js/"
echo "2. Test login at: http://localhost:3000/pages/login.html"
echo "3. Verify protection on other pages"
echo "4. Test role-based access control"
echo ""
echo "🗂️  Backup files created with timestamp suffix"
echo "   You can restore originals if needed"
echo ""

# Create a verification script
cat > "$FRONTEND_DIR/verify-auth-deployment.sh" << 'EOF'
#!/bin/bash
# Quick verification of auth deployment

echo "🔍 Verifying Authentication Deployment"
echo "====================================="

PAGES_DIR="$(dirname "$0")/pages"
count=0
protected=0

for file in "$PAGES_DIR"/*.html; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        ((count++))
        
        if grep -q "auth-protection.js" "$file" && [ "$filename" != "login.html" ]; then
            echo "✅ $filename - protected"
            ((protected++))
        elif [ "$filename" == "login.html" ]; then
            echo "⏭️  $filename - login page (skipped)"
        else
            echo "❌ $filename - not protected"
        fi
    fi
done

echo ""
echo "📊 Summary: $protected/$count pages protected"

if [ $protected -eq $((count - 1)) ]; then
    echo "🎉 All pages successfully protected!"
else
    echo "⚠️  Some pages may need manual verification"
fi
EOF

chmod +x "$FRONTEND_DIR/verify-auth-deployment.sh"
echo "📝 Created verification script: frontend/verify-auth-deployment.sh"

echo ""
echo "🚀 Your Enhanced CSP System now has universal local authentication!"