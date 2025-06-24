#!/bin/bash
# enhanced_deploy_auth.sh

PROJECT_ROOT="/home/mate/PAIN/csp-agent-network/csp-agent-network-1"
FRONTEND_DIR="$PROJECT_ROOT/enhanced_csp/frontend"
PAGES_DIR="$FRONTEND_DIR/pages"

echo "🔐 Deploying Unified Authentication to All Pages..."

# Function to add auth to a single page
add_auth_to_page() {
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
    
    # Check if already has auth
    if grep -q "auth-wrapper.js" "$file"; then
        echo "⚠️  $filename already has auth - updating..."
    fi
    
    # Create temp file
    temp_file=$(mktemp)
    
    # Process the file
    while IFS= read -r line; do
        echo "$line" >> "$temp_file"
        
        # Add auth scripts before closing head tag
        if [[ "$line" == *"</head>"* ]]; then
            cat >> "$temp_file" << 'EOF'
    <!-- CSP Unified Authentication -->
    <script src="https://alcdn.msauth.net/browser/2.38.3/js/msal-browser.min.js"></script>
    <script src="../js/auth-wrapper.js"></script>
    <link rel="stylesheet" href="../css/auth-wrapper.css">
EOF
        fi
        
        # Add ready handler before closing body tag
        if [[ "$line" == *"</body>"* ]]; then
            cat >> "$temp_file" << 'EOF'
    <!-- Authentication Ready Handler -->
    <script>
        document.addEventListener('cspAuthReady', (event) => {
            console.log('🎉 Page authenticated and ready:', event.detail);
            // Page-specific initialization can go here
        });
        
        document.addEventListener('cspAuthError', (event) => {
            console.error('❌ Authentication error:', event.detail);
            // Handle auth errors
        });
    </script>
EOF
        fi
    done < "$file"
    
    # Replace original with modified version
    mv "$temp_file" "$file"
    echo "✅ Updated: $filename"
}

# Process all HTML files
count=0
for file in "$PAGES_DIR"/*.html "$PROJECT_ROOT/enhanced_csp"/*.html; do
    if [ -f "$file" ]; then
        add_auth_to_page "$file"
        ((count++))
    fi
done

echo ""
echo "🎉 Authentication deployment complete!"
echo "📊 Processed $count HTML files"
echo ""
echo "Next steps:"
echo "1. Test login at: http://localhost:3000/pages/login.html"
echo "2. Verify protection on other pages"
echo "3. Test role-based access control"