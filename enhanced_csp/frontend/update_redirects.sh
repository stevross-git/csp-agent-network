#!/bin/bash

# Enhanced CSP System - Update Authentication Redirects
# Updates all authentication redirects to point to index.html instead of dashboard.html

echo "🔧 Enhanced CSP System - Updating Authentication Redirects"
echo "==========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FRONTEND_DIR="$(pwd)"
if [[ ! "$FRONTEND_DIR" == *"frontend"* ]]; then
    echo -e "${RED}❌ Please run this script from the frontend directory${NC}"
    exit 1
fi

echo -e "${BLUE}📁 Frontend directory: $FRONTEND_DIR${NC}"

# Function to update redirects in a file
update_redirects_in_file() {
    local file="$1"
    local filename=$(basename "$file")
    local changes=0
    
    if [ ! -f "$file" ]; then
        return 0
    fi
    
    echo -e "${YELLOW}🔍 Checking: $filename${NC}"
    
    # Create temporary file
    local temp_file=$(mktemp)
    
    # Update dashboard.html redirects to index.html
    if sed 's/dashboard\.html/index.html/g' "$file" > "$temp_file"; then
        if ! diff -q "$file" "$temp_file" > /dev/null; then
            mv "$temp_file" "$file"
            ((changes++))
            echo -e "${GREEN}   ✅ Updated dashboard.html → index.html${NC}"
        else
            rm "$temp_file"
        fi
    else
        rm "$temp_file"
        echo -e "${RED}   ❌ Error processing file${NC}"
        return 1
    fi
    
    # Update specific redirect patterns
    temp_file=$(mktemp)
    
    # Update "Go to Dashboard" to "Go to Main Page"
    if sed 's/Go to Dashboard/Go to Main Page/g' "$file" > "$temp_file"; then
        if ! diff -q "$file" "$temp_file" > /dev/null; then
            mv "$temp_file" "$file"
            ((changes++))
            echo -e "${GREEN}   ✅ Updated button text${NC}"
        else
            rm "$temp_file"
        fi
    else
        rm "$temp_file"
    fi
    
    # Update dashboard icon to home icon
    temp_file=$(mktemp)
    if sed 's/📊 Go to Dashboard/🏠 Go to Main Page/g' "$file" > "$temp_file"; then
        if ! diff -q "$file" "$temp_file" > /dev/null; then
            mv "$temp_file" "$file"
            ((changes++))
            echo -e "${GREEN}   ✅ Updated button icon and text${NC}"
        else
            rm "$temp_file"
        fi
    else
        rm "$temp_file"
    fi
    
    # Update "redirecting to dashboard" messages
    temp_file=$(mktemp)
    if sed 's/Redirecting to dashboard/Redirecting to main page/g' "$file" > "$temp_file"; then
        if ! diff -q "$file" "$temp_file" > /dev/null; then
            mv "$temp_file" "$file"
            ((changes++))
            echo -e "${GREEN}   ✅ Updated status messages${NC}"
        else
            rm "$temp_file"
        fi
    else
        rm "$temp_file"
    fi
    
    # Update path references in authentication wrappers
    temp_file=$(mktemp)
    if sed 's|/enhanced_csp/frontend/pages/dashboard\.html|/pages/index.html|g' "$file" > "$temp_file"; then
        if ! diff -q "$file" "$temp_file" > /dev/null; then
            mv "$temp_file" "$file"
            ((changes++))
            echo -e "${GREEN}   ✅ Updated absolute path references${NC}"
        else
            rm "$temp_file"
        fi
    else
        rm "$temp_file"
    fi
    
    # Update relative path references
    temp_file=$(mktemp)
    if sed 's|../pages/dashboard\.html|index.html|g' "$file" > "$temp_file"; then
        if ! diff -q "$file" "$temp_file" > /dev/null; then
            mv "$temp_file" "$file"
            ((changes++))
            echo -e "${GREEN}   ✅ Updated relative path references${NC}"
        else
            rm "$temp_file"
        fi
    else
        rm "$temp_file"
    fi
    
    if [ $changes -eq 0 ]; then
        echo -e "${BLUE}   ℹ️  No changes needed${NC}"
    else
        echo -e "${GREEN}   🎉 Applied $changes changes${NC}"
    fi
    
    return $changes
}

# Main processing
echo ""
echo -e "${YELLOW}🔄 Processing authentication files...${NC}"
echo ""

total_changes=0
files_processed=0

# Process HTML files
for file in pages/*.html js/*.js css/*.css config/*.js services/*.js middleware/*.js; do
    if [ -f "$file" ]; then
        update_redirects_in_file "$file"
        changes=$?
        if [ $changes -gt 0 ]; then
            ((total_changes+=changes))
        fi
        ((files_processed++))
    fi
done

echo ""
echo -e "${BLUE}📊 Summary:${NC}"
echo -e "${GREEN}   ✅ Files processed: $files_processed${NC}"
echo -e "${GREEN}   ✅ Total changes: $total_changes${NC}"

if [ $total_changes -gt 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 Successfully updated authentication redirects!${NC}"
    echo ""
    echo -e "${BLUE}What was changed:${NC}"
    echo "   • dashboard.html → index.html"
    echo "   • 'Go to Dashboard' → 'Go to Main Page'"
    echo "   • '📊 Go to Dashboard' → '🏠 Go to Main Page'"
    echo "   • Status messages updated"
    echo "   • Path references corrected"
    echo ""
    echo -e "${YELLOW}⚠️  Recommendations:${NC}"
    echo "   1. Test the login flow to ensure it redirects to index.html"
    echo "   2. Verify all authentication wrappers point to correct paths"
    echo "   3. Check that protected pages redirect properly"
    echo "   4. Clear browser cache if needed"
else
    echo ""
    echo -e "${BLUE}ℹ️  All files are already up to date!${NC}"
fi

echo ""
echo -e "${GREEN}✅ Redirect update complete!${NC}"

# Verify critical files
echo ""
echo -e "${YELLOW}🔍 Verifying critical files...${NC}"

critical_files=("pages/login.html" "js/auth-wrapper.js" "js/auth_middleware.js")
for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        if grep -q "index\.html" "$file"; then
            echo -e "${GREEN}   ✅ $file: Contains index.html references${NC}"
        else
            echo -e "${YELLOW}   ⚠️  $file: No index.html references found${NC}"
        fi
        
        if grep -q "dashboard\.html" "$file"; then
            echo -e "${YELLOW}   ⚠️  $file: Still contains dashboard.html references${NC}"
        fi
    else
        echo -e "${RED}   ❌ $file: File not found${NC}"
    fi
done

echo ""
echo -e "${BLUE}🚀 Ready to test! Try logging in to verify the redirect works.${NC}"