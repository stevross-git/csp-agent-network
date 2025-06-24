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
