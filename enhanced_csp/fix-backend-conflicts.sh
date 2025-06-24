#!/bin/bash
# fix-backend-conflicts.sh - Remove conflicting endpoint definitions

BACKEND_FILE="/home/mate/PAIN/csp-agent-network/csp-agent-network-1/enhanced_csp/backend/main.py"

echo "🔧 Fixing Backend Endpoint Conflicts"
echo "===================================="

# Create backup
cp "$BACKEND_FILE" "$BACKEND_FILE.backup.$(date +%Y%m%d_%H%M%S)"
echo "✅ Created backup of main.py"

# Remove the conflicting simple endpoints section
echo "🗑️  Removing conflicting simple auth endpoints..."

# Find and remove the section with duplicate endpoints
python3 << 'EOF'
import re

# Read the file
with open('/home/mate/PAIN/csp-agent-network/csp-agent-network-1/enhanced_csp/backend/main.py', 'r') as f:
    content = f.read()

# Remove the conflicting section that starts with "# Add to your FastAPI app"
# and contains the simple USERS_DB endpoints
conflict_start = content.find("# Add to your FastAPI app")
if conflict_start != -1:
    # Find the end of this section (look for the next major comment)
    conflict_end = content.find("# ============================================================================", conflict_start)
    if conflict_end == -1:
        # If no next section found, look for the end of local auth endpoints
        conflict_end = content.find("# Test credentials for immediate use:", conflict_start)
        if conflict_end != -1:
            # Include the test credentials comment
            conflict_end = content.find("\n# ", conflict_end + 1)
    
    if conflict_end != -1:
        # Remove the conflicting section
        new_content = content[:conflict_start] + content[conflict_end:]
        
        # Write back
        with open('/home/mate/PAIN/csp-agent-network/csp-agent-network-1/enhanced_csp/backend/main.py', 'w') as f:
            f.write(new_content)
        
        print("✅ Removed conflicting endpoints section")
    else:
        print("⚠️  Could not find end of conflicting section")
else:
    print("ℹ️  No conflicting section found")

EOF

echo ""
echo "🧹 Cleaned up backend conflicts"
echo ""
echo "🔧 Next steps:"
echo "1. Restart your backend: python3 -m backend.main"
echo "2. Test the endpoints again"
echo ""
echo "✅ The backend should now use only the proper LocalAuthService endpoints"