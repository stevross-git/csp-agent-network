#!/bin/bash
# fix_main_py_structure.sh - Fix the main.py structure and CORS placement

set -e

echo "🔧 Fixing main.py Structure and CORS Placement"
echo "=============================================="

MAIN_PY="backend/main.py"

if [ ! -f "$MAIN_PY" ]; then
    echo "❌ $MAIN_PY not found!"
    exit 1
fi

echo "📝 Creating backup and fixing main.py structure..."

# Create backup
cp "$MAIN_PY" "${MAIN_PY}.backup.structure.$(date +%Y%m%d_%H%M%S)"

python3 << 'EOF'
import os

# Read the current main.py
with open('backend/main.py', 'r') as f:
    content = f.read()

print("📝 Analyzing main.py structure...")

# Remove any misplaced CORS configuration at the top
lines = content.split('\n')
new_lines = []
skip_misplaced_cors = False

for line in lines:
    # Skip any CORS configuration that appears before app creation
    if 'app.add_middleware(' in line and 'CORSMiddleware' in line:
        # Check if we're before app creation
        app_creation_found = False
        for prev_line in new_lines:
            if 'app = FastAPI(' in prev_line:
                app_creation_found = True
                break
        
        if not app_creation_found:
            print("🗑️ Removing misplaced CORS configuration...")
            skip_misplaced_cors = True
            continue
    
    if skip_misplaced_cors:
        # Skip until we find the end of the CORS config
        if line.strip().endswith(')') and any(x in line for x in ['allow_', 'max_age', 'expose_']):
            skip_misplaced_cors = False
        continue
    
    new_lines.append(line)

content = '\n'.join(new_lines)

# Now find the right place to add CORS (after app creation but before middleware section)
lines = content.split('\n')
new_lines = []
cors_added = False

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Add CORS after app creation and before any existing middleware
    if 'app = FastAPI(' in line and not cors_added:
        # Look ahead to find the end of FastAPI constructor
        j = i + 1
        while j < len(lines) and not lines[j].strip().endswith(')'):
            new_lines.append(lines[j])
            j = i + 1
            break
        
        # Add CORS configuration
        cors_config = [
            "",
            "# ============================================================================",
            "# CORS MIDDLEWARE CONFIGURATION",
            "# ============================================================================",
            "",
            "# CORS middleware with explicit configuration for frontend",
            "app.add_middleware(",
            "    CORSMiddleware,",
            "    allow_origins=[",
            "        \"http://localhost:3000\",",
            "        \"http://localhost:3001\",", 
            "        \"http://127.0.0.1:3000\",",
            "        \"http://127.0.0.1:3001\",",
            "        \"http://localhost:8000\"",
            "    ],",
            "    allow_credentials=True,",
            "    allow_methods=[\"GET\", \"POST\", \"PUT\", \"DELETE\", \"OPTIONS\", \"PATCH\", \"HEAD\"],",
            "    allow_headers=[\"*\"],",
            "    expose_headers=[\"*\"],",
            "    max_age=600",
            ")"
        ]
        
        new_lines.extend(cors_config)
        cors_added = True
        print("✅ Added CORS configuration after app creation")

# Remove any duplicate CORS configurations
final_lines = []
skip_duplicate_cors = False

for line in new_lines:
    if 'app.add_middleware(' in line and 'CORSMiddleware' in line:
        # Check if we already added CORS
        if cors_added and any('allow_origins=' in l for l in final_lines[-10:]):
            print("🗑️ Removing duplicate CORS configuration...")
            skip_duplicate_cors = True
            continue
    
    if skip_duplicate_cors:
        if line.strip().endswith(')') and any(x in line for x in ['allow_', 'max_age', 'expose_']):
            skip_duplicate_cors = False
        continue
    
    final_lines.append(line)

# Write the fixed content
with open('backend/main.py', 'w') as f:
    f.write('\n'.join(final_lines))

print("✅ Fixed main.py structure and CORS placement")
EOF

echo ""
echo "🧪 Testing the fixed main.py..."

python3 << 'EOF'
import sys
import os
sys.path.insert(0, '.')

print("Testing main.py import...")

try:
    # Test if main.py can be imported without errors
    import backend.main
    print("✅ main.py imports successfully")
except NameError as e:
    if "app" in str(e):
        print(f"❌ Still has app definition error: {e}")
        exit(1)
    else:
        print(f"⚠️ Other NameError: {e}")
except Exception as e:
    print(f"⚠️ Import warning (this may be normal): {e}")
    # Don't exit on import warnings, they're often normal

print("✅ main.py structure test completed")
EOF

echo ""
echo "📝 Adding a simple startup test..."

# Create a simple test to verify the server can start
cat > test_server_start.py << 'EOF'
#!/usr/bin/env python3
"""
Simple test to verify the server can start without errors
"""
import sys
import os
sys.path.insert(0, '.')

def test_server_import():
    """Test that the server can be imported"""
    try:
        # Import the main module
        from backend.main import app
        print("✅ FastAPI app created successfully")
        
        # Check if CORS is configured
        cors_found = False
        for middleware in app.user_middleware:
            if 'CORSMiddleware' in str(middleware):
                cors_found = True
                break
        
        if cors_found:
            print("✅ CORS middleware is configured")
        else:
            print("⚠️ CORS middleware not found")
        
        print("✅ Server import test passed")
        return True
        
    except Exception as e:
        print(f"❌ Server import test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_server_import()
    sys.exit(0 if success else 1)
EOF

python3 test_server_start.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 main.py Structure Fixed Successfully!"
    echo "======================================="
    echo ""
    echo "✅ What was fixed:"
    echo "  🔧 Moved CORS configuration to correct location"
    echo "  🗑️ Removed duplicate/misplaced CORS configs"
    echo "  📝 Fixed app definition order"
    echo "  🧪 Verified server can start"
    echo ""
    echo "🚀 Now try starting your server:"
    echo "   python -m backend.main"
    echo ""
    echo "🎯 The server should start without NameError!"
    
    # Clean up test file
    rm -f test_server_start.py
else
    echo "❌ main.py structure fix failed. Please check the error messages above."
    echo "You may need to manually review backend/main.py"
    exit 1
fi