#!/bin/bash
# Quick fix for indentation error in main.py

echo "🔧 Fixing indentation error at line 846..."

# First, let's see what's around line 846
echo "📝 Context around line 846:"
sed -n '840,850p' backend/main.py | cat -n

# The issue is likely that the network shutdown code has incorrect indentation
# Let's fix it by finding the pattern and correcting the indentation

# Create a temporary Python script to fix the indentation
cat > fix_indent.py << 'EOF'
import re

with open('backend/main.py', 'r') as f:
    content = f.read()

# Fix the shutdown section - the network shutdown should be inside the finally block
# Find the finally block in the lifespan function and fix indentation

# Split into lines for easier processing
lines = content.split('\n')
fixed_lines = []
in_lifespan = False
in_finally = False
finally_indent_level = 0

for i, line in enumerate(lines):
    # Track if we're in the lifespan function
    if 'async def lifespan(app: FastAPI):' in line:
        in_lifespan = True
    
    # Track if we're in the finally block of lifespan
    if in_lifespan and line.strip() == 'finally:':
        in_finally = True
        finally_indent_level = len(line) - len(line.lstrip())
    
    # If we find the network shutdown code with wrong indentation, fix it
    if in_finally and 'if NETWORK_AVAILABLE and network_service:' in line:
        # Should be indented 4 spaces from the finally block
        fixed_line = ' ' * (finally_indent_level + 4) + line.strip()
        fixed_lines.append(fixed_line)
        continue
    
    if in_finally and 'await shutdown_network_service()' in line:
        # Should be indented 8 spaces from the finally block
        fixed_line = ' ' * (finally_indent_level + 8) + line.strip()
        fixed_lines.append(fixed_line)
        continue
    
    # End of lifespan function
    if in_lifespan and line.strip() and not line[0].isspace() and 'def' in line:
        in_lifespan = False
        in_finally = False
    
    fixed_lines.append(line)

# Join and write back
with open('backend/main.py', 'w') as f:
    f.write('\n'.join(fixed_lines))

print("Fixed indentation issues")
EOF

python3 fix_indent.py
rm fix_indent.py

echo ""
echo "✅ Indentation should be fixed now!"
echo ""
echo "Try running again:"
echo "  python -m backend.main"
