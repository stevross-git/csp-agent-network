#!/bin/bash
# fix_syntax_error.sh - Fix syntax error in main.py

set -e

echo "🔧 Fixing Syntax Error in main.py"
echo "================================="

MAIN_PY="backend/main.py"

echo "📝 Checking syntax error around line 929..."

# Show lines around 929 to see the error
echo "Lines around 929:"
sed -n '925,935p' "$MAIN_PY" | cat -n

echo ""
echo "📝 Analyzing and fixing syntax errors..."

python3 << 'EOF'
import ast
import sys

# Read the file
with open('backend/main.py', 'r') as f:
    content = f.read()

print("🔍 Checking for syntax errors...")

# Try to parse the file to find syntax errors
try:
    ast.parse(content)
    print("✅ No syntax errors found")
except SyntaxError as e:
    print(f"❌ Syntax error found at line {e.lineno}: {e.msg}")
    print(f"Error text: {e.text}")
    
    # Read lines around the error
    lines = content.split('\n')
    error_line = e.lineno - 1  # Convert to 0-based index
    
    print(f"\nContext around line {e.lineno}:")
    for i in range(max(0, error_line - 3), min(len(lines), error_line + 4)):
        marker = " >>> " if i == error_line else "     "
        print(f"{marker}{i+1:4d}: {lines[i]}")
    
    # Common fixes for syntax errors
    if error_line < len(lines):
        line = lines[error_line]
        
        # Check for common issues
        fixes_applied = []
        
        # Fix missing commas in dictionaries/lists
        if '{' in line and '}' not in line and not line.strip().endswith(','):
            # Look for the next line to see if we need a comma
            if error_line + 1 < len(lines):
                next_line = lines[error_line + 1].strip()
                if next_line.startswith('"') or next_line.startswith("'") or next_line.endswith(':'):
                    if not line.strip().endswith(','):
                        lines[error_line] = line + ','
                        fixes_applied.append("Added missing comma")
        
        # Fix unmatched quotes
        if line.count('"') % 2 != 0 or line.count("'") % 2 != 0:
            # Try to fix unmatched quotes
            if line.count('"') % 2 != 0:
                lines[error_line] = line + '"'
                fixes_applied.append("Added missing quote")
        
        # Fix unmatched parentheses/brackets
        if line.count('(') != line.count(')'):
            if line.count('(') > line.count(')'):
                lines[error_line] = line + ')'
                fixes_applied.append("Added missing closing parenthesis")
        
        if line.count('[') != line.count(']'):
            if line.count('[') > line.count(']'):
                lines[error_line] = line + ']'
                fixes_applied.append("Added missing closing bracket")
        
        if line.count('{') != line.count('}'):
            if line.count('{') > line.count('}'):
                lines[error_line] = line + '}'
                fixes_applied.append("Added missing closing brace")
        
        # Check for invalid trailing content
        if line.strip().endswith('EOF'):
            # Remove EOF from Python code
            lines[error_line] = line.replace('EOF', '').rstrip()
            fixes_applied.append("Removed invalid EOF")
        
        if fixes_applied:
            print(f"\n🔧 Applied fixes: {', '.join(fixes_applied)}")
            
            # Write the fixed content
            fixed_content = '\n'.join(lines)
            with open('backend/main.py', 'w') as f:
                f.write(fixed_content)
            
            # Test the fix
            try:
                ast.parse(fixed_content)
                print("✅ Syntax error fixed!")
            except SyntaxError as e2:
                print(f"❌ Still has syntax error at line {e2.lineno}: {e2.msg}")
                # Restore original and try different approach
                with open('backend/main.py', 'w') as f:
                    f.write(content)
                
                # Try to fix by removing problematic lines
                print("🔧 Trying to remove problematic content...")
                lines = content.split('\n')
                
                # Remove anything that looks like shell script remnants
                filtered_lines = []
                for line in lines:
                    # Skip lines that look like shell script
                    if any(pattern in line for pattern in ['EOF', '#!/bin/bash', 'echo "', '${', 'set -e']):
                        print(f"🗑️ Removing shell script line: {line.strip()}")
                        continue
                    filtered_lines.append(line)
                
                fixed_content = '\n'.join(filtered_lines)
                with open('backend/main.py', 'w') as f:
                    f.write(fixed_content)
                
                # Test again
                try:
                    ast.parse(fixed_content)
                    print("✅ Syntax error fixed by removing shell script remnants!")
                except SyntaxError as e3:
                    print(f"❌ Still has syntax error: {e3.msg}")
                    print("Manual intervention required")
        else:
            print("❌ Could not automatically fix syntax error")
            print("Manual intervention required")

except Exception as e:
    print(f"❌ Error analyzing file: {e}")
EOF

echo ""
echo "🧪 Testing the fixed file..."

python3 << 'EOF'
import sys
import os
sys.path.insert(0, '.')

print("Testing fixed main.py...")

try:
    # Test syntax by parsing
    with open('backend/main.py', 'r') as f:
        content = f.read()
    
    import ast
    ast.parse(content)
    print("✅ Syntax is now valid")
    
    # Test import
    try:
        import backend.main
        print("✅ main.py imports successfully")
    except Exception as e:
        print(f"⚠️ Import issue (may be normal): {e}")
    
except SyntaxError as e:
    print(f"❌ Still has syntax error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Other error: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Syntax Error Fixed!"
    echo "====================="
    echo ""
    echo "✅ main.py syntax is now valid"
    echo ""
    echo "🚀 Try starting your server:"
    echo "   python -m backend.main"
    echo ""
else
    echo ""
    echo "❌ Syntax error still exists. Let's check the file manually:"
    echo ""
    echo "Showing end of file to check for issues:"
    tail -20 "$MAIN_PY"
    echo ""
    echo "You may need to manually edit backend/main.py to remove any shell script remnants"
fi