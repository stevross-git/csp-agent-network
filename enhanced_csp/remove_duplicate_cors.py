#!/usr/bin/env python3
"""
Remove duplicate CORS middleware configurations from main.py
"""

import os
import re
import ast
from datetime import datetime

def remove_duplicate_cors():
    """Remove duplicate CORS middleware configurations"""
    main_file = "backend/main.py"
    
    if not os.path.exists(main_file):
        print(f"❌ File {main_file} not found!")
        return False
    
    # Create backup
    backup_name = f"{main_file}.backup.dedup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        with open(main_file, 'r') as f:
            content = f.read()
        
        with open(backup_name, 'w') as f:
            f.write(content)
        print(f"✅ Created backup: {backup_name}")
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return False
    
    # Find all CORS middleware blocks
    lines = content.split('\n')
    cors_blocks = []
    current_block = None
    
    for i, line in enumerate(lines):
        if 'app.add_middleware(' in line and 'CORSMiddleware' in line:
            if current_block is not None:
                # Finish previous block
                cors_blocks.append(current_block)
            # Start new block
            current_block = {'start': i, 'lines': [line]}
        elif current_block is not None:
            current_block['lines'].append(line)
            # Check if this is the end of the CORS block
            if line.strip().endswith(')') and any(x in line for x in ['allow_', 'max_age', 'expose_']) and line.count('(') <= line.count(')'):
                current_block['end'] = i
                cors_blocks.append(current_block)
                current_block = None
    
    print(f"🔍 Found {len(cors_blocks)} CORS middleware blocks")
    
    if len(cors_blocks) <= 1:
        print("✅ No duplicate CORS configurations found")
        return True
    
    # Show the blocks found
    for i, block in enumerate(cors_blocks):
        print(f"\n📍 CORS Block {i+1} (lines {block['start']+1}-{block.get('end', '?')+1}):")
        for line in block['lines'][:3]:  # Show first few lines
            print(f"   {line}")
        if len(block['lines']) > 3:
            print(f"   ... ({len(block['lines'])} total lines)")
    
    # Remove duplicate blocks (keep only the first one)
    new_lines = []
    skip_ranges = []
    
    # Mark ranges to skip (all CORS blocks except the first one)
    for block in cors_blocks[1:]:  # Skip first block, remove the rest
        start = block['start']
        end = block.get('end', start + len(block['lines']) - 1)
        skip_ranges.append((start, end))
        print(f"🗑️ Will remove CORS block at lines {start+1}-{end+1}")
    
    # Rebuild content without duplicate CORS blocks
    for i, line in enumerate(lines):
        # Check if this line is in a skip range
        should_skip = False
        for start, end in skip_ranges:
            if start <= i <= end:
                should_skip = True
                break
        
        if not should_skip:
            new_lines.append(line)
    
    # Also ensure we don't have any early CORS before app creation
    final_lines = []
    app_created = False
    skip_early_cors = False
    
    for line in new_lines:
        # Track if app has been created
        if 'app = FastAPI(' in line:
            app_created = True
        
        # Skip CORS before app creation
        if 'app.add_middleware(' in line and 'CORSMiddleware' in line and not app_created:
            print("🗑️ Removing CORS configuration that appears before app creation")
            skip_early_cors = True
            continue
        
        if skip_early_cors:
            if line.strip().endswith(')') and any(x in line for x in ['allow_', 'max_age', 'expose_']):
                skip_early_cors = False
            continue
        
        final_lines.append(line)
    
    final_content = '\n'.join(final_lines)
    
    # Test the fix
    try:
        ast.parse(final_content)
        print("✅ Syntax validation passed")
        
        # Write the fixed content
        with open(main_file, 'w') as f:
            f.write(final_content)
        
        print(f"✅ Removed {len(cors_blocks) - 1} duplicate CORS configurations")
        return True
        
    except SyntaxError as e:
        print(f"❌ Still has syntax error: {e}")
        # Restore backup
        with open(backup_name, 'r') as f:
            original_content = f.read()
        with open(main_file, 'w') as f:
            f.write(original_content)
        print(f"🔄 Restored from backup")
        return False

def find_app_definition():
    """Find where the FastAPI app is defined"""
    main_file = "backend/main.py"
    
    with open(main_file, 'r') as f:
        lines = f.readlines()
    
    app_lines = []
    for i, line in enumerate(lines):
        if 'app = FastAPI(' in line:
            app_lines.append(i + 1)
    
    print(f"🔍 FastAPI app definitions found at lines: {app_lines}")
    return app_lines

def show_cors_locations():
    """Show where CORS configurations are located"""
    main_file = "backend/main.py"
    
    with open(main_file, 'r') as f:
        lines = f.readlines()
    
    cors_lines = []
    for i, line in enumerate(lines):
        if 'app.add_middleware(' in line and 'CORSMiddleware' in line:
            cors_lines.append(i + 1)
    
    print(f"🔍 CORS middleware configurations found at lines: {cors_lines}")
    
    # Show context around each CORS configuration
    for line_num in cors_lines:
        print(f"\n📍 CORS at line {line_num}:")
        start = max(0, line_num - 3)
        end = min(len(lines), line_num + 5)
        for i in range(start, end):
            marker = " >>> " if i == line_num - 1 else "     "
            print(f"{marker}{i+1:4d}: {lines[i].rstrip()}")

def test_import():
    """Test if the fixed file can be imported"""
    try:
        import sys
        sys.path.insert(0, '.')
        
        # Try to import
        import importlib
        if 'backend.main' in sys.modules:
            importlib.reload(sys.modules['backend.main'])
        else:
            import backend.main
        
        print("✅ main.py imports successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Analyzing CORS configurations in main.py...")
    
    print("\n1. Finding FastAPI app definitions:")
    find_app_definition()
    
    print("\n2. Finding CORS middleware locations:")
    show_cors_locations()
    
    print("\n3. Removing duplicate CORS configurations:")
    if remove_duplicate_cors():
        print("\n4. Testing import...")
        if test_import():
            print("\n🎉 Success! Duplicate CORS configurations removed!")
            print("🚀 Try running: python main.py")
        else:
            print("\n⚠️ CORS duplicates removed but import still has issues")
    else:
        print("\n❌ Failed to remove duplicates - manual intervention required")
        
    print("\n💡 If issues persist, you may need to:")
    print("   1. Check for any remaining duplicate middleware")
    print("   2. Ensure CORS comes after app = FastAPI(...)")
    print("   3. Look for any shell script remnants mixed in the code")