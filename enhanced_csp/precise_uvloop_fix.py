#!/usr/bin/env python3
"""
Fix the syntax error in runtime/csp_runtime_environment.py
"""

def fix_syntax_error():
    """Fix the syntax error caused by the uvloop fix"""
    
    runtime_file = 'runtime/csp_runtime_environment.py'
    
    try:
        with open(runtime_file, 'r') as f:
            content = f.read()
        
        # Find the CSPRuntimeExecutor class and fix the __init__ method
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Look for the problematic section around the uvloop setup
            if 'class CSPRuntimeExecutor:' in line:
                # Copy class definition
                fixed_lines.append(line)
                i += 1
                
                # Copy until we reach __init__
                while i < len(lines) and 'def __init__(self, config: RuntimeConfig):' not in lines[i]:
                    fixed_lines.append(lines[i])
                    i += 1
                
                # Copy __init__ definition
                if i < len(lines):
                    fixed_lines.append(lines[i])  # def __init__ line
                    i += 1
                    
                    # Copy __init__ body, fixing the uvloop section
                    while i < len(lines) and not (lines[i].strip().startswith('def ') and not lines[i].strip().startswith('def __init__')):
                        current_line = lines[i]
                        
                        # Skip problematic try/except/if statements around uvloop
                        if ('if uvloop is not None:' in current_line or 
                            'uvloop.install()' in current_line or 
                            'logging.info("Using uvloop' in current_line or
                            'else:' in current_line and 'uvloop' in lines[i-1] or
                            'logging.warning("uvloop not available' in current_line):
                            # Skip these lines and replace with a simple safe version
                            if 'if uvloop is not None:' in current_line:
                                indent = len(current_line) - len(current_line.lstrip())
                                fixed_lines.append(' ' * indent + '# Setup event loop optimization')
                                fixed_lines.append(' ' * indent + 'if uvloop is not None:')
                                fixed_lines.append(' ' * (indent + 4) + 'try:')
                                fixed_lines.append(' ' * (indent + 8) + 'uvloop.install()')
                                fixed_lines.append(' ' * (indent + 8) + 'logging.info("Using uvloop for high performance")')
                                fixed_lines.append(' ' * (indent + 4) + 'except Exception as e:')
                                fixed_lines.append(' ' * (indent + 8) + 'logging.warning(f"Failed to install uvloop: {e}")')
                                fixed_lines.append(' ' * indent + 'else:')
                                fixed_lines.append(' ' * (indent + 4) + 'logging.warning("uvloop not available, using default event loop")')
                                
                                # Skip ahead past the problematic section
                                while i < len(lines) and ('uvloop' in lines[i] or 'logging.warning("uvloop' in lines[i] or (lines[i].strip() == 'else:' and i > 0 and 'uvloop' in lines[i-1])):
                                    i += 1
                                continue
                            else:
                                i += 1
                                continue
                        else:
                            fixed_lines.append(current_line)
                            i += 1
            else:
                fixed_lines.append(line)
                i += 1
        
        # Write back the fixed content
        with open(runtime_file, 'w') as f:
            f.write('\n'.join(fixed_lines))
        
        print("✅ Fixed syntax error in runtime/csp_runtime_environment.py")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix syntax error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing syntax error...")
    if fix_syntax_error():
        print("✅ Syntax error fixed!")
        print("🚀 Try running: python main.py")
    else:
        print("❌ Manual fix required")