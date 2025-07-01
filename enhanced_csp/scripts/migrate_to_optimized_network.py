# scripts/migrate_to_optimized_network.py
"""
Migration script to update existing code to use optimized networking
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import Set, List, Tuple

try:
    import astor
except ImportError:
    print("Error: astor not installed. Please run: pip install astor")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkOptimizationMigrator(ast.NodeTransformer):
    """AST transformer to migrate code to optimized networking"""
    
    def __init__(self):
        self.imports_to_add = set()
        self.modified = False
        
    def visit_ImportFrom(self, node):
        """Update import statements"""
        # Update BaseCommChannel imports to AdvancedAICommChannel
        if node.module == 'enhanced_csp.ai_comm':
            for alias in node.names:
                if alias.name == 'BaseCommChannel':
                    # Replace with AdvancedAICommChannel
                    alias.name = 'AdvancedAICommChannel'
                    self.modified = True
                    logger.info("Replaced BaseCommChannel import with AdvancedAICommChannel")
                    
            # Ensure AdvancedAICommChannel is imported
            has_advanced = any(alias.name == 'AdvancedAICommChannel' for alias in node.names)
            if not has_advanced:
                node.names.append(ast.alias(name='AdvancedAICommChannel', asname=None))
                self.modified = True
                
            # Ensure AdvancedCommPattern is imported
            has_pattern = any(alias.name == 'AdvancedCommPattern' for alias in node.names)
            if not has_pattern:
                node.names.append(ast.alias(name='AdvancedCommPattern', asname=None))
                self.modified = True
                
        return node
        
    def visit_Call(self, node):
        """Update channel creation calls"""
        # Update BaseCommChannel() to AdvancedAICommChannel()
        if isinstance(node.func, ast.Name):
            if node.func.id == 'BaseCommChannel':
                node.func.id = 'AdvancedAICommChannel'
                self.modified = True
                
                # Add pattern argument if missing
                has_pattern = any(kw.arg == 'pattern' for kw in node.keywords)
                if not has_pattern:
                    # Default to BROADCAST pattern
                    pattern_arg = ast.keyword(
                        arg='pattern',
                        value=ast.Attribute(
                            value=ast.Name(id='AdvancedCommPattern', ctx=ast.Load()),
                            attr='BROADCAST',
                            ctx=ast.Load()
                        )
                    )
                    node.keywords.append(pattern_arg)
                    
            elif node.func.id == 'AdvancedAICommChannel':
                # Check if config argument exists
                has_config = any(kw.arg == 'config' for kw in node.keywords)
                
                if not has_config:
                    # Add network optimization config
                    config_dict = ast.Dict(
                        keys=[ast.Str(s='network')],
                        values=[ast.Dict(
                            keys=[ast.Str(s='optimization_enabled')],
                            values=[ast.NameConstant(value=True)]
                        )]
                    )
                    node.keywords.append(
                        ast.keyword(arg='config', value=config_dict)
                    )
                    self.modified = True
                    
        return self.generic_visit(node)

def find_python_files(root_path: Path) -> List[Path]:
    """Find all Python files in the project"""
    python_files = []
    
    # Directories to search
    search_dirs = ['enhanced_csp', 'examples', 'tests']
    
    for dir_name in search_dirs:
        dir_path = root_path / dir_name
        if dir_path.exists():
            python_files.extend(dir_path.glob("**/*.py"))
            
    # Exclude network module itself and migration script
    excluded_patterns = ['network/', 'migrate_to_optimized_network.py', '__pycache__']
    
    filtered_files = []
    for file_path in python_files:
        if not any(pattern in str(file_path) for pattern in excluded_patterns):
            filtered_files.append(file_path)
            
    return filtered_files

def migrate_file(filepath: Path) -> Tuple[bool, str]:
    """Migrate a single Python file"""
    logger.info(f"Checking {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Skip if file is empty
        if not content.strip():
            return False, "Empty file"
            
        tree = ast.parse(content)
        
        # Apply transformations
        migrator = NetworkOptimizationMigrator()
        new_tree = migrator.visit(tree)
        
        if migrator.modified:
            # Generate new code
            new_content = astor.to_source(new_tree)
            
            # Backup original
            backup_path = filepath.with_suffix(filepath.suffix + '.bak')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Write updated code
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            logger.info(f"✅ Migrated {filepath}")
            return True, "Success"
        else:
            return False, "No changes needed"
            
    except SyntaxError as e:
        logger.error(f"Syntax error in {filepath}: {e}")
        return False, f"Syntax error: {e}"
    except Exception as e:
        logger.error(f"Failed to migrate {filepath}: {e}")
        return False, f"Error: {e}"

def create_config_template():
    """Create a template configuration file for network optimization"""
    config_template = """# Network Optimization Configuration
network_optimization:
  enabled: true
  
  compression:
    default_algorithm: "lz4"
    min_size_bytes: 256
    max_decompress_mb: 100
    
  batching:
    max_size: 100
    max_bytes: 1048576
    max_wait_ms: 50
    queue_size: 10000
    
  connection_pool:
    min: 10
    max: 100
    keepalive_timeout: 300
    http2: true
    
  adaptive:
    enabled: true
    interval_seconds: 10
"""
    
    config_path = Path("config/network_optimization.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_template)
        logger.info(f"Created configuration template at {config_path}")

def main():
    """Run migration on all Python files"""
    project_root = Path(__file__).parent.parent
    
    print("🚀 Network Optimization Migration Tool")
    print("=" * 50)
    
    # Find all Python files
    python_files = find_python_files(project_root)
    logger.info(f"Found {len(python_files)} Python files to check")
    
    # Track results
    migrated = 0
    skipped = 0
    failed = 0
    
    for filepath in python_files:
        success, message = migrate_file(filepath)
        if success:
            migrated += 1
        elif "No changes needed" in message:
            skipped += 1
        else:
            failed += 1
            
    # Create config template
    create_config_template()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Migration Summary:")
    print(f"   Files migrated: {migrated}")
    print(f"   Files skipped:  {skipped}")
    print(f"   Files failed:   {failed}")
    print(f"   Total checked:  {len(python_files)}")
    
    if migrated > 0:
        print("\n✅ Migration successful!")
        print("\n📋 Next steps:")
        print("1. Review the changes in migrated files")
        print("2. Update config/network_optimization.yaml with your settings")
        print("3. Run tests to ensure everything works:")
        print("   pytest tests/")
        print("4. Remove backup files (*.py.bak) after verification")
    else:
        print("\n✅ No migration needed - your code is already up to date!")
        
    # Check requirements
    print("\n📦 Don't forget to update requirements:")
    print("   pip install astor  # For migration script")
    print("   pip install -r requirements.txt  # For network optimization")

if __name__ == "__main__":
    main()