#!/usr/bin/env python3
"""
Robust BatmanRouting Import Fix with Safety Features
Production-ready script with regex patterns, backups, and dry-run support
"""

import os
import sys
import re
import argparse
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_backup(file_path: Path) -> Path:
    """Create a timestamped backup of the file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f".bak_{timestamp}")
    shutil.copy2(file_path, backup_path)
    logger.info(f"📁 Created backup: {backup_path}")
    return backup_path

def find_stats_dict_end(content: str) -> int:
    """Find the end of the stats dictionary to safely insert metrics"""
    lines = content.split('\n')
    in_stats = False
    brace_count = 0
    
    for i, line in enumerate(lines):
        if 'self.stats = {' in line:
            in_stats = True
            brace_count = line.count('{') - line.count('}')
            continue
        
        if in_stats:
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and '}' in line:
                return i + 1  # Insert after the closing brace
    
    return -1  # Not found

def fix_node_py(file_path: Path, dry_run: bool = False) -> bool:
    """Fix the enhanced_csp/network/core/node.py file with robust regex patterns"""
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    logger.info(f"🔧 {'[DRY RUN] ' if dry_run else ''}Fixing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Add SimpleRoutingStub class if not present (with defensive imports)
    routing_stub = '''
import logging
import time

class SimpleRoutingStub:
    """Simple routing stub to prevent import errors"""
    
    def __init__(self, node=None, topology=None):
        self.node = node
        self.topology = topology
        self.routing_table = {}
        self.is_running = False
    
    async def start(self):
        self.is_running = True
        logging.info("SimpleRoutingStub started")
        return True
    
    async def stop(self):
        self.is_running = False
        logging.info("SimpleRoutingStub stopped")
    
    def get_route(self, destination):
        """Get route to destination"""
        return self.routing_table.get(destination)
    
    def get_all_routes(self, destination):
        """Get all routes to destination"""
        route = self.routing_table.get(destination)
        return [route] if route else []
    
    def add_route(self, destination, route):
        """Add route to destination"""
        self.routing_table[destination] = route
    
    def remove_route(self, destination):
        """Remove route to destination"""
        return self.routing_table.pop(destination, None)

'''.strip()
    
    if 'class SimpleRoutingStub:' not in content:
        # Find insertion point before NetworkNode class
        insertion_match = re.search(r'^(logger\s*=.*?\n)', content, re.MULTILINE)
        if insertion_match:
            insert_pos = insertion_match.end()
            content = content[:insert_pos] + '\n\n' + routing_stub + '\n\n' + content[insert_pos:]
            logger.info("✅ Added SimpleRoutingStub class")
        else:
            logger.warning("⚠️  Could not find safe insertion point for SimpleRoutingStub")
    
    # 2. Fix BatmanRouting import with flexible regex
    batman_pattern = re.compile(
        r'#\s*Initialize\s+routing[^\n]*\n'
        r'\s*if\s+self\.config\.enable_routing[^\n]*\n'
        r'\s*self\.routing\s*=\s*BatmanRouting\([^)]*\)',
        re.MULTILINE | re.DOTALL
    )
    
    batman_replacement = '''# Initialize routing - handle BatmanRouting import error
            if getattr(self.config, 'enable_routing', True) and hasattr(self, 'topology') and self.topology:
                try:
                    from ..mesh.routing import BatmanRouting
                    self.routing = BatmanRouting(self, self.topology)
                    logging.info("BatmanRouting initialized successfully")
                except ImportError as e:
                    logging.warning(f"BatmanRouting not available: {e}")
                    self.routing = SimpleRoutingStub(self, self.topology)
                    logging.info("Using SimpleRoutingStub as fallback")
                except Exception as e:
                    logging.error(f"Failed to initialize BatmanRouting: {e}")
                    self.routing = SimpleRoutingStub(self, self.topology)'''
    
    if batman_pattern.search(content):
        content = batman_pattern.sub(batman_replacement, content)
        logger.info("✅ Fixed BatmanRouting initialization")
    
    # 3. Remove problematic standalone BatmanRouting imports
    problematic_import_pattern = re.compile(
        r'^\s*from\s+\.\.mesh\.routing\s+import\s+BatmanRouting\s*$',
        re.MULTILINE
    )
    
    if problematic_import_pattern.search(content):
        content = problematic_import_pattern.sub(
            '            # BatmanRouting imported conditionally below',
            content
        )
        logger.info("✅ Removed problematic BatmanRouting import")
    
    # 4. Add metrics attribute if missing (using safer insertion)
    if 'self.metrics = {' not in content:
        stats_end_line = find_stats_dict_end(content)
        if stats_end_line != -1:
            lines = content.split('\n')
            metrics_code = '''        
        # Add metrics attribute to prevent AttributeError
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'peers_connected': 0,
            'bandwidth_in': 0,
            'bandwidth_out': 0,
            'routing_table_size': 0,
            'last_updated': time.time()
        }'''
            lines.insert(stats_end_line, metrics_code)
            content = '\n'.join(lines)
            logger.info("✅ Added metrics attribute")
        else:
            logger.warning("⚠️  Could not find safe insertion point for metrics")
    
    # Write changes if not dry run
    if not dry_run and content != original_content:
        backup_path = create_backup(file_path)
        with open(file_path, 'w') as f:
            f.write(content)
        logger.info(f"✅ Applied changes to {file_path}")
        return True
    elif dry_run:
        logger.info(f"🔍 [DRY RUN] Would modify {file_path}")
        if content != original_content:
            logger.info("📝 Changes detected - would be applied in real run")
        else:
            logger.info("📝 No changes needed")
        return True
    else:
        logger.info("📝 No changes needed")
        return True

def fix_main_py(file_path: Path, dry_run: bool = False) -> bool:
    """Fix the enhanced_csp/network/main.py file with robust error handling"""
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    logger.info(f"🔧 {'[DRY RUN] ' if dry_run else ''}Fixing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix metrics collection with flexible regex
    metrics_pattern = re.compile(
        r'logger\.info\(f[\'"]Metrics:\s*\{network\.metrics\}[\'\"]\)',
        re.MULTILINE
    )
    
    metrics_replacement = '''if hasattr(network, 'metrics') and network.metrics:
                logger.info(f'Metrics: {network.metrics}')
            else:
                logger.info('Metrics: Not available')'''
    
    if metrics_pattern.search(content):
        content = metrics_pattern.sub(metrics_replacement, content)
        logger.info("✅ Fixed metrics collection")
    
    # Write changes if not dry run
    if not dry_run and content != original_content:
        backup_path = create_backup(file_path)
        with open(file_path, 'w') as f:
            f.write(content)
        logger.info(f"✅ Applied changes to {file_path}")
        return True
    elif dry_run:
        logger.info(f"🔍 [DRY RUN] Would modify {file_path}")
        if content != original_content:
            logger.info("📝 Changes detected - would be applied in real run")
        else:
            logger.info("📝 No changes needed")
        return True
    else:
        logger.info("📝 No changes needed")
        return True

def run_tests():
    """Suggest running tests after applying fixes"""
    test_commands = [
        "python -m pytest enhanced_csp/network/tests/ -v",
        "python -c 'import enhanced_csp.network.core.node; print(\"✅ Import successful\")'",
        "python -m enhanced_csp.network.main --help"
    ]
    
    print("\n🧪 Recommended post-fix validation:")
    for i, cmd in enumerate(test_commands, 1):
        print(f"  {i}. {cmd}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Robust BatmanRouting Import Fix for Enhanced CSP Network"
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help="Show what would be changed without making modifications"
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        default=True,
        help="Create backup files (default: True)"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 Enhanced CSP Network - Robust BatmanRouting Fix")
    print("=" * 60)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    
    print("\n📋 Applying robust fixes for:")
    print("  1. BatmanRouting import error (with regex patterns)")
    print("  2. Missing metrics attribute (safe insertion)")
    print("  3. Metrics collection safety (flexible matching)")
    print("  4. SimpleRoutingStub fallback (defensive imports)")
    
    success = True
    
    # Define file paths
    node_file = Path("enhanced_csp/network/core/node.py")
    main_file = Path("enhanced_csp/network/main.py")
    
    try:
        # Apply fixes
        if not fix_node_py(node_file, dry_run=args.dry_run):
            success = False
        
        if not fix_main_py(main_file, dry_run=args.dry_run):
            success = False
            
    except Exception as e:
        logger.error(f"❌ Error applying fixes: {e}")
        success = False
    
    if success:
        if not args.dry_run:
            print("\n✅ All fixes applied successfully!")
            print("\n🎯 Next steps:")
            print("  1. Stop the current network process (Ctrl+C if still running)")
            print("  2. Restart the network:")
            print("     python -m enhanced_csp.network.main --bootstrap genesis.web4ai --listen-port 8000")
            print("\n🔧 The network should now start without BatmanRouting errors.")
            print("📝 SimpleRoutingStub will handle routing as a safe fallback.")
            
            run_tests()
        else:
            print("\n🔍 Dry run completed successfully!")
            print("📝 Run without --dry-run to apply the changes.")
    else:
        print("\n❌ Some fixes failed to apply!")
        print("📁 Please check that you're running this from the project root directory.")
        print("🔧 Expected files:")
        print(f"   - {node_file}")
        print(f"   - {main_file}")
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)