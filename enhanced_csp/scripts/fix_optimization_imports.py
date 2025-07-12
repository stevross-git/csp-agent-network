#!/usr/bin/env python3
"""
Fix import issues in optimization files.
Converts relative imports to work with current directory structure.
"""

import os
import re
from pathlib import Path


def fix_file_imports(file_path: Path) -> bool:
    """Fix imports in a single file."""
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False

    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content

        # Replace relative imports with absolute ones
        replacements = [
            # Core imports
            (r"from \.core\.config import", "from core.config import"),
            (r"from \.core\.types import", "from core.types import"),
            (r"from \.utils import", "from utils import"),
            # P2P imports
            (r"from \.p2p\.transport import", "from p2p.transport import"),
            (r"from \.p2p\.quic_transport import", "from p2p.quic_transport import"),
            # Other modules
            (r"from \.zero_copy import", "from zero_copy import"),
            (r"from \.batching import", "from batching import"),
            (r"from \.compression import", "from compression import"),
            (r"from \.connection_pool import", "from connection_pool import"),
            (r"from \.protocol_optimizer import", "from protocol_optimizer import"),
            (r"from \.adaptive_optimizer import", "from adaptive_optimizer import"),
            # Catch-all for remaining relative imports
            (r"from \.([a-zA-Z_][a-zA-Z0-9_]*) import", r"from \1 import"),
        ]

        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)

        # Insert sys.path modification if needed
        if ("from core." in content or "from p2p." in content) and "import sys" not in content:
            lines = content.split("\n")
            insert_pos = 0
            # Determine insertion point after shebang and docstring
            for idx, line in enumerate(lines):
                if line.startswith("#!"):
                    continue
                if line.startswith('"""') and '"""' in line[3:]:
                    insert_pos = idx + 1
                    break
                if line.startswith('"""'):
                    for j in range(idx + 1, len(lines)):
                        if '"""' in lines[j]:
                            insert_pos = j + 1
                            break
                    break
                if line.strip() == "":
                    continue
                insert_pos = idx
                break

            path_setup = [
                "",
                "import sys",
                "from pathlib import Path",
                "sys.path.insert(0, str(Path(__file__).parent))",
                "",
            ]
            lines = lines[:insert_pos] + path_setup + lines[insert_pos:]
            content = "\n".join(lines)

        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            print(f"✅ Fixed imports in: {file_path.name}")
            return True
        else:
            print(f"ℹ️  No changes needed: {file_path.name}")
            return False

    except Exception as exc:
        print(f"❌ Error fixing {file_path}: {exc}")
        return False


def main() -> None:
    """Fix imports in all optimization files."""
    print("🔧 Fixing import issues in optimization files...")

    current_dir = Path(__file__).parent
    files_to_fix = [
        "compression.py",
        "batching.py",
        "connection_pool.py",
        "protocol_optimizer.py",
        "zero_copy.py",
        "adaptive_optimizer.py",
        "optimized_channel.py",
        "p2p/quic_transport.py",
    ]

    fixed_count = 0
    for name in files_to_fix:
        if fix_file_imports(current_dir / name):
            fixed_count += 1

    print("\n📊 Summary:")
    print(f"  Files processed: {len(files_to_fix)}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Files skipped: {len(files_to_fix) - fixed_count}")

    if fixed_count:
        print("\n✅ Import fixes applied! You can now test the optimizations:")
        print("  python enhanced_csp/network/test_optimizations.py")
    else:
        print("\n💡 All files already have correct imports!")


if __name__ == "__main__":
    main()
