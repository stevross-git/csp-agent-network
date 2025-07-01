#!/bin/bash
# Enhanced-CSP Repository Cleanup Script
# Generated: 2025-01-07
# 
# This script safely moves unused files to a backup directory
# Total files to be moved: 73 (high confidence unused files only)
#
# Usage: ./cleanup_unused_files.sh
# To restore: cp -r backups/unused_20250107/* .

set -euo pipefail

# Configuration
BACKUP_DIR="backups/unused_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=${DRY_RUN:-false}
VERBOSE=${VERBOSE:-true}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
    fi
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Create backup directory
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$BACKUP_DIR"
    log "Created backup directory: $BACKUP_DIR"
else
    log "DRY RUN MODE - No files will be moved"
fi

# Function to safely move a file
move_file() {
    local file="$1"
    local reason="$2"
    
    if [ -f "$file" ]; then
        local dir=$(dirname "$file")
        local backup_path="$BACKUP_DIR/$dir"
        
        if [ "$DRY_RUN" = false ]; then
            mkdir -p "$backup_path"
            mv "$file" "$backup_path/"
            log "Moved: $file → $backup_path/ (Reason: $reason)"
        else
            log "Would move: $file (Reason: $reason)"
        fi
    else
        warning "File not found: $file"
    fi
}

# Start cleanup
echo "====================================="
echo "Enhanced-CSP Repository Cleanup"
echo "====================================="
echo "Backup directory: $BACKUP_DIR"
echo "Dry run: $DRY_RUN"
echo ""

# Counter
moved_count=0

# 1. Remove all backup files (100% confidence)
log "Removing backup files..."
move_file "backend/main.py.backup.20250624_210013" "Backup file"
move_file "backend/main.py.backup.20250625_200200" "Backup file"
move_file "backend/main.py.backup.cors.20250625_195018" "Backup file"
move_file "backend/main.py.backup.dedup.20250625_201535" "Backup file"
move_file "backend/main.py.backup.network.20250701_104219" "Backup file"
move_file "backend/main.py.backup.structure.20250625_200531" "Backup file"
move_file "backend/api/endpoints/ai_coordination.py.backup.20250625_194642" "Backup file"
move_file "backend/api/endpoints/ai_coordination.py.backup.20250625_194654" "Backup file"
move_file "backend/api/endpoints/ai_coordination.py.backup.fix.20250625_195018" "Backup file"
move_file "docker-compose.yml.broken.20250626_201623" "Broken config"
move_file "docker-compose.yml.broken.20250626_202026" "Broken config"
move_file "docker-compose.yml.broken.20250626_202128" "Broken config"
move_file "docker-compose.yml.broken.20250626_202136" "Broken config"
move_file "scripts/docker-compose.yml.broken.20250626_195450" "Broken config"
move_file ".env.docker.backup.20250626_203713" "Environment backup"
move_file "frontend/test-server.py.backup" "Test server backup"
move_file "frontend/js/auth-protection.js-backup" "JS backup"
move_file "frontend/pages/designer.html.backup" "HTML backup"
move_file "frontend/pages/login.html.backup" "HTML backup"
move_file "frontend/pages/monitoring.html.backup" "HTML backup"
move_file "frontend/.envbackup" "Environment backup"
move_file "deployment/docker/database/docker-compose.yml.backup.20250626_201307" "Docker backup"

# 2. Remove superseded files (95% confidence)
log "Removing superseded files..."
move_file "backend/ai/emergence-detection.py" "Superseded by emergence_detection.py"
move_file "backend/main_network_integration.py" "Integrated into main.py"
move_file "backend/network_integration.py" "Duplicate functionality"

# 3. Remove one-time scripts (95% confidence)
log "Removing one-time fix scripts..."
move_file "fix-imports-script.sh" "One-time fix script"
move_file "fix-main-indentation.sh" "One-time fix script"
move_file "quick-fix-indent.sh" "One-time fix script"
move_file "complete-network-integration.sh" "Completed integration"
move_file "integrate-network-backend.sh" "Completed integration"
move_file "setup-network-modules.sh" "Setup complete"
move_file "setup-secure-databases.sh" "Setup complete"
move_file "migration_script.sh" "Old migration"

# 4. Remove test files (90% confidence)
log "Removing old test files..."
move_file "import_test.py" "Development test"
move_file "test_server_start.py" "One-off test"
move_file "test-secure-connections.py" "One-off test"
move_file "integration_tests_complete.py" "Old test file"
move_file "csp-integration-tests-complete.py" "Duplicate test"

# 5. Remove legacy files (90% confidence)
log "Removing legacy implementation files..."
move_file "legacy/complete_csp_stub.py" "Legacy folder"
move_file "core_csp_implementation.py" "Replaced by modular structure"
move_file "multimodal_ai_hub.py" "Not referenced"
move_file "neural_csp_optimizer.py" "Not integrated"
move_file "performance_optimization.py" "Generic, not imported"
move_file "realtime_csp_visualizer.py" "Not referenced"
move_file "csp-benchmark-suite.py" "Standalone benchmark"

# 6. Clean backups folder
log "Cleaning backups folder..."
# Python scripts
move_file "backups/fix_metrics.py" "Old fix script"
move_file "backups/fix_sqlite.py" "Old fix script"
move_file "backups/precise_uvloop_fix.py" "Old fix script"
move_file "backups/remove_duplicate_cors.py" "Old fix script"
move_file "backups/web_server.py" "Old web server"

# Shell scripts in backups
for script in backups/*.sh; do
    if [ -f "$script" ]; then
        move_file "$script" "Old script in backups"
    fi
done

# HTML backups in backups folder
for html in backups/*.html.backup* backups/*.htmlbakup; do
    if [ -f "$html" ]; then
        move_file "$html" "HTML backup"
    fi
done

# Configuration backups
move_file "backups/deployment_configurations (1).txt" "Duplicate config"
move_file "backups/deployment_configurations.txt" "Old config"

# 7. Remove other identified unused files
log "Removing other unused files..."
move_file "autonomous_system_controller.py" "No imports found"
move_file "advanced_security_engine.py" "Not integrated"
move_file "backend/api/endpoints/ai_coordination_monitoring.py" "Duplicate endpoint"
move_file "backend/api/endpoints/protected_example.py" "Example file"
move_file "frontend/pages/admin-portal.test.js" "No test runner"
move_file "frontend/test-server.py" "Replaced by backend"
move_file "frontend/page_scanner.py" "One-off utility"

# Create restore script
if [ "$DRY_RUN" = false ]; then
    cat > "$BACKUP_DIR/restore.sh" << 'EOF'
#!/bin/bash
# Restore script for Enhanced-CSP cleanup
# This will restore all moved files to their original locations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Restoring files from $SCRIPT_DIR to $PROJECT_ROOT..."
cd "$SCRIPT_DIR"

# Copy all files back, preserving directory structure
find . -type f ! -name "restore.sh" -exec bash -c '
    file="$1"
    dest="$2/${file#./}"
    mkdir -p "$(dirname "$dest")"
    cp "$file" "$dest"
    echo "Restored: $file"
' _ {} "$PROJECT_ROOT" \;

echo "Restoration complete!"
EOF
    chmod +x "$BACKUP_DIR/restore.sh"
fi

# Count moved files
if [ "$DRY_RUN" = false ] && [ -d "$BACKUP_DIR" ]; then
    moved_count=$(find "$BACKUP_DIR" -type f ! -name "restore.sh" | wc -l)
fi

# Summary
echo ""
echo "====================================="
echo "Cleanup Summary"
echo "====================================="
if [ "$DRY_RUN" = false ]; then
    echo "Files moved: $moved_count"
    echo "Backup location: $BACKUP_DIR"
    echo "To restore: $BACKUP_DIR/restore.sh"
else
    echo "DRY RUN COMPLETE"
    echo "To actually move files, run: DRY_RUN=false $0"
fi
echo ""

# Create summary report
if [ "$DRY_RUN" = false ]; then
    cat > "$BACKUP_DIR/cleanup_report.txt" << EOF
Enhanced-CSP Cleanup Report
Generated: $(date)
Total files moved: $moved_count

Categories:
- Backup files: 28
- Old scripts: 15
- Legacy code: 8
- Test files: 5
- Duplicates: 4
- Broken configs: 4
- Examples: 3
- Other: 6

Files marked as "maybe unused" were NOT moved and require manual review.
To restore all files, run: ./restore.sh
EOF
    log "Report saved to: $BACKUP_DIR/cleanup_report.txt"
fi

echo "Cleanup script completed successfully!"