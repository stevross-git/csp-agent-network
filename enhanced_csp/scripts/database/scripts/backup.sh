#!/bin/bash
# Database backup script for Enhanced CSP Project
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
echo "Starting database backups at $(date)"
mkdir -p "$BACKUP_DIR"

# Backup main PostgreSQL database
echo "Backing up main PostgreSQL database..."
pg_dump -h postgres -U csp_user -d csp_visual_designer > "$BACKUP_DIR/main_db_$DATE.sql"
if [ $? -eq 0 ]; then
    echo "✅ Main database backup completed"
    gzip "$BACKUP_DIR/main_db_$DATE.sql"
else
    echo "❌ Main database backup failed"
fi

echo "Backup process completed at $(date)"
