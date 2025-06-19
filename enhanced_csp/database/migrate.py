#!/usr/bin/env python3
"""
Database Migration Module for CSP System
=========================================

This module handles database initialization and migrations for the CSP system.
It creates the necessary database structure and handles both PostgreSQL and SQLite.
"""

import os
import sys
import time
import sqlite3
import asyncio
from pathlib import Path
from typing import Dict, Any

try:
    import psycopg2
    from psycopg2 import sql
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ PostgreSQL driver not available, using SQLite only")


class DatabaseMigrator:
    """Database migration and setup class"""
    
    def __init__(self):
        self.data_dir = Path.cwd() / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Database connection settings
        self.postgres_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'enhanced_csp',
            'user': 'csp_user',
            'password': 'csp_password'
        }
        
        self.sqlite_path = self.data_dir / "csp_system.db"
    
    def run_migrations(self):
        """Run all database migrations"""
        print("🗄️ Starting database migrations...")
        
        # Try PostgreSQL first, fallback to SQLite
        if POSTGRES_AVAILABLE:
            if self._setup_postgresql():
                print("✅ PostgreSQL migration completed successfully")
                return True
        
        # Fallback to SQLite
        self._setup_sqlite()
        print("✅ SQLite migration completed successfully")
        return True
    
    def _setup_postgresql(self) -> bool:
        """Setup PostgreSQL database"""
        try:
            print("🐘 Setting up PostgreSQL database...")
            
            # Wait for PostgreSQL to be ready
            max_retries = 30
            for attempt in range(max_retries):
                try:
                    conn = psycopg2.connect(**self.postgres_config)
                    conn.close()
                    break
                except psycopg2.OperationalError:
                    if attempt < max_retries - 1:
                        print(f"⏳ Waiting for PostgreSQL... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2)
                    else:
                        print("❌ PostgreSQL not ready, falling back to SQLite")
                        return False
            
            # Create tables
            conn = psycopg2.connect(**self.postgres_config)
            cursor = conn.cursor()
            
            # Create enhanced_csp schema
            cursor.execute("CREATE SCHEMA IF NOT EXISTS enhanced_csp;")
            cursor.execute("SET search_path TO enhanced_csp, public;")
            
            # Engines table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS engines (
                    engine_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL DEFAULT 'initializing',
                    consciousness_enabled BOOLEAN DEFAULT true,
                    quantum_enabled BOOLEAN DEFAULT false,
                    neural_mesh_enabled BOOLEAN DEFAULT false,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Agents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id VARCHAR(255) PRIMARY KEY,
                    engine_id VARCHAR(255) REFERENCES engines(engine_id),
                    agent_type VARCHAR(100),
                    status VARCHAR(50) DEFAULT 'inactive',
                    capabilities TEXT[],
                    consciousness_level DECIMAL(3,2) DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Processes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processes (
                    process_id VARCHAR(255) PRIMARY KEY,
                    engine_id VARCHAR(255) REFERENCES engines(engine_id),
                    process_type VARCHAR(100),
                    status VARCHAR(50) DEFAULT 'pending',
                    priority INTEGER DEFAULT 5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Channels table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS channels (
                    channel_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    channel_type VARCHAR(100),
                    status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(255),
                    metric_value DECIMAL(10,4),
                    labels JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Insert default engine if not exists
            cursor.execute("""
                INSERT INTO engines (engine_id, name, status, consciousness_enabled, quantum_enabled)
                VALUES ('default_engine', 'Default CSP Engine', 'running', true, false)
                ON CONFLICT (engine_id) DO NOTHING;
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print("✅ PostgreSQL tables created successfully")
            return True
            
        except Exception as e:
            print(f"❌ PostgreSQL setup failed: {e}")
            return False
    
    def _setup_sqlite(self):
        """Setup SQLite database as fallback"""
        print("💾 Setting up SQLite database...")
        
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Engines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS engines (
                engine_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'initializing',
                consciousness_enabled BOOLEAN DEFAULT 1,
                quantum_enabled BOOLEAN DEFAULT 0,
                neural_mesh_enabled BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Agents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                engine_id TEXT REFERENCES engines(engine_id),
                agent_type TEXT,
                status TEXT DEFAULT 'inactive',
                capabilities TEXT,
                consciousness_level REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Processes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processes (
                process_id TEXT PRIMARY KEY,
                engine_id TEXT REFERENCES engines(engine_id),
                process_type TEXT,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Channels table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS channels (
                channel_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                channel_type TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                labels TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Insert default engine if not exists
        cursor.execute("""
            INSERT OR IGNORE INTO engines (engine_id, name, status, consciousness_enabled, quantum_enabled)
            VALUES ('default_engine', 'Default CSP Engine', 'running', 1, 0);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ SQLite database created at: {self.sqlite_path}")
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database health and return status"""
        status = {
            'postgresql': {'available': False, 'connected': False},
            'sqlite': {'available': True, 'connected': False, 'path': str(self.sqlite_path)}
        }
        
        # Check PostgreSQL
        if POSTGRES_AVAILABLE:
            status['postgresql']['available'] = True
            try:
                conn = psycopg2.connect(**self.postgres_config)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM enhanced_csp.engines;")
                engine_count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                status['postgresql']['connected'] = True
                status['postgresql']['engine_count'] = engine_count
            except Exception as e:
                status['postgresql']['error'] = str(e)
        
        # Check SQLite
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM engines;")
            engine_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            status['sqlite']['connected'] = True
            status['sqlite']['engine_count'] = engine_count
        except Exception as e:
            status['sqlite']['error'] = str(e)
        
        return status


def main():
    """Main migration function"""
    print("🗄️ CSP System Database Migration")
    print("=" * 40)
    
    migrator = DatabaseMigrator()
    
    if migrator.run_migrations():
        print("\n✅ Database migration completed successfully!")
        
        # Show health check
        health = migrator.check_database_health()
        print("\n📊 Database Status:")
        for db_type, info in health.items():
            if info['connected']:
                engine_count = info.get('engine_count', 0)
                print(f"  ✅ {db_type.upper()}: Connected ({engine_count} engines)")
            elif info['available']:
                error = info.get('error', 'Connection failed')
                print(f"  ❌ {db_type.upper()}: {error}")
            else:
                print(f"  ⏭️  {db_type.upper()}: Not available")
        
        return 0
    else:
        print("\n❌ Database migration failed!")
        return 1


async def migrate_main() -> None:
    """Asynchronous entry point used by the main application."""
    migrator = DatabaseMigrator()
    await asyncio.to_thread(migrator.run_migrations)


if __name__ == "__main__":
    sys.exit(main())
