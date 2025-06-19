import sqlite3
from pathlib import Path
import os

# Find the SQLite database
possible_paths = [
    "data/csp_system.db",
    "csp-system/data/csp_system.db", 
    os.path.expanduser("~/.csp/data/csp_system.db")
]

db_path = None
for path in possible_paths:
    if os.path.exists(path):
        db_path = path
        break

if not db_path:
    # Create the database in the most likely location
    os.makedirs("data", exist_ok=True)
    db_path = "data/csp_system.db"

print(f"Using SQLite database: {db_path}")

# Connect and create tables
conn = sqlite3.connect(db_path)

# Create the engines table
conn.execute('''
    CREATE TABLE IF NOT EXISTS engines (
        engine_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'initializing',
        consciousness_enabled BOOLEAN DEFAULT 1,
        quantum_enabled BOOLEAN DEFAULT 1,
        neural_mesh_enabled BOOLEAN DEFAULT 1,
        performance_tier TEXT DEFAULT 'standard',
        security_level TEXT DEFAULT 'standard',
        configuration TEXT,
        performance_metrics TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

# Create other missing tables that might be needed
conn.execute('''
    CREATE TABLE IF NOT EXISTS agents (
        agent_id TEXT PRIMARY KEY,
        engine_id TEXT,
        agent_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        consciousness_level REAL DEFAULT 0.8,
        quantum_enabled BOOLEAN DEFAULT 1,
        configuration TEXT,
        performance_metrics TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
    )
''')

conn.execute('''
    CREATE TABLE IF NOT EXISTS channels (
        channel_id TEXT PRIMARY KEY,
        engine_id TEXT,
        channel_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        configuration TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
    )
''')

conn.execute('''
    CREATE TABLE IF NOT EXISTS processes (
        process_id TEXT PRIMARY KEY,
        engine_id TEXT,
        process_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'ready',
        configuration TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
    )
''')

conn.commit()
conn.close()

print("✅ SQLite database tables created successfully!")
