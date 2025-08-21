#!/usr/bin/env python3
"""
Simple database test to verify database functionality independently
"""

import sys
from pathlib import Path

# Add the tests directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Test direct database import
try:
    import sqlite3
    from datetime import datetime
    import uuid
    
    print("✅ Basic dependencies imported successfully")
    
    # Test database creation
    db_path = Path(__file__).parent / "test_simple.db"
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_table (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert test data
        test_id = str(uuid.uuid4())
        conn.execute("INSERT INTO test_table (id, name) VALUES (?, ?)", 
                    (test_id, "Test Entry"))
        
        # Query test data
        cursor = conn.execute("SELECT * FROM test_table WHERE id = ?", (test_id,))
        result = cursor.fetchone()
        
        if result:
            print(f"✅ Database creation and operations successful: {result}")
        else:
            print("❌ Database query failed")
    
    # Clean up
    if db_path.exists():
        db_path.unlink()
        print("✅ Database cleanup completed")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()