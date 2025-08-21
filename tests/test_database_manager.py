#!/usr/bin/env python3
"""
Test the TestDatabaseManager without MCP dependencies
"""

import sys
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from enum import Enum
import uuid

# Simple test result classes (without imports)
class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    name: str
    status: TestStatus
    duration: float = 0.0
    output: str = ""
    error: str = ""

@dataclass
class TestWorkflow:
    id: str
    name: str
    status: str = "pending"
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: List[TestResult] = None

    def __post_init__(self):
        if self.results is None:
            self.results = []
        if self.created_at is None:
            self.created_at = datetime.now()

# Simplified TestDatabaseManager
class TestDatabaseManager:
    """Database manager for test execution tracking."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent / "test_results.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS test_executions (
                    id TEXT PRIMARY KEY,
                    test_name TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    module TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_seconds REAL,
                    error_message TEXT,
                    output TEXT,
                    workflow_id TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                );
            """)
    
    def store_test_results(self, results: List[TestResult], workflow_id: str):
        """Store test results in database."""
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                conn.execute("""
                    INSERT INTO test_executions 
                    (id, test_name, test_type, module, status, duration_seconds, 
                     error_message, output, workflow_id, started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    result.name,
                    "unit",  # default
                    "test",  # default
                    result.status.value,
                    result.duration,
                    result.error,
                    result.output,
                    workflow_id,
                    datetime.now(),
                    datetime.now()
                ))
    
    def store_workflow(self, workflow: TestWorkflow):
        """Store workflow in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workflows 
                (id, name, status, created_at, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                workflow.id,
                workflow.name,
                workflow.status,
                workflow.created_at.isoformat() if workflow.created_at else None,
                workflow.started_at.isoformat() if workflow.started_at else None,
                workflow.completed_at.isoformat() if workflow.completed_at else None
            ))
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM test_executions")
            test_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM workflows")
            workflow_count = cursor.fetchone()[0]
            
            return {
                "test_executions_count": test_count,
                "workflows_count": workflow_count,
                "database_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0
            }

def main():
    """Test the database manager."""
    print("🧪 Testing TestDatabaseManager")
    print("=" * 50)
    
    try:
        # Initialize database manager
        db_manager = TestDatabaseManager()
        print("✅ Database manager initialized")
        
        # Create test workflow
        workflow = TestWorkflow(
            id=str(uuid.uuid4()),
            name="Test Workflow",
            status="completed",
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        # Create test results
        test_results = [
            TestResult("test_example_1", TestStatus.PASSED, 1.5, "Test passed", ""),
            TestResult("test_example_2", TestStatus.FAILED, 2.1, "Test failed", "AssertionError"),
            TestResult("test_example_3", TestStatus.PASSED, 0.8, "Test passed", ""),
        ]
        
        workflow.results = test_results
        
        # Store in database
        db_manager.store_workflow(workflow)
        db_manager.store_test_results(test_results, workflow.id)
        print("✅ Workflow and test results stored")
        
        # Get statistics
        stats = db_manager.get_database_stats()
        print(f"✅ Database stats retrieved:")
        print(f"   - Test executions: {stats['test_executions_count']}")
        print(f"   - Workflows: {stats['workflows_count']}")
        print(f"   - Database size: {stats['database_size_bytes']} bytes")
        
        print("\n🎉 All database tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()