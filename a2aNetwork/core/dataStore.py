"""
DataStore for LNN Fallback Client

Provides a persistent storage solution for training and validation data using SQLite.
"""

import sqlite3
import json
import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DataStore:
    """Handles SQLite database operations for LNN training data."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'data', 'lnn_training.db')
        
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.db_path)
            self._create_table()
            logger.info(f"DataStore initialized with database at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _create_table(self):
        """Creates the training_data table if it doesn't exist."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    expected_result TEXT NOT NULL,
                    data_type TEXT NOT NULL, -- 'train' or 'validation'
                    timestamp TEXT NOT NULL
                )
            """)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to create table: {e}")

    def add_data(self, prompt: str, expected_result: Dict[str, Any], data_type: str):
        """Adds a new data point to the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO training_data (prompt, expected_result, data_type, timestamp)
                VALUES (?, ?, ?, datetime('now'))
            """, (prompt, json.dumps(expected_result), data_type))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to add data: {e}")

    def get_data(self, data_type: str) -> List[Dict[str, Any]]:
        """Retrieves all data of a specific type."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT prompt, expected_result FROM training_data WHERE data_type = ?", (data_type,))
            rows = cursor.fetchall()
            return [{'prompt': row[0], 'expected': json.loads(row[1])} for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get data: {e}")
            return []

    def get_data_count(self, data_type: str) -> int:
        """Gets the count of data of a specific type."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM training_data WHERE data_type = ?", (data_type,))
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Failed to get data count: {e}")
            return 0

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")
