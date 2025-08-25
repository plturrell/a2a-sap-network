#!/usr/bin/env python3
"""
Intelligent Scan Manager for GleanAgent
Provides smart scheduling, historical tracking, and optimization of code analysis scans

Features:
- Historical scan tracking with detailed audit trails
- Smart scheduling based on file change detection
- Frequency optimization to avoid redundant scans
- Priority-based scanning for critical components
- Performance analytics and recommendations
- Change impact analysis
- Incremental scanning capabilities
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import subprocess
import git

logger = logging.getLogger(__name__)


class ScanPriority(Enum):
    """Scan priority levels"""
    CRITICAL = 1    # Core SDK, security-critical files
    HIGH = 2        # Main agent implementations
    MEDIUM = 3      # Supporting modules, utilities
    LOW = 4         # Documentation, tests, examples


class ChangeType(Enum):
    """Types of file changes"""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class FileMetadata:
    """Metadata for tracked files"""
    path: str
    size: int
    checksum: str
    last_modified: float
    last_scanned: Optional[float] = None
    scan_count: int = 0
    priority: ScanPriority = ScanPriority.MEDIUM
    issues_found: int = 0
    quality_score: float = 0.0
    change_frequency: float = 0.0  # changes per day


@dataclass
class ScanSession:
    """Represents a complete scan session"""
    session_id: str
    timestamp: float
    duration: float
    total_files: int
    files_scanned: int
    files_skipped: int
    issues_found: int
    quality_score: float
    triggered_by: str  # 'manual', 'scheduled', 'change_detected', 'forced'
    scan_types: List[str]
    directories: List[str]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeEvent:
    """Represents a file change event"""
    timestamp: float
    file_path: str
    change_type: ChangeType
    old_checksum: Optional[str] = None
    new_checksum: Optional[str] = None
    git_commit: Optional[str] = None
    author: Optional[str] = None


class IntelligentScanManager:
    """
    Manages intelligent scanning with historical tracking and optimization
    """

    def __init__(self, base_directory: str, db_path: str = None):
        self.base_directory = Path(base_directory)
        self.db_path = db_path or self.base_directory / "intelligent_scan_manager.db"

        # Initialize database
        self._init_database()

        # Scan configuration
        self.scan_frequencies = {
            ScanPriority.CRITICAL: timedelta(hours=6),   # Every 6 hours
            ScanPriority.HIGH: timedelta(hours=12),      # Every 12 hours
            ScanPriority.MEDIUM: timedelta(days=1),      # Daily
            ScanPriority.LOW: timedelta(days=3)          # Every 3 days
        }

        # File patterns for priority classification
        self.priority_patterns = {
            ScanPriority.CRITICAL: [
                "*/sdk/*",
                "*/core/*",
                "**/security*",
                "**/auth*",
                "**/blockchain*"
            ],
            ScanPriority.HIGH: [
                "*/agents/*/active/*",
                "*/agents/*Agent*",
                "**/server.py",
                "**/service.py"
            ],
            ScanPriority.MEDIUM: [
                "*/agents/*",
                "*/services/*",
                "*/utils/*"
            ],
            ScanPriority.LOW: [
                "**/test_*",
                "**/tests/*",
                "**/*_test.py",
                "**/docs/*",
                "**/examples/*",
                "**/*.md"
            ]
        }

        # Performance tracking
        self.performance_history = []
        self.change_detection_cache = {}

        # Git integration
        try:
            self.git_repo = git.Repo(self.base_directory, search_parent_directories=True)
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            self.git_repo = None
            logger.warning("Git repository not found - change tracking will be limited")

    def _init_database(self):
        """Initialize the intelligent scan manager database"""
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # File metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    path TEXT PRIMARY KEY,
                    size INTEGER,
                    checksum TEXT,
                    last_modified REAL,
                    last_scanned REAL,
                    scan_count INTEGER DEFAULT 0,
                    priority TEXT,
                    issues_found INTEGER DEFAULT 0,
                    quality_score REAL DEFAULT 0.0,
                    change_frequency REAL DEFAULT 0.0,
                    created_at REAL,
                    updated_at REAL
                )
            """)

            # Scan sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    duration REAL,
                    total_files INTEGER,
                    files_scanned INTEGER,
                    files_skipped INTEGER,
                    issues_found INTEGER,
                    quality_score REAL,
                    triggered_by TEXT,
                    scan_types TEXT,
                    directories TEXT,
                    performance_metrics TEXT,
                    created_at REAL
                )
            """)

            # Change events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS change_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    file_path TEXT,
                    change_type TEXT,
                    old_checksum TEXT,
                    new_checksum TEXT,
                    git_commit TEXT,
                    author TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at REAL
                )
            """)

            # Scan recommendations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    recommendation_type TEXT,
                    file_path TEXT,
                    reason TEXT,
                    priority INTEGER,
                    estimated_duration REAL,
                    executed BOOLEAN DEFAULT FALSE,
                    created_at REAL
                )
            """)

            # Performance analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    metric_name TEXT,
                    metric_value REAL,
                    context TEXT,
                    created_at REAL
                )
            """)

            conn.commit()

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
            return ""

    def _classify_file_priority(self, file_path: Path) -> ScanPriority:
        """Classify file priority based on path patterns"""
        path_str = str(file_path)

        for priority, patterns in self.priority_patterns.items():
            for pattern in patterns:
                if file_path.match(pattern) or Path(path_str).match(pattern):
                    return priority

        return ScanPriority.MEDIUM

    async def scan_directory_changes(self, directory: str = None) -> List[ChangeEvent]:
        """Scan for file changes since last check"""
        target_dir = Path(directory) if directory else self.base_directory
        changes = []
        current_time = time.time()

        # Get all Python files
        python_files = list(target_dir.rglob("*.py"))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for file_path in python_files:
                try:
                    file_stats = file_path.stat()
                    current_checksum = self._calculate_file_checksum(file_path)

                    # Get stored metadata
                    cursor.execute(
                        "SELECT checksum, last_modified FROM file_metadata WHERE path = ?",
                        (str(file_path),)
                    )
                    stored_data = cursor.fetchone()

                    if stored_data:
                        stored_checksum, stored_modified = stored_data

                        # Check if file changed
                        if current_checksum != stored_checksum:
                            change_event = ChangeEvent(
                                timestamp=current_time,
                                file_path=str(file_path),
                                change_type=ChangeType.MODIFIED,
                                old_checksum=stored_checksum,
                                new_checksum=current_checksum
                            )

                            # Try to get git info
                            if self.git_repo:
                                try:
                                    commits = list(self.git_repo.iter_commits(
                                        paths=str(file_path), max_count=1
                                    ))
                                    if commits:
                                        change_event.git_commit = commits[0].hexsha
                                        change_event.author = str(commits[0].author)
                                except Exception:
                                    pass

                            changes.append(change_event)

                            # Update metadata
                            self._update_file_metadata(file_path, file_stats, current_checksum)
                    else:
                        # New file
                        change_event = ChangeEvent(
                            timestamp=current_time,
                            file_path=str(file_path),
                            change_type=ChangeType.ADDED,
                            new_checksum=current_checksum
                        )
                        changes.append(change_event)

                        # Store new metadata
                        self._store_file_metadata(file_path, file_stats, current_checksum)

                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")

            # Store change events
            for change in changes:
                cursor.execute("""
                    INSERT INTO change_events
                    (timestamp, file_path, change_type, old_checksum, new_checksum, git_commit, author, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    change.timestamp, change.file_path, change.change_type.value,
                    change.old_checksum, change.new_checksum, change.git_commit,
                    change.author, current_time
                ))

            conn.commit()

        logger.info(f"Detected {len(changes)} file changes in {target_dir}")
        return changes

    def _store_file_metadata(self, file_path: Path, file_stats, checksum: str):
        """Store new file metadata"""
        priority = self._classify_file_priority(file_path)
        current_time = time.time()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO file_metadata
                (path, size, checksum, last_modified, priority, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(file_path), file_stats.st_size, checksum,
                file_stats.st_mtime, priority.name, current_time, current_time
            ))
            conn.commit()

    def _update_file_metadata(self, file_path: Path, file_stats, checksum: str):
        """Update existing file metadata"""
        current_time = time.time()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE file_metadata
                SET size = ?, checksum = ?, last_modified = ?, updated_at = ?
                WHERE path = ?
            """, (
                file_stats.st_size, checksum, file_stats.st_mtime,
                current_time, str(file_path)
            ))
            conn.commit()

    async def generate_scan_recommendations(self) -> List[Dict[str, Any]]:
        """Generate intelligent scan recommendations based on changes and priorities"""
        recommendations = []
        current_time = time.time()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Find files that need scanning based on priority and frequency
            for priority, frequency in self.scan_frequencies.items():
                frequency_seconds = frequency.total_seconds()

                cursor.execute("""
                    SELECT path, last_scanned, scan_count, change_frequency, issues_found
                    FROM file_metadata
                    WHERE priority = ?
                    AND (last_scanned IS NULL OR last_scanned < ?)
                    ORDER BY change_frequency DESC, issues_found DESC
                """, (priority.name, current_time - frequency_seconds))

                files_needing_scan = cursor.fetchall()

                for file_data in files_needing_scan:
                    path, last_scanned, scan_count, change_freq, issues = file_data

                    # Calculate priority score
                    time_since_scan = current_time - (last_scanned or 0)
                    priority_score = (
                        (5 - priority.value) * 20 +  # Base priority weight
                        min(change_freq * 10, 30) +   # Change frequency weight
                        min(issues * 5, 20) +         # Issues weight
                        min(time_since_scan / 3600, 10)  # Time weight
                    )

                    recommendations.append({
                        "file_path": path,
                        "priority": priority.name,
                        "priority_score": priority_score,
                        "reason": f"Due for {priority.name.lower()} priority scan",
                        "last_scanned": last_scanned,
                        "change_frequency": change_freq,
                        "issues_found": issues,
                        "estimated_duration": self._estimate_scan_duration(path)
                    })

            # Find files with recent changes
            cursor.execute("""
                SELECT DISTINCT file_path FROM change_events
                WHERE timestamp > ? AND processed = FALSE
            """, (current_time - 3600,))  # Last hour

            changed_files = cursor.fetchall()

            for (file_path,) in changed_files:
                if not any(r["file_path"] == file_path for r in recommendations):
                    recommendations.append({
                        "file_path": file_path,
                        "priority": "CHANGE_TRIGGERED",
                        "priority_score": 80,  # High priority for changes
                        "reason": "File recently changed",
                        "estimated_duration": self._estimate_scan_duration(file_path)
                    })

        # Sort by priority score
        recommendations.sort(key=lambda x: x["priority_score"], reverse=True)

        logger.info(f"Generated {len(recommendations)} scan recommendations")
        return recommendations

    def _estimate_scan_duration(self, file_path: str) -> float:
        """Estimate scan duration based on file size and historical data"""
        try:
            file_size = Path(file_path).stat().st_size
            # Base estimate: ~1 second per 10KB, with minimum of 2 seconds
            base_duration = max(2.0, file_size / 10240)

            # Adjust based on file type
            if "test" in file_path:
                base_duration *= 0.7  # Tests are usually faster
            elif any(pattern in file_path for pattern in ["sdk", "core", "blockchain"]):
                base_duration *= 1.5  # Core files take longer

            return min(base_duration, 60.0)  # Cap at 60 seconds
        except Exception:
            return 10.0  # Default estimate

    async def execute_smart_scan(self,
                                max_files: int = 50,
                                max_duration: float = 300,
                                force_priority: ScanPriority = None) -> ScanSession:
        """Execute a smart scan based on recommendations"""
        from . import GleanAgent  # Import here to avoid circular imports

        start_time = time.time()
        session_id = f"smart_scan_{int(start_time)}"

        # Get recommendations
        recommendations = await self.generate_scan_recommendations()

        if force_priority:
            recommendations = [
                r for r in recommendations
                if r.get("priority") == force_priority.name
            ]

        # Limit by max files and duration
        selected_files = []
        total_estimated_duration = 0

        for rec in recommendations[:max_files]:
            estimated = rec["estimated_duration"]
            if total_estimated_duration + estimated <= max_duration:
                selected_files.append(rec)
                total_estimated_duration += estimated
            else:
                break

        logger.info(f"Smart scan {session_id}: analyzing {len(selected_files)} files")

        # Initialize GleanAgent
        agent = GleanAgent()

        # Perform scans
        files_scanned = 0
        files_skipped = 0
        total_issues = 0
        quality_scores = []

        for file_rec in selected_files:
            file_path = file_rec["file_path"]

            try:
                # Run analysis on individual file
                result = await agent.analyze_code_refactoring(file_path, max_suggestions=10)

                if "error" not in result:
                    files_scanned += 1
                    issues = result.get("total_suggestions", 0)
                    total_issues += issues

                    # Calculate simple quality score
                    quality_score = max(0, 100 - (issues * 5))
                    quality_scores.append(quality_score)

                    # Update file metadata
                    self._update_scan_metadata(file_path, issues, quality_score)
                else:
                    files_skipped += 1
                    logger.warning(f"Failed to scan {file_path}: {result.get('error')}")

            except Exception as e:
                files_skipped += 1
                logger.error(f"Error scanning {file_path}: {e}")

        duration = time.time() - start_time
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        # Create session record
        session = ScanSession(
            session_id=session_id,
            timestamp=start_time,
            duration=duration,
            total_files=len(selected_files),
            files_scanned=files_scanned,
            files_skipped=files_skipped,
            issues_found=total_issues,
            quality_score=avg_quality,
            triggered_by="smart_algorithm",
            scan_types=["refactoring", "quality"],
            directories=[str(self.base_directory)],
            performance_metrics={
                "files_per_second": files_scanned / duration if duration > 0 else 0,
                "avg_scan_time": duration / files_scanned if files_scanned > 0 else 0,
                "estimated_vs_actual": total_estimated_duration / duration if duration > 0 else 1
            }
        )

        # Store session
        await self._store_scan_session(session)

        logger.info(f"Smart scan completed: {files_scanned} files, {total_issues} issues, {avg_quality:.1f}% quality")
        return session

    def _update_scan_metadata(self, file_path: str, issues_found: int, quality_score: float):
        """Update scan metadata for a file"""
        current_time = time.time()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE file_metadata
                SET last_scanned = ?, scan_count = scan_count + 1,
                    issues_found = ?, quality_score = ?, updated_at = ?
                WHERE path = ?
            """, (current_time, issues_found, quality_score, current_time, file_path))
            conn.commit()

    async def _store_scan_session(self, session: ScanSession):
        """Store scan session in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scan_sessions
                (session_id, timestamp, duration, total_files, files_scanned, files_skipped,
                 issues_found, quality_score, triggered_by, scan_types, directories,
                 performance_metrics, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id, session.timestamp, session.duration,
                session.total_files, session.files_scanned, session.files_skipped,
                session.issues_found, session.quality_score, session.triggered_by,
                json.dumps(session.scan_types), json.dumps(session.directories),
                json.dumps(session.performance_metrics), time.time()
            ))
            conn.commit()

    async def get_scan_history(self, days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
        """Get scan history for the specified period"""
        cutoff_time = time.time() - (days * 24 * 3600)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, timestamp, duration, total_files, files_scanned,
                       issues_found, quality_score, triggered_by
                FROM scan_sessions
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (cutoff_time, limit))

            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    "session_id": row[0],
                    "timestamp": datetime.fromtimestamp(row[1]).isoformat(),
                    "duration": row[2],
                    "total_files": row[3],
                    "files_scanned": row[4],
                    "issues_found": row[5],
                    "quality_score": row[6],
                    "triggered_by": row[7]
                })

        return sessions

    async def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive analytics dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Overall statistics
            cursor.execute("SELECT COUNT(*) FROM file_metadata")
            total_files = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM scan_sessions WHERE timestamp > ?",
                          (time.time() - 7 * 24 * 3600,))
            scans_last_week = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(quality_score) FROM file_metadata WHERE quality_score > 0")
            avg_quality = cursor.fetchone()[0] or 0

            # Priority distribution
            cursor.execute("""
                SELECT priority, COUNT(*), AVG(quality_score), SUM(issues_found)
                FROM file_metadata
                GROUP BY priority
            """)
            priority_stats = {}
            for row in cursor.fetchall():
                priority_stats[row[0]] = {
                    "file_count": row[1],
                    "avg_quality": row[2] or 0,
                    "total_issues": row[3] or 0
                }

            # Recent changes
            cursor.execute("""
                SELECT COUNT(*) FROM change_events
                WHERE timestamp > ?
            """, (time.time() - 24 * 3600,))
            changes_last_24h = cursor.fetchone()[0]

            # Performance trends
            cursor.execute("""
                SELECT timestamp, quality_score, issues_found
                FROM scan_sessions
                WHERE timestamp > ?
                ORDER BY timestamp
            """, (time.time() - 30 * 24 * 3600,))

            trend_data = []
            for row in cursor.fetchall():
                trend_data.append({
                    "date": datetime.fromtimestamp(row[0]).strftime("%Y-%m-%d"),
                    "quality": row[1],
                    "issues": row[2]
                })

        return {
            "overview": {
                "total_files_tracked": total_files,
                "scans_last_week": scans_last_week,
                "average_quality_score": avg_quality,
                "changes_last_24h": changes_last_24h
            },
            "priority_distribution": priority_stats,
            "trend_data": trend_data,
            "last_updated": datetime.now().isoformat()
        }

    async def schedule_optimal_scan(self) -> Dict[str, Any]:
        """Recommend optimal scan schedule based on change patterns"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Analyze change patterns by hour of day
            cursor.execute("""
                SELECT strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                       COUNT(*) as change_count
                FROM change_events
                WHERE timestamp > ?
                GROUP BY hour
                ORDER BY change_count
            """, (time.time() - 7 * 24 * 3600,))

            hourly_changes = dict(cursor.fetchall())

            # Find quietest hours (least changes)
            if hourly_changes:
                quietest_hours = sorted(hourly_changes.keys(),
                                      key=lambda x: hourly_changes[x])[:3]
            else:
                quietest_hours = ["02", "03", "04"]  # Default quiet hours

            # Calculate recommended frequencies
            cursor.execute("""
                SELECT priority, AVG(change_frequency), COUNT(*)
                FROM file_metadata
                GROUP BY priority
            """)

            priority_recommendations = {}
            for row in cursor.fetchall():
                priority, avg_change_freq, file_count = row

                # Adjust frequency based on change rate
                if avg_change_freq > 1.0:  # More than 1 change per day
                    recommended_freq = "6 hours"
                elif avg_change_freq > 0.5:  # More than 1 change per 2 days
                    recommended_freq = "12 hours"
                elif avg_change_freq > 0.1:  # More than 1 change per 10 days
                    recommended_freq = "24 hours"
                else:
                    recommended_freq = "72 hours"

                priority_recommendations[priority] = {
                    "recommended_frequency": recommended_freq,
                    "file_count": file_count,
                    "change_frequency": avg_change_freq or 0
                }

        return {
            "optimal_scan_hours": [int(h) for h in quietest_hours],
            "priority_recommendations": priority_recommendations,
            "generated_at": datetime.now().isoformat()
        }


# CLI Interface for Intelligent Scan Manager
class IntelligentScanCLI:
    """CLI interface for intelligent scan management"""

    def __init__(self, base_directory: str):
        self.manager = IntelligentScanManager(base_directory)

    async def scan_changes(self, directory: str = None):
        """Scan for file changes"""
        changes = await self.manager.scan_directory_changes(directory)

        print(f"🔍 CHANGE DETECTION RESULTS")
        print(f"📁 Directory: {directory or self.manager.base_directory}")
        print(f"📊 Changes found: {len(changes)}")
        print("-" * 50)

        for change in changes[:10]:  # Show first 10
            emoji = {"added": "🆕", "modified": "✏️", "deleted": "🗑️"}.get(change.change_type.value, "📝")
            print(f"{emoji} {change.change_type.value.title()}: {Path(change.file_path).name}")
            if change.author:
                print(f"   👤 Author: {change.author}")
            if change.git_commit:
                print(f"   📌 Commit: {change.git_commit[:8]}")

        if len(changes) > 10:
            print(f"   ... and {len(changes) - 10} more changes")

    async def show_recommendations(self):
        """Show scan recommendations"""
        recommendations = await self.manager.generate_scan_recommendations()

        print(f"🎯 INTELLIGENT SCAN RECOMMENDATIONS")
        print(f"📊 Total recommendations: {len(recommendations)}")
        print("-" * 60)

        for i, rec in enumerate(recommendations[:15], 1):
            score = rec["priority_score"]
            file_name = Path(rec["file_path"]).name

            print(f"{i:2d}. {score:5.1f} pts - {file_name}")
            print(f"    📂 {rec['priority']} priority")
            print(f"    💡 {rec['reason']}")
            print(f"    ⏱️  Est. duration: {rec['estimated_duration']:.1f}s")
            print()

    async def execute_smart_scan(self, max_files: int = 20):
        """Execute a smart scan"""
        print(f"🚀 EXECUTING SMART SCAN")
        print(f"📊 Max files: {max_files}")
        print("-" * 50)

        session = await self.manager.execute_smart_scan(max_files=max_files)

        print(f"✅ Smart scan completed!")
        print(f"📊 Session ID: {session.session_id}")
        print(f"⏱️  Duration: {session.duration:.1f}s")
        print(f"📁 Files scanned: {session.files_scanned}")
        print(f"⚠️  Issues found: {session.issues_found}")
        print(f"📈 Quality score: {session.quality_score:.1f}%")

        perf = session.performance_metrics
        print(f"🚄 Performance: {perf['files_per_second']:.1f} files/sec")

    async def show_analytics(self):
        """Show analytics dashboard"""
        analytics = await self.manager.get_analytics_dashboard()

        print(f"📈 ANALYTICS DASHBOARD")
        print("=" * 60)

        overview = analytics["overview"]
        print(f"📊 Overview:")
        print(f"   📁 Total files tracked: {overview['total_files_tracked']}")
        print(f"   🔍 Scans last week: {overview['scans_last_week']}")
        print(f"   📈 Average quality: {overview['average_quality_score']:.1f}%")
        print(f"   🔄 Changes last 24h: {overview['changes_last_24h']}")

        print(f"\n📋 Priority Distribution:")
        for priority, stats in analytics["priority_distribution"].items():
            print(f"   {priority}: {stats['file_count']} files, "
                  f"{stats['avg_quality']:.1f}% quality, "
                  f"{stats['total_issues']} issues")

        # Show recent trends
        trends = analytics["trend_data"][-7:]  # Last 7 days
        if trends:
            print(f"\n📈 Recent Quality Trends:")
            for trend in trends:
                print(f"   {trend['date']}: {trend['quality']:.1f}% quality, {trend['issues']} issues")

    async def show_history(self, days: int = 7):
        """Show scan history"""
        history = await self.manager.get_scan_history(days=days)

        print(f"📜 SCAN HISTORY (Last {days} days)")
        print("=" * 70)

        for session in history:
            timestamp = datetime.fromisoformat(session["timestamp"]).strftime("%Y-%m-%d %H:%M")
            print(f"🕐 {timestamp} - {session['triggered_by']}")
            print(f"   📁 {session['files_scanned']}/{session['total_files']} files")
            print(f"   ⚠️  {session['issues_found']} issues")
            print(f"   📈 {session['quality_score']:.1f}% quality")
            print(f"   ⏱️  {session['duration']:.1f}s duration")
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Intelligent Scan Manager CLI")
    parser.add_argument("command", choices=["changes", "recommendations", "scan", "analytics", "history"])
    parser.add_argument("--directory", "-d", help="Target directory")
    parser.add_argument("--max-files", "-m", type=int, default=20, help="Max files for smart scan")
    parser.add_argument("--days", type=int, default=7, help="Days for history")

    args = parser.parse_args()

    cli = IntelligentScanCLI(args.directory or ".")

    async def main():
        if args.command == "changes":
            await cli.scan_changes(args.directory)
        elif args.command == "recommendations":
            await cli.show_recommendations()
        elif args.command == "scan":
            await cli.execute_smart_scan(args.max_files)
        elif args.command == "analytics":
            await cli.show_analytics()
        elif args.command == "history":
            await cli.show_history(args.days)

    asyncio.run(main())
