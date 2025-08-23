"""
Task Helper Mixin - Extracted from A2AAgentBase to reduce God Object complexity
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .types import TaskStatus

logger = logging.getLogger(__name__)


class TaskHelperMixin:
    """Mixin providing task management helper methods"""
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        if hasattr(self, 'tasks'):
            return self.tasks.get(task_id)
        return None
    
    def list_active_tasks(self) -> List[Dict[str, Any]]:
        """List all active tasks"""
        if not hasattr(self, 'tasks'):
            return []
        
        active_statuses = [TaskStatus.PENDING, TaskStatus.RUNNING]
        return [
            task for task in self.tasks.values() 
            if task.get("status") in active_statuses
        ]
    
    def list_completed_tasks(self) -> List[Dict[str, Any]]:
        """List all completed tasks"""
        if not hasattr(self, 'tasks'):
            return []
        
        completed_statuses = [TaskStatus.COMPLETED, TaskStatus.FAILED]
        return [
            task for task in self.tasks.values() 
            if task.get("status") in completed_statuses
        ]
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get comprehensive task statistics"""
        if not hasattr(self, 'tasks'):
            return {"error": "Task system not initialized"}
        
        total_tasks = len(self.tasks)
        active_tasks = len(self.list_active_tasks())
        completed_tasks = len(self.list_completed_tasks())
        
        # Calculate success rate
        completed_list = self.list_completed_tasks()
        successful = len([t for t in completed_list if t.get("status") == TaskStatus.COMPLETED])
        success_rate = (successful / len(completed_list)) * 100 if completed_list else 0
        
        return {
            "total_tasks": total_tasks,
            "active_tasks": active_tasks,
            "completed_tasks": completed_tasks,
            "success_rate_percent": round(success_rate, 2),
            "pending": len([t for t in self.tasks.values() if t.get("status") == TaskStatus.PENDING]),
            "running": len([t for t in self.tasks.values() if t.get("status") == TaskStatus.RUNNING]),
            "completed": len([t for t in self.tasks.values() if t.get("status") == TaskStatus.COMPLETED]),
            "failed": len([t for t in self.tasks.values() if t.get("status") == TaskStatus.FAILED])
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        if not hasattr(self, 'tasks') or task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.get("status") in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            task["status"] = TaskStatus.FAILED
            task["error"] = "Task cancelled by user"
            task["completed_at"] = datetime.utcnow().isoformat()
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up tasks older than specified hours"""
        if not hasattr(self, 'tasks'):
            return 0
        
        from datetime import timedelta
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        old_task_ids = []
        for task_id, task in self.tasks.items():
            task_time_str = task.get("created_at") or task.get("completed_at")
            if task_time_str:
                try:
                    task_time = datetime.fromisoformat(task_time_str.replace('Z', '+00:00'))
                    if task_time < cutoff_time and task.get("status") in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        old_task_ids.append(task_id)
                except ValueError:
                    continue
        
        # Remove old tasks
        for task_id in old_task_ids:
            del self.tasks[task_id]
        
        if old_task_ids:
            logger.info(f"Cleaned up {len(old_task_ids)} old tasks")
        
        return len(old_task_ids)