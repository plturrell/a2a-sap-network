"""
A2A Task Tracking System
Provides comprehensive tracking of tasks, help requests, and completion status for all agents
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4
from enum import Enum
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_HELP = "waiting_for_help"
    HELP_RECEIVED = "help_received"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HelpRequestStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    RESPONDED = "responded"
    APPLIED = "applied"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ChecklistItem(BaseModel):
    """Individual checklist item within a task"""

    item_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: List[str] = Field(default_factory=list)  # IDs of prerequisite items
    help_request_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: List[str] = Field(default_factory=list)
    estimated_duration: Optional[int] = None  # minutes
    actual_duration: Optional[int] = None  # minutes


class HelpRequest(BaseModel):
    """Tracks help requests made by the agent"""

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    checklist_item_id: Optional[str] = None
    problem_type: str
    problem_description: str
    target_agent: str
    urgency: str = "medium"
    status: HelpRequestStatus = HelpRequestStatus.PENDING
    request_sent_at: Optional[datetime] = None
    response_received_at: Optional[datetime] = None
    help_applied_at: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    effectiveness_rating: Optional[int] = None  # 1-5 scale
    follow_up_needed: bool = False
    context: Dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """Comprehensive task tracking with checklist and help request monitoring"""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    checklist: List[ChecklistItem] = Field(default_factory=list)
    help_requests: List[HelpRequest] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None

    context_id: Optional[str] = None
    assigned_to: Optional[str] = None
    requester: Optional[str] = None

    progress_percentage: float = 0.0
    estimated_duration: Optional[int] = None  # minutes
    actual_duration: Optional[int] = None  # minutes

    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    error_history: List[Dict[str, Any]] = Field(default_factory=list)


class AgentTaskTracker:
    """Comprehensive task and help request tracking for A2A agents"""

    def __init__(self, agent_id: str, agent_name: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.tasks: Dict[str, Task] = {}
        self.active_help_requests: Dict[str, HelpRequest] = {}
        self.help_request_history: List[HelpRequest] = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "help_requests_sent": 0,
            "help_requests_successful": 0,
            "average_task_duration": 0.0,
            "average_help_response_time": 0.0,
        }

        logger.info(f"✅ Task tracker initialized for {agent_name} ({agent_id})")

    def create_task(
        self,
        name: str,
        description: str,
        checklist_items: List[str],
        priority: TaskPriority = TaskPriority.MEDIUM,
        estimated_duration: Optional[int] = None,
        deadline: Optional[datetime] = None,
        context_id: Optional[str] = None,
        tags: List[str] = None,
    ) -> str:
        """Create a new task with checklist items"""

        task = Task(
            name=name,
            description=description,
            priority=priority,
            estimated_duration=estimated_duration,
            deadline=deadline,
            context_id=context_id,
            assigned_to=self.agent_id,
            tags=tags or [],
        )

        # Create checklist items
        checklist = []
        for item_desc in checklist_items:
            checklist_item = ChecklistItem(
                description=item_desc,
                estimated_duration=(
                    estimated_duration // len(checklist_items) if estimated_duration else None
                ),
            )
            checklist.append(checklist_item)

        task.checklist = checklist

        self.tasks[task.task_id] = task

        logger.info(f"📋 Created task '{name}' with {len(checklist_items)} checklist items")
        return task.task_id

    def start_task(self, task_id: str) -> bool:
        """Start a task and mark it as in progress"""
        if task_id not in self.tasks:
            logger.error(f"❌ Task {task_id} not found")
            return False

        task = self.tasks[task_id]
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()

        logger.info(f"🚀 Started task '{task.name}' ({task_id})")
        return True

    def start_checklist_item(self, task_id: str, item_id: str) -> bool:
        """Start a specific checklist item"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        for item in task.checklist:
            if item.item_id == item_id:
                # Check dependencies
                if not self._are_dependencies_met(task, item):
                    logger.warning(
                        f"⚠️ Cannot start item '{item.description}' - dependencies not met"
                    )
                    return False

                item.status = TaskStatus.IN_PROGRESS
                item.started_at = datetime.utcnow()

                # Update task status if this is the first item being worked on
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.IN_PROGRESS
                    task.started_at = datetime.utcnow()

                logger.info(f"▶️ Started checklist item: {item.description}")
                self._update_task_progress(task_id)
                return True

        return False

    def complete_checklist_item(self, task_id: str, item_id: str, notes: str = None) -> bool:
        """Complete a checklist item"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        for item in task.checklist:
            if item.item_id == item_id:
                item.status = TaskStatus.COMPLETED
                item.completed_at = datetime.utcnow()

                if item.started_at:
                    duration = (item.completed_at - item.started_at).total_seconds() / 60
                    item.actual_duration = int(duration)

                if notes:
                    item.notes.append(f"{datetime.utcnow().isoformat()}: {notes}")

                logger.info(f"✅ Completed checklist item: {item.description}")
                self._update_task_progress(task_id)
                self._check_task_completion(task_id)
                return True

        return False

    def fail_checklist_item(
        self, task_id: str, item_id: str, error: str, seek_help: bool = True
    ) -> bool:
        """Fail a checklist item and optionally trigger help-seeking"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        for item in task.checklist:
            if item.item_id == item_id:
                item.status = TaskStatus.FAILED
                item.notes.append(f"{datetime.utcnow().isoformat()}: FAILED - {error}")

                # Update task error tracking
                task.error_count += 1
                task.last_error = error
                task.error_history.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "item_id": item_id,
                        "error": error,
                        "seek_help": seek_help,
                    }
                )

                logger.error(f"❌ Failed checklist item: {item.description} - {error}")

                if seek_help:
                    item.status = TaskStatus.WAITING_FOR_HELP
                    logger.info(f"🆘 Item marked as waiting for help: {item.description}")

                self._update_task_progress(task_id)
                return True

        return False

    def create_help_request(
        self,
        task_id: str,
        problem_type: str,
        problem_description: str,
        target_agent: str,
        urgency: str = "medium",
        checklist_item_id: str = None,
        context: Dict[str, Any] = None,
    ) -> str:
        """Create a help request for a task"""

        help_request = HelpRequest(
            task_id=task_id,
            checklist_item_id=checklist_item_id,
            problem_type=problem_type,
            problem_description=problem_description,
            target_agent=target_agent,
            urgency=urgency,
            context=context or {},
        )

        self.active_help_requests[help_request.request_id] = help_request

        # Link to checklist item if specified
        if checklist_item_id and task_id in self.tasks:
            task = self.tasks[task_id]
            for item in task.checklist:
                if item.item_id == checklist_item_id:
                    item.help_request_id = help_request.request_id
                    break

            task.help_requests.append(help_request)

        logger.info(f"🆘 Created help request for '{problem_type}' targeting {target_agent}")
        return help_request.request_id

    def mark_help_request_sent(self, request_id: str) -> bool:
        """Mark help request as sent"""
        if request_id in self.active_help_requests:
            help_request = self.active_help_requests[request_id]
            help_request.status = HelpRequestStatus.SENT
            help_request.request_sent_at = datetime.utcnow()
            self.performance_metrics["help_requests_sent"] += 1

            logger.info(f"📤 Help request sent: {request_id}")
            return True
        return False

    def receive_help_response(self, request_id: str, response_data: Dict[str, Any]) -> bool:
        """Receive and process help response"""
        if request_id in self.active_help_requests:
            help_request = self.active_help_requests[request_id]
            help_request.status = HelpRequestStatus.RESPONDED
            help_request.response_received_at = datetime.utcnow()
            help_request.response_data = response_data

            # Calculate response time
            if help_request.request_sent_at:
                response_time = (
                    help_request.response_received_at - help_request.request_sent_at
                ).total_seconds()
                self._update_help_response_metrics(response_time)

            # Update associated checklist item
            if help_request.checklist_item_id and help_request.task_id in self.tasks:
                task = self.tasks[help_request.task_id]
                for item in task.checklist:
                    if item.item_id == help_request.checklist_item_id:
                        item.status = TaskStatus.HELP_RECEIVED
                        item.notes.append(
                            f"{datetime.utcnow().isoformat()}: Help received from {help_request.target_agent}"
                        )
                        break

            logger.info(f"📥 Help response received for request: {request_id}")
            return True
        return False

    def apply_help_solution(
        self, request_id: str, effectiveness_rating: int = 5, notes: str = None
    ) -> bool:
        """Apply help solution and rate its effectiveness"""
        if request_id in self.active_help_requests:
            help_request = self.active_help_requests[request_id]
            help_request.status = HelpRequestStatus.APPLIED
            help_request.help_applied_at = datetime.utcnow()
            help_request.effectiveness_rating = effectiveness_rating

            if effectiveness_rating >= 3:
                self.performance_metrics["help_requests_successful"] += 1

            # Update associated checklist item
            if help_request.checklist_item_id and help_request.task_id in self.tasks:
                task = self.tasks[help_request.task_id]
                for item in task.checklist:
                    if item.item_id == help_request.checklist_item_id:
                        if effectiveness_rating >= 3:
                            item.status = TaskStatus.IN_PROGRESS  # Ready to retry
                            if notes:
                                item.notes.append(
                                    f"{datetime.utcnow().isoformat()}: Help applied - {notes}"
                                )
                        else:
                            help_request.follow_up_needed = True
                            item.notes.append(
                                f"{datetime.utcnow().isoformat()}: Help ineffective (rating: {effectiveness_rating})"
                            )
                        break

            # Move to history
            self.help_request_history.append(help_request)
            del self.active_help_requests[request_id]

            logger.info(f"🔧 Help solution applied with rating {effectiveness_rating}/5")
            return True
        return False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive task status"""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]

        # Calculate completion stats
        completed_items = sum(1 for item in task.checklist if item.status == TaskStatus.COMPLETED)
        total_items = len(task.checklist)
        progress = (completed_items / total_items * 100) if total_items > 0 else 0

        # Calculate duration
        duration = None
        if task.started_at:
            end_time = task.completed_at or datetime.utcnow()
            duration = int((end_time - task.started_at).total_seconds() / 60)

        return {
            "task_id": task_id,
            "name": task.name,
            "description": task.description,
            "status": task.status,
            "priority": task.priority,
            "progress_percentage": progress,
            "checklist_summary": {
                "total_items": total_items,
                "completed": completed_items,
                "in_progress": sum(
                    1 for item in task.checklist if item.status == TaskStatus.IN_PROGRESS
                ),
                "waiting_for_help": sum(
                    1 for item in task.checklist if item.status == TaskStatus.WAITING_FOR_HELP
                ),
                "failed": sum(1 for item in task.checklist if item.status == TaskStatus.FAILED),
            },
            "help_requests": {
                "active": len(
                    [
                        hr
                        for hr in task.help_requests
                        if hr.status in [HelpRequestStatus.PENDING, HelpRequestStatus.SENT]
                    ]
                ),
                "completed": len(
                    [hr for hr in task.help_requests if hr.status == HelpRequestStatus.APPLIED]
                ),
            },
            "duration_minutes": duration,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error_count": task.error_count,
            "last_error": task.last_error,
        }

    def get_all_tasks_summary(self) -> Dict[str, Any]:
        """Get summary of all tasks"""
        total_tasks = len(self.tasks)
        completed_tasks = sum(
            1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED
        )
        in_progress_tasks = sum(
            1 for task in self.tasks.values() if task.status == TaskStatus.IN_PROGRESS
        )
        failed_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED)

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "summary": {
                "total_tasks": total_tasks,
                "completed": completed_tasks,
                "in_progress": in_progress_tasks,
                "failed": failed_tasks,
                "completion_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            },
            "help_requests": {
                "active": len(self.active_help_requests),
                "total_sent": self.performance_metrics["help_requests_sent"],
                "successful": self.performance_metrics["help_requests_successful"],
                "success_rate": (
                    (
                        self.performance_metrics["help_requests_successful"]
                        / self.performance_metrics["help_requests_sent"]
                        * 100
                    )
                    if self.performance_metrics["help_requests_sent"] > 0
                    else 0
                ),
            },
            "performance": self.performance_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_active_help_requests(self) -> List[Dict[str, Any]]:
        """Get all active help requests"""
        return [
            {
                "request_id": hr.request_id,
                "task_id": hr.task_id,
                "problem_type": hr.problem_type,
                "target_agent": hr.target_agent,
                "status": hr.status,
                "urgency": hr.urgency,
                "created_at": hr.request_sent_at.isoformat() if hr.request_sent_at else "Not sent",
                "waiting_time_minutes": (
                    int((datetime.utcnow() - hr.request_sent_at).total_seconds() / 60)
                    if hr.request_sent_at
                    else 0
                ),
            }
            for hr in self.active_help_requests.values()
        ]

    def _are_dependencies_met(self, task: Task, item: ChecklistItem) -> bool:
        """Check if all dependencies for a checklist item are met"""
        if not item.depends_on:
            return True

        for dep_id in item.depends_on:
            for dep_item in task.checklist:
                if dep_item.item_id == dep_id and dep_item.status != TaskStatus.COMPLETED:
                    return False
        return True

    def _update_task_progress(self, task_id: str):
        """Update task progress based on checklist completion"""
        task = self.tasks[task_id]
        if not task.checklist:
            return

        completed = sum(1 for item in task.checklist if item.status == TaskStatus.COMPLETED)
        total = len(task.checklist)
        task.progress_percentage = (completed / total) * 100

    def _check_task_completion(self, task_id: str):
        """Check if task is complete and update status"""
        task = self.tasks[task_id]

        if all(item.status == TaskStatus.COMPLETED for item in task.checklist):
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()

            if task.started_at:
                duration = (task.completed_at - task.started_at).total_seconds() / 60
                task.actual_duration = int(duration)
                self._update_task_duration_metrics(duration)

            self.performance_metrics["tasks_completed"] += 1
            logger.info(f"🎉 Task completed: {task.name}")

        elif any(item.status == TaskStatus.FAILED for item in task.checklist):
            # Check if we have help requests pending for failed items
            failed_items_with_help = any(
                item.status == TaskStatus.WAITING_FOR_HELP
                for item in task.checklist
                if item.status in [TaskStatus.FAILED, TaskStatus.WAITING_FOR_HELP]
            )

            if not failed_items_with_help:
                task.status = TaskStatus.FAILED
                self.performance_metrics["tasks_failed"] += 1
                logger.warning(f"💥 Task failed: {task.name}")

    def _update_help_response_metrics(self, response_time_seconds: float):
        """Update help response time metrics"""
        current_avg = self.performance_metrics["average_help_response_time"]
        total_requests = self.performance_metrics["help_requests_sent"]

        if total_requests > 1:
            self.performance_metrics["average_help_response_time"] = (
                current_avg * (total_requests - 1) + response_time_seconds
            ) / total_requests
        else:
            self.performance_metrics["average_help_response_time"] = response_time_seconds

    def _update_task_duration_metrics(self, duration_minutes: float):
        """Update task duration metrics"""
        current_avg = self.performance_metrics["average_task_duration"]
        completed_tasks = self.performance_metrics["tasks_completed"]

        if completed_tasks > 1:
            self.performance_metrics["average_task_duration"] = (
                current_avg * (completed_tasks - 1) + duration_minutes
            ) / completed_tasks
        else:
            self.performance_metrics["average_task_duration"] = duration_minutes
