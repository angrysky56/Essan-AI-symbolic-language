import random
import re
from typing import List, Tuple, Dict, Any, Optional, Set, Callable
import nltk
import spacy
from nltk.corpus import wordnet
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime
from collections import deque
from uuid import uuid4
from tabulate import tabulate  # For prettier console output

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NLP Manager
class NLPManager:
    _instance = None
    _nlp = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NLPManager, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        if self._nlp is None:
            try:
                for resource in ['punkt', 'wordnet', 'averaged_perceptron_tagger']:
                    try:
                        nltk.data.find(f'tokenizers/{resource}')
                    except LookupError:
                        nltk.download(resource)

                try:
                    self._nlp = spacy.load("en_core_web_sm")  # Using smaller model for faster loading
                except OSError:
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                    self._nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.error(f"Failed to initialize NLP components: {e}")
                raise
        return self._nlp

nlp = NLPManager().initialize()

# Enhanced Task Status with better state management
class TaskStatus(Enum):
    TODO = ("To Do", 1)
    IN_PROGRESS = ("In Progress", 2)
    BLOCKED = ("Blocked", 3)
    REVIEW = ("Review", 4)
    COMPLETED = ("Completed", 5)
    ARCHIVED = ("Archived", 6)
    CANCELLED = ("Cancelled", 7)

    def __init__(self, label: str, order: int):
        self.label = label
        self.order = order

    def __str__(self):
        return self.label

    def can_transition_to(self, new_status: 'TaskStatus') -> bool:
        """Define valid state transitions"""
        valid_transitions = {
            TaskStatus.TODO: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
            TaskStatus.IN_PROGRESS: {TaskStatus.BLOCKED, TaskStatus.REVIEW, TaskStatus.CANCELLED},
            TaskStatus.BLOCKED: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
            TaskStatus.REVIEW: {TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.CANCELLED},
            TaskStatus.COMPLETED: {TaskStatus.ARCHIVED},
            TaskStatus.ARCHIVED: set(),
            TaskStatus.CANCELLED: {TaskStatus.TODO}
        }
        return new_status in valid_transitions.get(self, set())

class TaskPriority(Enum):
    CRITICAL = 5, "Critical", "🔴"
    HIGH = 4, "High", "🟠"
    MEDIUM = 3, "Medium", "🟡"
    LOW = 2, "Low", "🟢"
    TRIVIAL = 1, "Trivial", "⚪"

    def __init__(self, order: int, label: str, icon: str):
        self._order = order  # Using _order to avoid conflict with enum's value
        self._label = label
        self._icon = icon

    @property
    def order(self) -> int:
        return self._order

    @property
    def label(self) -> str:
        return self._label

    @property
    def icon(self) -> str:
        return self._icon

    def __str__(self):
        return f"{self.icon} {self.label}"

    def __lt__(self, other):
        if not isinstance(other, TaskPriority):
            return NotImplemented
        return self.order < other.order

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    dependencies: Set[str] = field(default_factory=set)
    resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    subtasks: Set[str] = field(default_factory=set)
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    progress: float = 0.0  # Track progress percentage

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for display"""
        return {
            "ID": self.id[:8],
            "Description": self.description[:50] + "..." if len(self.description) > 50 else self.description,
            "Status": str(self.status),
            "Priority": str(self.priority),
            "Progress": f"{self.progress:.0f}%",
            "Due Date": self.due_date.strftime("%Y-%m-%d") if self.due_date else "Not set",
            "Dependencies": len(self.dependencies),
            "Subtasks": len(self.subtasks)
        }

    def update_modified(self):
        self.modified_at = datetime.now()

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.current_task: Optional[Task] = None

    def add_task(self, task: Task):
        self.tasks[task.id] = task
        self.current_task = task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def remove_task(self, task_id: str):
        if task_id in self.tasks:
            del self.tasks[task_id]
            if self.current_task and self.current_task.id == task_id:
                self.current_task = None

class EntityRelationship:
    def __init__(self, source: str, target: str, relationship_type: str, confidence: float = 1.0):
        self.id: str = str(uuid4())
        self.source = source
        self.target = target
        self.relationship_type = relationship_type
        self.created_at = datetime.now()
        self.confidence = confidence
        self.metadata: Dict[str, Any] = {}

    def __repr__(self):
        return f"{self.source} --[{self.relationship_type}:{self.confidence:.2f}]--> {self.target}"

class EntityRelationshipManager:
    def __init__(self):
        self.relationships: List[EntityRelationship] = []

    def add_relationship(self, relationship: EntityRelationship):
        self.relationships.append(relationship)

class EssanInterpreter:
    def __init__(self):
        self.nlp = nlp
        self.task_manager = TaskManager()
        self.entity_manager = EntityRelationshipManager()
        self.entities: Dict[str, List[str]] = {}
        self.context: Dict[str, Any] = {}
        self.action_history: List[Dict[str, Any]] = []

    def interpret(self, symbol_chain: str, task_description: str) -> Dict[str, Any]:
        """Process symbol chain with improved feedback"""
        results = {
            'actions': [],
            'entities': [],
            'status': 'success',
            'summary': [],
            'errors': []
        }

        print("\n=== Starting New Interpretation ===")
        print(f"Task: {task_description}")
        print(f"Symbol Chain: {symbol_chain}\n")

        try:
            for symbol in symbol_chain:
                action_result = self._execute_symbol_action(symbol, task_description)
                results['actions'].append(action_result)

                # Print real-time feedback
                self._print_action_feedback(symbol, action_result)

                if action_result.get('status') == 'error':
                    results['errors'].append(action_result.get('message'))

            # Print final summary
            self._print_final_summary(results)

        except Exception as e:
            logger.error(f"Error in interpretation: {e}")
            results['status'] = 'error'
            results['errors'].append(str(e))

        return results

    def _execute_symbol_action(self, symbol: str, task_description: str) -> Dict[str, Any]:
        """Execute action corresponding to a symbol"""
        symbol_actions = {
            "⧬": self.handle_initiate_task,
            "⨿": self.handle_extract_essence,
            "⧈": self.handle_create_connections,
            "⫰": self.handle_move_task,
            "⧉": self.handle_strengthen_priority,
            "⍞": self.handle_reduce_priority,
            "⧿": self.handle_create_subtasks,
            "⩘": self.handle_complete_task,
            "⩉": self.handle_query_tasks,
            "◬": self.handle_change_properties,
            "⧾": self.handle_set_purpose,
            "║": self.handle_set_boundary
        }

        action_method = symbol_actions.get(symbol)

        if action_method:
            return action_method(task_description)
        else:
            return {
                'status': 'error',
                'message': f"No action defined for symbol {symbol}"
            }

    def handle_initiate_task(self, task_description: str) -> Dict[str, Any]:
        """Initiate a new task with the given description"""
        task = Task(description=task_description)
        self.task_manager.add_task(task)
        return {
            'status': 'success',
            'task_id': task.id,
            'message': 'Task initiated'
        }

    def handle_extract_essence(self, task_description: str) -> Dict[str, Any]:
        """Extract key entities and relationships from the task description"""
        doc = self.nlp(task_description)
        entities = [ent.text for ent in doc.ents]
        # For simplicity, we'll just store the entities
        self.entities[task_description] = entities
        return {
            'status': 'success',
            'entities': entities,
            'message': 'Essence extracted'
        }

    def handle_create_connections(self, task_description: str) -> Dict[str, Any]:
        """Create connections between entities in the task description"""
        # For simplicity, assume we create relationships between consecutive entities
        entities = self.entities.get(task_description, [])
        relationships = []
        for i in range(len(entities) - 1):
            relationship = EntityRelationship(entities[i], entities[i+1], "related_to")
            self.entity_manager.add_relationship(relationship)
            relationships.append(str(relationship))
        return {
            'status': 'success',
            'relationships': relationships,
            'message': 'Connections created'
        }

    def handle_move_task(self, task_description: str) -> Dict[str, Any]:
        """Change the status of the current task"""
        task = self.task_manager.current_task
        if not task:
            return {'status': 'error', 'message': 'No current task to move.'}

        # For simplicity, we move the task to the next status in order
        status_order = sorted(TaskStatus, key=lambda s: s.order)  # sort by order
        current_index = status_order.index(task.status)
        if current_index < len(status_order) - 1:
            task.status = status_order[current_index + 1]
            task.update_modified()
            return {
                'status': 'success',
                'task_id': task.id,
                'new_status': str(task.status)
            }
        else:
            return {
                'status': 'success',
                'task_id': task.id,
                'message': 'Task already at final status'
            }

    def handle_strengthen_priority(self, task_description: str) -> Dict[str, Any]:
        """Increase task priority if possible"""
        task = self.task_manager.current_task
        if not task:
            return {'status': 'error', 'message': 'No current task to strengthen.'}

        # Get current priority order
        current_order = task.priority.order

        # Find next higher priority if available
        priority_options = sorted(TaskPriority, key=lambda x: x.order, reverse=True)
        for priority in priority_options:
            if priority.order > current_order:
                task.priority = priority
                task.update_modified()
                return {
                    'status': 'success',
                    'task_id': task.id,
                    'new_priority': str(task.priority)
                }

        return {
            'status': 'success',
            'task_id': task.id,
            'message': 'Task already at maximum priority'
        }

    def handle_reduce_priority(self, task_description: str) -> Dict[str, Any]:
        """Decrease task priority if possible"""
        task = self.task_manager.current_task
        if not task:
            return {'status': 'error', 'message': 'No current task to diminish.'}

        # Get current priority order
        current_order = task.priority.order

        # Find next lower priority if available
        priority_options = sorted(TaskPriority, key=lambda x: x.order)
        for priority in priority_options:
            if priority.order < current_order:
                task.priority = priority
                task.update_modified()
                return {
                    'status': 'success',
                    'task_id': task.id,
                    'new_priority': str(task.priority)
                }

        return {
            'status': 'success',
            'task_id': task.id,
            'message': 'Task already at minimum priority'
        }

    def handle_create_subtasks(self, task_description: str) -> Dict[str, Any]:
        """Create subtasks from the task description"""
        task = self.task_manager.current_task
        if not task:
            return {'status': 'error', 'message': 'No current task to add subtasks to.'}

        # Assume subtasks are separated by commas in task_description
        subtasks_descriptions = [desc.strip() for desc in task_description.split(',')]
        # Exclude the main task description
        subtasks_descriptions = subtasks_descriptions[1:]  # skip the main task description
        for sub_desc in subtasks_descriptions:
            subtask = Task(description=sub_desc, parent_id=task.id)
            task.subtasks.add(subtask.id)
            self.task_manager.add_task(subtask)
        task.update_modified()
        return {
            'status': 'success',
            'task_id': task.id,
            'subtasks_created': len(subtasks_descriptions)
        }

    def handle_complete_task(self, task_description: str) -> Dict[str, Any]:
        """Mark the current task as completed"""
        task = self.task_manager.current_task
        if not task:
            return {'status': 'error', 'message': 'No current task to complete.'}

        task.status = TaskStatus.COMPLETED
        task.update_modified()
        return {
            'status': 'success',
            'task_id': task.id,
            'message': 'Task completed'
        }

    def handle_query_tasks(self, task_description: str) -> Dict[str, Any]:
        """Query tasks based on criteria in task_description"""
        # For simplicity, we return all tasks
        tasks = [task.to_dict() for task in self.task_manager.tasks.values()]
        return {
            'status': 'success',
            'tasks': tasks,
            'message': 'Tasks queried'
        }

    def handle_change_properties(self, task_description: str) -> Dict[str, Any]:
        """Change properties of the current task based on task_description"""
        # For simplicity, we parse 'key=value' pairs in task_description
        task = self.task_manager.current_task
        if not task:
            return {'status': 'error', 'message': 'No current task to change properties.'}

        try:
            properties = dict(item.strip().split('=') for item in task_description.split(','))
            for key, value in properties.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.update_modified()
            return {
                'status': 'success',
                'task_id': task.id,
                'message': 'Properties updated'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Failed to parse properties: {e}"
            }

    def handle_set_purpose(self, task_description: str) -> Dict[str, Any]:
        """Set the purpose or goal of the current task"""
        task = self.task_manager.current_task
        if not task:
            return {'status': 'error', 'message': 'No current task to set purpose.'}

        task.metadata['purpose'] = task_description
        task.update_modified()
        return {
            'status': 'success',
            'task_id': task.id,
            'message': 'Purpose set'
        }

    def handle_set_boundary(self, task_description: str) -> Dict[str, Any]:
        """Set boundaries or constraints for the current task"""
        task = self.task_manager.current_task
        if not task:
            return {'status': 'error', 'message': 'No current task to set boundaries.'}

        task.metadata['boundary'] = task_description
        task.update_modified()
        return {
            'status': 'success',
            'task_id': task.id,
            'message': 'Boundary set'
        }

    def _print_action_feedback(self, symbol: str, result: Dict[str, Any]):
        """Print formatted feedback for each action"""
        symbol_meanings = {
            "⨿": "Extracting Essence",
            "⧈": "Creating Connections",
            "⧬": "Initiating Task",
            "⫰": "Moving Task",
            "⧿": "Creating Subtasks",
            "◬": "Changing Properties",
            "⧉": "Strengthening Priority",
            "⩘": "Completing Task",
            "⩉": "Querying Tasks",
            "⍞": "Reducing Priority",
            "⧾": "Setting Purpose",
            "║": "Setting Boundary"
        }

        print(f"\n▶ {symbol} {symbol_meanings.get(symbol, 'Unknown Action')}")

        if result['status'] == 'success':
            for key, value in result.items():
                if key != 'status':
                    print(f"  ✓ {key}: {value}")
        else:
            print(f"  ✗ Error: {result.get('message', 'Unknown error')}")

    def _print_final_summary(self, results: Dict[str, Any]):
        """Print formatted final summary"""
        print("\n=== Interpretation Summary ===")

        # Print task table if there are tasks
        if self.task_manager.tasks:
            tasks_data = [task.to_dict() for task in self.task_manager.tasks.values()]
            print("\nTasks:")
            print(tabulate(tasks_data, headers="keys", tablefmt="grid"))

        # Print relationships if any
        if self.entity_manager.relationships:
            print("\nRelationships:")
            for rel in self.entity_manager.relationships:
                print(f"  {rel}")

        # Print errors if any
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  ⚠ {error}")

# Example usage
if __name__ == "__main__":
    interpreter = EssanInterpreter()

    # Example 1: Create and process a task
    print("\n=== Example 1: Basic Task Creation and Processing ===")
    task_desc = "Analyze sales data from Q1 and generate monthly reports"
    symbol_chain = "⧬⨿⧈⫰⧉⩘"  # Create, analyze, connect, move, strengthen, complete
    results = interpreter.interpret(symbol_chain, task_desc)

    # Example 2: Create task with subtasks
    print("\n=== Example 2: Task with Subtasks ===")
    task_desc = "Implement new feature with unit tests and documentation"
    subtasks = "Write unit tests, Create documentation, Implement core functionality"
    symbol_chain = "⧬⧿"  # Create main task and add subtasks
    results = interpreter.interpret(symbol_chain, task_desc + "," + subtasks)
