import importlib
from typing import Dict, Any, Type

from benchmark_suite.tasks.base_task import BaseTask
from benchmark_suite.tasks.vqa_rad import VQARADTask
from benchmark_suite.tasks.erqa import ERQATask
from benchmark_suite.tasks.sme_task import SMETask
from benchmark_suite.tasks.omni_vqa import OmniMedVQATask

class TaskManager:
    """Manages task registration and initialization"""
    
    _TASK_REGISTRY = {
        "vqa_rad": VQARADTask,
        "erqa": ERQATask,
        "sme": SMETask,
        "omnimed_vqa": OmniMedVQATask

    }
    
    @classmethod
    def register_task(cls, task_name: str, task_class: Type[BaseTask]) -> None:
        """Register a new task"""
        if not issubclass(task_class, BaseTask):
            raise TypeError(f"Task class must inherit from BaseTask")
        cls._TASK_REGISTRY[task_name] = task_class
        
    @classmethod
    def get_task_class(cls, task_name: str) -> Type[BaseTask]:
        """Get the task class for a given name"""
        if task_name not in cls._TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_name}")
        return cls._TASK_REGISTRY[task_name]
        
    @classmethod
    def load_task(cls, task_name: str, task_config: Dict[str, Any]) -> BaseTask:
        """Load and initialize a task from config"""
        task_class = cls.get_task_class(task_name)
        
        try:
            task = task_class(task_config)
            task.load_data()  # Initialize the task data
            return task
        except Exception as e:
            raise RuntimeError(f"Failed to load task {task_name}: {str(e)}")
            
    @classmethod
    def load_tasks(cls, config: Dict[str, Any]) -> Dict[str, BaseTask]:
        """Load all tasks specified in config"""
        tasks = {}
        for task_name, task_config in config["tasks"].items():
            try:
                tasks[task_name] = cls.load_task(task_name, task_config)
            except Exception as e:
                raise RuntimeError(f"Failed to load task '{task_name}': {str(e)}")
        return tasks
        
    @classmethod
    def load_external_task(cls, module_path: str, class_name: str) -> Type[BaseTask]:
        """Dynamically load an external task class"""
        try:
            module = importlib.import_module(module_path)
            task_class = getattr(module, class_name)
            
            if not issubclass(task_class, BaseTask):
                raise TypeError(f"Task class {class_name} must inherit from BaseTask")
                
            return task_class
        except ImportError:
            raise ImportError(f"Failed to import task from {module_path}")
        except AttributeError:
            raise AttributeError(f"Task class {class_name} not found in {module_path}")
            
    @classmethod
    def register_external_task(cls, task_name: str, module_path: str, class_name: str) -> None:
        """Register an external task"""
        task_class = cls.load_external_task(module_path, class_name)
        cls.register_task(task_name, task_class)
        
    @classmethod
    def get_available_tasks(cls) -> Dict[str, str]:
        """Get a dictionary of available tasks and their descriptions"""
        return {
            task_name: task_class.__doc__ or "No description available"
            for task_name, task_class in cls._TASK_REGISTRY.items()
        } 