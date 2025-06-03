import os
import yaml
from typing import Dict, Any, List

class ConfigParser:
    @staticmethod
    def _substitute_env_vars(value: Any) -> Any:
        """Recursively substitute environment variables in config values"""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            if env_var not in os.environ:
                raise ValueError(f"Environment variable {env_var} not found")
            return os.environ[env_var]
        elif isinstance(value, dict):
            return {k: ConfigParser._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [ConfigParser._substitute_env_vars(v) for v in value]
        return value

    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> None:
        """Validate model configuration"""
        required_fields = {
            "type": str,
            "model_name": str,
            "max_tokens": int,
            "temperature": float
        }
        
        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in model config")
            if not isinstance(config[field], field_type):
                raise TypeError(f"Field '{field}' must be of type {field_type.__name__}")
                
        if config["type"] not in ["ollama", "api"]:
            raise ValueError("Model type must be either 'ollama' or 'api'")

    @staticmethod
    def validate_task_config(config: Dict[str, Any]) -> None:
        """Validate task configuration"""
        required_fields = {
            "data_path": str,
            "metrics": list
        }
        
        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in task config")
            if not isinstance(config[field], field_type):
                raise TypeError(f"Field '{field}' must be of type {field_type.__name__}")
                
        # Validate metrics
        valid_metrics = ["accuracy", "f1", "bleu", "rouge"]
        for metric in config["metrics"]:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")

    @staticmethod
    def load_and_validate(config_path: str) -> Dict[str, Any]:
        """Load config file and validate its contents"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Substitute environment variables
        config = ConfigParser._substitute_env_vars(config)
        
        # Validate required top-level sections
        required_sections = ["models", "tasks", "logging"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config")
                
        # Validate each model config
        for model_name, model_config in config["models"].items():
            ConfigParser.validate_model_config(model_config)
            
        # Validate each task config
        for task_name, task_config in config["tasks"].items():
            ConfigParser.validate_task_config(task_config)
            
        # Validate logging config
        if "wandb" in config["logging"]:
            wandb_config = config["logging"]["wandb"]
            if "project" not in wandb_config or "entity" not in wandb_config:
                raise ValueError("Weights & Biases config must include 'project' and 'entity'")
                
        return config 