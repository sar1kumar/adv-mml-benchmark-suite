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
        """Validate model configuration with batch processing support"""
        # Basic required fields for all models
        required_fields = {
            "type": str,
            "mode": str,  # realtime or batch
            "model_name": str,
            "max_tokens": int,
            "temperature": float
        }
        
        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in model config")
            if not isinstance(config[field], field_type):
                raise TypeError(f"Field '{field}' must be of type {field_type.__name__}")
                
        # Validate model type
        if config["type"] not in ["ollama", "api"]:
            raise ValueError("Model type must be either 'ollama' or 'api'")
            
        # Validate mode
        valid_modes = ["realtime", "batch"]
        if config["mode"] not in valid_modes:
            raise ValueError(f"Model mode must be one of {valid_modes}, got '{config['mode']}'")
            
        # Mode-specific validation
        if config["mode"] == "realtime":
            # Realtime mode requires API key
            if config["type"] == "api" and "api_key" not in config:
                raise ValueError("Realtime mode requires 'api_key' field in model config")
                
        elif config["mode"] == "batch":
            if config["type"] != "api":
                raise ValueError("Batch mode is only supported for API models (type: 'api')")
                
            # Check for required batch fields (no api_key needed)
            batch_required_fields = {
                "project_id": str,
                "location": str
            }
            
            for field, field_type in batch_required_fields.items():
                if field not in config:
                    raise ValueError(f"Batch mode requires '{field}' field in model config")
                if not isinstance(config[field], field_type):
                    raise TypeError(f"Batch mode field '{field}' must be of type {field_type.__name__}")
                    
            # Validate batch_config section
            if "batch_config" not in config:
                raise ValueError("Batch mode requires 'batch_config' section in model config")
                
            ConfigParser._validate_batch_config(config["batch_config"])

    @staticmethod
    def _validate_batch_config(batch_config: Dict[str, Any]) -> None:
        """Validate the batch_config section"""
        required_batch_fields = {
            "gcs_bucket": str,
            "gcs_image_prefix": str,
            "output_prefix": str
        }
        
        for field, field_type in required_batch_fields.items():
            if field not in batch_config:
                raise ValueError(f"Missing required batch config field '{field}'")
            if not isinstance(batch_config[field], field_type):
                raise TypeError(f"Batch config field '{field}' must be of type {field_type.__name__}")
                
        # Validate GCS URIs format
        gcs_fields = ["gcs_bucket", "gcs_image_prefix", "output_prefix"]
        for field in gcs_fields:
            if field in batch_config:
                uri = batch_config[field]
                if not uri.startswith("gs://"):
                    raise ValueError(f"Batch config field '{field}' must be a valid GCS URI starting with 'gs://', got '{uri}'")
                    
        # Validate optional fields with correct types
        optional_batch_fields = {
            "upload_images": bool,
            "wait_for_completion": bool,
            "poll_interval": int,
            "cleanup_after_completion": bool,
            "gcs_requests_file": str,
            "gcs_metadata_file": str
        }
        
        for field, field_type in optional_batch_fields.items():
            if field in batch_config and not isinstance(batch_config[field], field_type):
                raise TypeError(f"Optional batch config field '{field}' must be of type {field_type.__name__}")

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
        valid_metrics = ["accuracy", "f1", "bleu", "rouge", "mean_iou", "meteor", "cider"]
        for metric in config["metrics"]:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")

    @staticmethod
    def validate_config_consistency(config: Dict[str, Any]) -> None:
        """Validate that configuration is consistent across models and tasks"""
        model_modes = [model_config["mode"] for model_config in config["models"].values()]
        
        # Check if mixing realtime and batch modes
        if "realtime" in model_modes and "batch" in model_modes:
            raise ValueError(
                "Cannot mix realtime and batch models in the same configuration. "
                "Please use separate config files for different processing modes."
            )
            
        # Validate batch mode requirements
        if "batch" in model_modes:
            batch_models = [
                name for name, model_config in config["models"].items() 
                if model_config["mode"] == "batch"
            ]
            
            if len(batch_models) > 1:
                raise ValueError(
                    "Only one batch model allowed per configuration. "
                    f"Found batch models: {batch_models}"
                )
                
        # Validate that tasks support the processing mode
        processing_mode = model_modes[0] if model_modes else "realtime"
        
        if processing_mode == "batch":
            # For batch mode, we could add validation to ensure tasks support batch processing
            # This would require loading the task classes, so we'll keep it simple for now
            pass

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
            try:
                ConfigParser.validate_model_config(model_config)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error in model '{model_name}': {e}")
            
        # Validate configuration consistency
        ConfigParser.validate_config_consistency(config)
            
        # Validate each task config
        for task_name, task_config in config["tasks"].items():
            try:
                ConfigParser.validate_task_config(task_config)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error in task '{task_name}': {e}")
            
        # Validate logging config
        if "wandb" in config["logging"]:
            wandb_config = config["logging"]["wandb"]
            if "project" not in wandb_config or "entity" not in wandb_config:
                raise ValueError("Weights & Biases config must include 'project' and 'entity'")
                
        return config 