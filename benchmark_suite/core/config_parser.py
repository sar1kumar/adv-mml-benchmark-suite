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
        """Validate task configuration with adversarial support"""
        required_fields = {
            "data_path": str,
            "metrics": list
        }
        
        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in task config")
            if not isinstance(config[field], field_type):
                raise TypeError(f"Field '{field}' must be of type {field_type.__name__}")
                
        # Validate adversarial config if present
        if "adversarial" in config:
            ConfigParser._validate_adversarial_config(config["adversarial"])

    @staticmethod
    def _validate_adversarial_config(adversarial_config: Dict[str, Any]) -> None:
        """Validate adversarial attack configuration"""
        if not isinstance(adversarial_config, dict):
            raise TypeError("Adversarial config must be a dictionary")
            
        # Check if adversarial attacks are enabled
        if not adversarial_config.get("enabled", False):
            return  # Skip validation if disabled
            
        required_fields = {
            "generator_type": str,
            "attack_mode": str,
        }
        
        for field, field_type in required_fields.items():
            if field not in adversarial_config:
                raise ValueError(f"Missing required adversarial field '{field}'")
            if not isinstance(adversarial_config[field], field_type):
                raise TypeError(f"Adversarial field '{field}' must be of type {field_type.__name__}")
        
        # Validate generator type
        valid_generators = ["siglip_embedding", "vqa", "fgsm", "pgd", "text_overlay"]
        if adversarial_config["generator_type"] not in valid_generators:
            raise ValueError(f"Generator type must be one of {valid_generators}, got '{adversarial_config['generator_type']}'")
        
        # Validate attack mode
        valid_attack_modes = ["repulsion", "attraction", "untargeted", "targeted"]
        if adversarial_config["attack_mode"] not in valid_attack_modes:
            raise ValueError(f"Attack mode must be one of {valid_attack_modes}, got '{adversarial_config['attack_mode']}'")
        
        # Validate optional parameters
        optional_fields = {
            "epsilon": float,
            "attack_steps": int,
            "attack_step_size": float,
            "target_option": str,
            "save_adversarial_images": bool,
            "adversarial_output_dir": str,
            "force_cpu": bool,
            "batch_size": int,
            "siglip_model_name": str
        }
        
        for field, field_type in optional_fields.items():
            if field in adversarial_config and not isinstance(adversarial_config[field], field_type):
                raise TypeError(f"Adversarial field '{field}' must be of type {field_type.__name__}")
        
        # Validate epsilon range
        if "epsilon" in adversarial_config:
            epsilon = adversarial_config["epsilon"]
            if not (0.0 <= epsilon <= 1.0):
                raise ValueError(f"Epsilon must be between 0.0 and 1.0, got {epsilon}")
        
        # Validate attack steps
        if "attack_steps" in adversarial_config:
            steps = adversarial_config["attack_steps"]
            if steps <= 0:
                raise ValueError(f"Attack steps must be positive, got {steps}")

    @staticmethod
    def validate_config_consistency(config: Dict[str, Any]) -> None:
        """Validate consistency between different config sections"""
        # Check mode consistency across models
        modes = [model_config.get("mode") for model_config in config["models"].values()]
        unique_modes = set(modes)
        
        if len(unique_modes) > 1:
            # Mixed modes are allowed but we should warn
            print(f"Warning: Mixed processing modes detected: {unique_modes}")
            
        # Validate adversarial configuration consistency
        for task_name, task_config in config["tasks"].items():
            if "adversarial" in task_config and task_config["adversarial"].get("enabled", False):
                # Check if the task type supports adversarial attacks
                task_type = task_config.get("type", "")
                supported_task_types = ["vqa_rad", "sme", "omni_vqa", "erqa"]  # Add more as needed
                
                if task_type not in supported_task_types:
                    print(f"Warning: Task '{task_name}' of type '{task_type}' may not fully support adversarial attacks")
                
                # Check consistency between batch mode and adversarial attacks
                for model_name, model_config in config["models"].items():
                    if model_config.get("mode") == "batch":
                        adv_config = task_config["adversarial"]
                        if not adv_config.get("save_adversarial_images", True):
                            raise ValueError(f"Batch mode requires saving adversarial images for task '{task_name}'")

    @staticmethod
    def load_and_validate(config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Substitute environment variables
        config = ConfigParser._substitute_env_vars(config)
        
        # Validate structure
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