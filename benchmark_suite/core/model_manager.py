import importlib
from typing import Dict, Any, Type
from benchmark_suite.models.base_model import BaseModel
from benchmark_suite.models.gemma_model import GemmaModel
from benchmark_suite.models.gemini_model import GeminiModel

class ModelManager:
    """Manages loading and initialization of models"""
    
    _MODEL_REGISTRY = {
        "ollama": GemmaModel,
        "api": GeminiModel
    }
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseModel]) -> None:
        """Register a new model type"""
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Model class must inherit from BaseModel")
        cls._MODEL_REGISTRY[model_type] = model_class
        
    @classmethod
    def get_model_class(cls, model_type: str) -> Type[BaseModel]:
        """Get the model class for a given type"""
        if model_type not in cls._MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._MODEL_REGISTRY[model_type]
        
    @classmethod
    def load_model(cls, model_config: Dict[str, Any]) -> BaseModel:
        """Load and initialize a model from config"""
        model_type = model_config.get("type", "").lower()
        model_class = cls.get_model_class(model_type)
        
        try:
            model = model_class(model_config)
            model.load_model()  # Initialize the model
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model of type {model_type}: {str(e)}")
            
    @classmethod
    def load_models(cls, config: Dict[str, Any]) -> Dict[str, BaseModel]:
        """Load all models specified in config"""
        models = {}
        for model_name, model_config in config["models"].items():
            try:
                models[model_name] = cls.load_model(model_config)
            except Exception as e:
                raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")
        return models
        
    @classmethod
    def load_external_model(cls, module_path: str, class_name: str) -> Type[BaseModel]:
        """Dynamically load an external model class"""
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            if not issubclass(model_class, BaseModel):
                raise TypeError(f"Model class {class_name} must inherit from BaseModel")
                
            return model_class
        except ImportError:
            raise ImportError(f"Failed to import model from {module_path}")
        except AttributeError:
            raise AttributeError(f"Model class {class_name} not found in {module_path}")
            
    @classmethod
    def register_external_model(cls, model_type: str, module_path: str, class_name: str) -> None:
        """Register an external model type"""
        model_class = cls.load_external_model(module_path, class_name)
        cls.register_model(model_type, model_class) 