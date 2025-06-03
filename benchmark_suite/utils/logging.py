import logging
import os
import wandb
from typing import Dict, Any, Optional

# A basic module-level logger for any logging needed outside the class instance
# or before it's initialized.
module_logger = logging.getLogger("benchmark_suite_module")
if not module_logger.handlers:
    module_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    module_logger.addHandler(ch)

class BenchmarkLogger:
    def __init__(self, name: str = "benchmark_suite", level: int = logging.INFO, log_file: Optional[str] = None):
        """
        Initializes the BenchmarkLogger.

        Args:
            name (str): Name of the logger.
            level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
            log_file (Optional[str]): Path to a file to save logs, in addition to console.
        """
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:  # Avoid adding multiple handlers if logger already configured
            self.logger.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            # File handler (optional)
            if log_file:
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                fh = logging.FileHandler(log_file)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
        
        self._wandb_run = None
        self.wandb_enabled = False

    def init_wandb(self, config: Dict[str, Any], project_name: str = "benchmark-suite", run_name: Optional[str] = None) -> Optional[wandb.sdk.wandb_run.Run]:
        """
        Initializes a new wandb run if wandb is enabled in the config.

        Args:
            config (dict): The configuration dictionary. Expected to contain a 'wandb' section
                           with an 'enabled' key (boolean).
            project_name (str): The wandb project name.
            run_name (str, optional): The name for this specific run. Defaults to None.
        
        Returns:
            Optional[wandb.sdk.wandb_run.Run]: The wandb run object if initialized, else None.
        """
        self.wandb_enabled = config.get("wandb", {}).get("enabled", False)
        if self.wandb_enabled:
            try:
                wandb_api_key = os.environ.get("WANDB_API_KEY")
                if not wandb_api_key:
                    self.logger.warning("WANDB_API_KEY not found in environment variables. Wandb will not be initialized.")
                    self._wandb_run = None
                    self.wandb_enabled = False # Disable if key is missing
                    return None

                self._wandb_run = wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config,
                    reinit=True  # Allow reinitialization
                )
                self.logger.info(f"Wandb initialized for project '{project_name}', run '{self._wandb_run.name if self._wandb_run else 'None'}'.")
            except Exception as e:
                self.logger.error(f"Failed to initialize wandb: {e}", exc_info=True)
                self._wandb_run = None
                self.wandb_enabled = False # Disable on error
        else:
            self.logger.info("Wandb is disabled in the configuration.")
            self._wandb_run = None
        return self._wandb_run

    def log_to_wandb(self, data_dict: Dict[str, Any]):
        """
        Logs a dictionary of data to the current wandb run.

        Args:
            data_dict (dict): Data to log.
        """
        if self._wandb_run and self.wandb_enabled:
            try:
                self._wandb_run.log(data_dict)
            except Exception as e:
                self.logger.error(f"Failed to log to wandb: {e}", exc_info=True)
        # else:
            # self.debug("Wandb not initialized or disabled. Skipping wandb logging.")

    def log_model_response(self, task_name: str, model_name: str, prompt: Any, response: Any, ground_truth: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Logs a single model response.

        Args:
            task_name (str): Name of the task.
            model_name (str): Name of the model.
            prompt (any): The input prompt.
            response (any): The model's response.
            ground_truth (any, optional): The ground truth.
            metadata (dict, optional): Additional metadata.
        """
        log_entry = {
            "task": task_name,
            "model": model_name,
            "prompt": str(prompt) if not isinstance(prompt, (str, int, float, bool, list, dict)) else prompt,
            "response": str(response) if not isinstance(response, (str, int, float, bool, list, dict)) else response,
            "ground_truth": str(ground_truth) if ground_truth is not None and not isinstance(ground_truth, (str, int, float, bool, list, dict)) else ground_truth,
        }
        if metadata:
            log_entry.update(metadata)

        self.log_to_wandb({f"model_responses/{task_name}/{model_name}": log_entry})
        self.debug(f"Logged response for {task_name} with {model_name}: {log_entry}")

    def log_summary_metrics(self, task_name: str, model_name: str, metrics: Dict[str, Any]):
        """
        Logs summary metrics for a task and model.

        Args:
            task_name (str): Name of the task.
            model_name (str): Name of the model.
            metrics (dict): Dictionary of metric names to values.
        """
        if not isinstance(metrics, dict):
            self.logger.error(f"Metrics for {task_name} with {model_name} must be a dictionary. Got: {metrics}")
            return

        wandb_metrics = {}
        for metric_name, value in metrics.items():
            metric_key = f"summary_metrics/{task_name}/{model_name}/{metric_name}"
            wandb_metrics[metric_key] = value
            self.info(f"Summary Metric | Task: {task_name}, Model: {model_name} | {metric_name}: {value}")
        
        self.log_to_wandb(wandb_metrics)

    def get_wandb_run(self) -> Optional[wandb.sdk.wandb_run.Run]:
        """Returns the current wandb run object."""
        return self._wandb_run

    def end_wandb_run(self):
        """Ends the current wandb run, if active."""
        if self._wandb_run and self.wandb_enabled:
            try:
                self._wandb_run.finish()
                self.logger.info(f"Wandb run '{self._wandb_run.name}' finished.")
            except Exception as e:
                self.logger.error(f"Error finishing wandb run: {e}", exc_info=True)
            finally:
                self._wandb_run = None
                self.wandb_enabled = False # Reset status

    # Convenience methods for standard logging
    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
