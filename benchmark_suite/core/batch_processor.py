# benchmark_suite/core/batch_processor.py
import os
import json
import time
import logging
from typing import Dict, Any, Tuple

try:
    from google.cloud.storage import Client, transfer_manager
    from google.cloud.aiplatform.compat.types import job_state as gca_job_state
except ImportError:
    print("Please install Google Cloud libraries: pip install google-cloud-storage google-cloud-aiplatform")
    Client = None
    transfer_manager = None
    gca_job_state = None
from benchmark_suite.tasks.base_task import BaseTask


class BatchProcessor:
    """
    Handles the entire batch processing workflow:
    1. Prepares and uploads input files.
    2. Manages image assets on GCS.
    3. Creates and monitors Vertex AI Batch Jobs.
    4. Downloads and processes results for evaluation.
    """
    def __init__(self, config: Dict[str, Any], model):
        if not storage or not gca_job_state:
            raise ImportError("Google Cloud libraries are not installed.")

        self.config = config
        self.model = model
        self.batch_config = model.batch_config
        self.logger = logging.getLogger("benchmark_suite.batch")
        
        # Initialize GCS client
        self.gcs_client = storage.Client(project=self.model.project_id)
        
    def evaluate_task(self, task_name: str, task: BaseTask) -> Dict[str, Any]:
        """Runs the complete batch evaluation for a single task."""
        try:
            self.logger.info(f"Starting batch evaluation for task: {task_name}")

            # Determine GCS paths for input files
            gcs_requests_path = self.batch_config.get("gcs_requests_file")
            gcs_metadata_path = self.batch_config.get("gcs_metadata_file")

            # Step 1: Prepare batch input files if not already provided on GCS
            if not gcs_requests_path or not gcs_metadata_path:
                self.logger.info("Preparing local batch input files.")
                local_requests_file, local_metadata_file = self._prepare_local_inputs(task_name, task)
                
                self.logger.info(f"Uploading batch requests file: {local_requests_file}")
                gcs_requests_path = self._upload_to_gcs(local_requests_file)
                # We keep metadata path local, as it's only needed for final evaluation.
                gcs_metadata_path = local_metadata_file
            else:
                self.logger.info("Using pre-existing batch files from GCS.")

            # Step 2: Upload images if configured to do so
            if self.batch_config.get("upload_images", False):
                self.logger.info("Uploading images to GCS.")
                self._upload_images_to_gcs(task)
            else:
                self.logger.info("Skipping image upload, assuming they exist in GCS.")
            
            # Step 3: Create and run the batch job
            self.logger.info(f"Submitting batch job for model {self.model.model_name}.")
            output_gcs_path = f"{self.batch_config['output_prefix']}/{task_name}_{int(time.time())}"
            job_name = self.model.create_batch_job(gcs_requests_path, output_gcs_path)
            
            # Step 4: Monitor the job and process results
            if self.batch_config.get("wait_for_completion", True):
                self.logger.info(f"Monitoring batch job: {job_name}")
                job_succeeded = self._monitor_batch_job(job_name)
                
                if job_succeeded:
                    self.logger.info("Batch job succeeded. Processing results.")
                    return self._process_batch_results(output_gcs_path, gcs_metadata_path, task)
                else:
                    self.logger.error("Batch job failed or was cancelled.")
                    return {"error": "Batch job did not succeed."}
            else:
                self.logger.info("Batch job submitted. Not waiting for completion.")
                return {
                    "status": "batch_job_submitted",
                    "job_name": job_name,
                    "output_path": output_gcs_path
                }

        except Exception as e:
            self.logger.error(f"Error during batch evaluation for {task_name}: {e}", exc_info=True)
            return {"error": str(e)}

    def _prepare_local_inputs(self, task_name: str, task: BaseTask) -> Tuple[str, str]:
        """Prepares batch input and metadata files locally."""
        # This will be implemented in the next step by calling task methods.
        raise NotImplementedError
    
    def _upload_to_gcs(self, local_file_path: str) -> str:
        """Uploads a single file to the configured GCS bucket."""
        # This will be implemented in the next step.
        raise NotImplementedError

    def _upload_images_to_gcs(self, task: BaseTask) -> None:
        """Uploads all images for a task to GCS."""
        # This will be implemented in the next step.
        raise NotImplementedError

    def _monitor_batch_job(self, job_name: str) -> bool:
        """Monitors a Vertex AI batch job until completion."""
        # This will be implemented in the next step.
        raise NotImplementedError

    def _process_batch_results(self, output_gcs_path: str, metadata_path: str, task: BaseTask) -> Dict[str, Any]:
        """Downloads results from GCS and triggers task evaluation."""
        # This will be implemented in the next step.
        raise NotImplementedError