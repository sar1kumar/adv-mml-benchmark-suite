# benchmark_suite/core/batch_processor.py
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Import Google Cloud clients with proper error handling
try:
    from google.cloud.storage import Client, transfer_manager
    from google.genai.types import JobState
except ImportError:
    print("Please install Google Cloud libraries: pip install google-cloud-storage google-genai")
    Client = None
    transfer_manager = None
    JobState = None

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
        if not Client or not transfer_manager or not JobState:
            raise ImportError("Google Cloud libraries are not installed. Please run: pip install google-cloud-storage google-genai")

        self.config = config
        self.model = model
        self.batch_config = model.batch_config
        self.logger = logging.getLogger("benchmark_suite.batch")
        
        # Initialize GCS client - Fixed: use Client directly, not storage.Client
        self.gcs_client = Client(project=self.model.project_id)
        
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
        """
        Prepares batch input and metadata files locally by calling task methods.
        Returns paths to the local requests.jsonl and metadata.json files.
        """
        self.logger.info(f"Calling {task_name}.prepare_batch_data() to get examples.")
        batch_data = task.prepare_batch_data()

        if not batch_data:
            raise ValueError(f"Task {task_name} returned no data from prepare_batch_data().")

        requests = []
        metadata = []
        gcs_image_prefix = self.batch_config["gcs_image_prefix"]

        self.logger.info(f"Formatting {len(batch_data)} examples into batch requests.")
        for example in batch_data:
            final_request = task.format_batch_request(example, gcs_image_prefix)
            requests.append(final_request)

            metadata_entry = {
                "example_id": example.get("example_id"),
                "ground_truth": example.get("ground_truth"),
            }
            metadata.append(metadata_entry)

        output_dir = "batch_inputs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        requests_file = os.path.join(output_dir, f"{task_name}_requests_{timestamp}.jsonl")
        metadata_file = os.path.join(output_dir, f"{task_name}_metadata_{timestamp}.json")

        self.logger.info(f"Saving batch requests to {requests_file}")
        with open(requests_file, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + '\n')

        self.logger.info(f"Saving metadata to {metadata_file}")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return requests_file, metadata_file
    
    def _upload_to_gcs(self, local_file_path: str) -> str:
        """Uploads a single file to the configured GCS bucket."""
        bucket_name = self.batch_config["gcs_bucket"].replace("gs://", "")
        blob_name = f"batch_inputs/{os.path.basename(local_file_path)}"
        
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        self.logger.info(f"Uploading {local_file_path} to gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(local_file_path)
        
        return f"gs://{bucket_name}/{blob_name}"

    def _upload_images_to_gcs(self, task: BaseTask) -> None:
        """Uploads all images for a task to GCS using transfer_manager."""
        local_image_paths = task.get_image_paths()
        if not local_image_paths:
            self.logger.warning(f"Task {task.__class__.__name__} returned no images to upload.")
            return

        bucket_name = self.batch_config["gcs_bucket"].replace("gs://", "")
        bucket = self.gcs_client.bucket(bucket_name)
        gcs_prefix = self.batch_config["gcs_image_prefix"].replace(f"gs://{bucket_name}/", "")

        # Prepare list of files to upload (skip existing ones)
        files_to_upload = []
        for local_path in local_image_paths:
            blob_name = f"{gcs_prefix}/{os.path.basename(local_path)}"
            if not bucket.blob(blob_name).exists():
                files_to_upload.append((local_path, blob_name))

        if not files_to_upload:
            self.logger.info("All task images already exist in GCS. Skipping upload.")
            return

        self.logger.info(f"Uploading {len(files_to_upload)} new images to gs://{bucket_name}/{gcs_prefix}...")
        
        # Use transfer_manager for efficient batch upload
        source_files = [pair[0] for pair in files_to_upload]
        blob_names = [pair[1] for pair in files_to_upload]
        
        results = transfer_manager.upload_many_from_filenames(
            bucket, 
            blob_names, 
            source_directory="",  # We provide full paths
            source_filenames=source_files,
            max_workers=8
        )

        for (source_file, blob_name), result in zip(files_to_upload, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to upload {source_file} due to: {result}")
            else:
                self.logger.debug(f"Successfully uploaded {source_file} to {blob_name}")
        
        self.logger.info("Image upload complete.")

    def _monitor_batch_job(self, job_name: str) -> bool:
        """Monitors a Vertex AI batch job until completion."""
        poll_interval = self.batch_config.get("poll_interval", 60)
        
        # Define completed states using correct JobState enum values
        completed_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }
        
        self.logger.info("Starting job monitoring...")
        while True:
            # Get the current job status
            job = self.model.get_batch_job_status(job_name)
            job_state = job.state
            self.logger.info(f"Job '{job.name}' state: {job_state.name}")
            
            # Check if job has completed
            if job_state in completed_states:
                if job_state == JobState.JOB_STATE_SUCCEEDED:
                    self.logger.info("✅ Batch job completed successfully!")
                    return True
                elif job_state == JobState.JOB_STATE_FAILED:
                    self.logger.error("❌ Batch job failed!")
                    return False
                elif job_state == JobState.JOB_STATE_CANCELLED:
                    self.logger.error("⏹️ Batch job was cancelled!")
                    return False
                elif job_state == JobState.JOB_STATE_PAUSED:
                    self.logger.warning("⏸️ Batch job is paused!")
                    return False
                    
            # Wait before checking again
            self.logger.info(f"⏳ Job still running. Waiting {poll_interval} seconds before next check...")
            time.sleep(poll_interval)

    def _process_batch_results(self, output_gcs_path: str, metadata_path: str, task: BaseTask) -> Dict[str, Any]:
        """Downloads results from GCS and triggers task evaluation."""
        bucket_name, prefix = output_gcs_path.replace("gs://", "").split("/", 1)
        bucket = self.gcs_client.bucket(bucket_name)
        
        timestamp = int(time.time())
        local_results_dir = Path(f"batch_results/temp_{timestamp}")
        local_results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Downloading results from {output_gcs_path} to {local_results_dir}")
        
        blobs = self.gcs_client.list_blobs(bucket, prefix=prefix)
        downloaded_files = [blob for blob in blobs if blob.name.endswith(".jsonl")]

        if not downloaded_files:
            raise FileNotFoundError(f"No '.jsonl' result files found in {output_gcs_path}")

        consolidated_results_path = local_results_dir / "consolidated_results.jsonl"
        with open(consolidated_results_path, 'w') as outfile:
            for blob in downloaded_files:
                self.logger.debug(f"Downloading and consolidating {blob.name}")
                content = blob.download_as_text()
                outfile.write(content)

        self.logger.info(f"Consolidated results saved to {consolidated_results_path}")

        return task.evaluate_batch_results(str(consolidated_results_path), metadata_path)