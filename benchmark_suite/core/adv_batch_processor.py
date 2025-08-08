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
from benchmark_suite.core.adversarial_manager import AdversarialManager


class AdvBatchProcessor:
    """
    Handles the entire batch processing workflow with adversarial support:
    1. Generates adversarial images if enabled
    2. Uploads adversarial images to GCS
    3. Prepares and uploads input files with adversarial image URIs
    4. Creates and monitors Vertex AI Batch Jobs
    5. Downloads and processes results for evaluation
    """
    def __init__(self, config: Dict[str, Any], model, adversarial_manager: AdversarialManager):
        if not Client or not transfer_manager or not JobState:
            raise ImportError("Google Cloud libraries are not installed. Please run: pip install google-cloud-storage google-genai")

        self.config = config
        self.model = model
        self.batch_config = model.batch_config
        self.adversarial_manager = adversarial_manager
        self.logger = logging.getLogger("benchmark_suite.adv_batch")
        
        # Initialize GCS client
        self.gcs_client = Client(project=self.model.project_id)
        
    def evaluate_task(self, task_name: str, task: BaseTask) -> Dict[str, Any]:
        """Runs the complete batch evaluation for a single task with adversarial support."""
        try:
            self.logger.info(f"Starting adversarial batch evaluation for task: {task_name}")

            # Step 1: Generate adversarial images
            if self.adversarial_manager.has_adversarial_task(task_name):
                self.logger.info(f"Generating adversarial images for task: {task_name}")
                original_images = task.get_image_paths()
                
                # Get adversarial output directory from config
                task_config = self.config["tasks"][task_name]
                adv_output_dir = task_config["adversarial"]["adversarial_output_dir"]
                
                # Generate adversarial images (saves to adversarial_output_dir)
                self.adversarial_manager.generate_adversarial_images(
                    task_name, original_images, adv_output_dir
                )
                
                # Update task to use adversarial images
                if hasattr(task, '_update_image_paths'):
                    # Create a simple mapping for the task to know to use adversarial paths
                    dummy_mapping = {path: path for path in original_images}  # Task will handle the actual mapping
                    task._update_image_paths(dummy_mapping)

            # Step 2: Upload adversarial images to GCS
            task_config = self.config["tasks"][task_name]
            if "adversarial" in task_config and task_config["adversarial"].get("enabled", False):
                adv_output_dir = task_config["adversarial"]["adversarial_output_dir"]
                uploaded_images = self._upload_directory_to_gcs(adv_output_dir)
                adversarial_enabled = True
            else:
                # Upload original images if no adversarial images
                original_images = task.get_image_paths()
                uploaded_images = self._upload_images_to_gcs(original_images)
                adversarial_enabled = False

            # Step 3: Prepare batch input files
            self.logger.info("Preparing adversarial batch input files.")
            local_requests_file, local_metadata_file = self._prepare_local_inputs(task_name, task)
            
            # Step 4: Upload batch files to GCS
            output_prefix = self.batch_config['output_prefix']
            self.logger.info(f"Uploading batch files to: {output_prefix}/batch_inputs/")
            
            gcs_requests_path = self._upload_to_gcs(local_requests_file)
            
            # Step 5: Create and submit batch job
            self.logger.info("Creating adversarial batch prediction job.")
            job_name = self.model.create_batch_job(gcs_requests_path, output_prefix)
            self.logger.info(f"Adversarial batch job created: {job_name}")
            
            # Step 6: Monitor the job and process results
            if self.batch_config.get("wait_for_completion", True):
                self.logger.info(f"Monitoring adversarial batch job: {job_name}")
                job_succeeded = self._monitor_batch_job(job_name)
                
                if job_succeeded:
                    self.logger.info("Adversarial batch job completed successfully.")
                    results = self._process_batch_results(output_prefix, local_metadata_file, task)
                    
                    # Add simple adversarial metadata
                    results["adversarial"] = {
                        "enabled": adversarial_enabled,
                        "uploaded_images": len(uploaded_images)
                    }
                    
                    return results
                else:
                    self.logger.error("Adversarial batch job failed.")
                    return {"error": "Adversarial batch job did not succeed."}
            else:
                self.logger.info("Adversarial batch job submitted. Not waiting for completion.")
                return {
                    "status": "adversarial_batch_job_submitted",
                    "job_name": job_name,
                    "output_path": output_prefix,
                    "adversarial_images": len(uploaded_images)
                }

        except Exception as e:
            self.logger.error(f"Error during adversarial batch evaluation for {task_name}: {e}", exc_info=True)
            return {"error": str(e)}

    def _prepare_local_inputs(self, task_name: str, task: BaseTask) -> Tuple[str, str]:
        """
        Prepares adversarial batch input and metadata files locally.
        Returns paths to the local requests.jsonl and metadata.json files.
        """
        self.logger.info(f"Calling {task_name}.prepare_batch_data() to get adversarial examples.")
        batch_data = task.prepare_batch_data()

        if not batch_data:
            raise ValueError(f"Task {task_name} returned no data from prepare_batch_data().")

        requests = []
        metadata = []
        gcs_image_prefix = self.batch_config["gcs_image_prefix"]

        self.logger.info(f"Formatting {len(batch_data)} adversarial examples into batch requests.")
        for example in batch_data:
            final_request = task.format_batch_request(example, gcs_image_prefix)
            requests.append(final_request)

            metadata_entry = {
                "example_id": example.get("example_id"),
                "ground_truth": example.get("ground_truth"),
                "question_type": example.get("question_type"),
                "adversarial": True,
                "original_data": example
            }
            metadata.append(metadata_entry)

        # Use output_dir from config for adversarial batch inputs
        output_dir = self.config.get("output_dir", "results")
        output_dir = os.path.join(output_dir, f"adv_batch_inputs_{task_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        requests_file = os.path.join(output_dir, f"{task_name}_adv_requests_{timestamp}.jsonl")
        metadata_file = os.path.join(output_dir, f"{task_name}_adv_metadata_{timestamp}.json")

        self.logger.info(f"Saving adversarial batch requests to {requests_file}")
        with open(requests_file, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + '\n')

        self.logger.info(f"Saving adversarial metadata to {metadata_file}")
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

    def _upload_directory_to_gcs(self, local_directory: str) -> List[str]:
        """Uploads all image files from a directory to GCS using transfer_manager."""
        if not os.path.exists(local_directory):
            self.logger.warning(f"Adversarial directory does not exist: {local_directory}")
            return []

        # Find all image files in the directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        image_files = []
        
        for filename in os.listdir(local_directory):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                full_path = os.path.join(local_directory, filename)
                if os.path.isfile(full_path):
                    image_files.append(full_path)

        if not image_files:
            self.logger.warning(f"No image files found in adversarial directory: {local_directory}")
            return []

        self.logger.info(f"Found {len(image_files)} image files in {local_directory}")

        bucket_name = self.batch_config["gcs_bucket"].replace("gs://", "")
        bucket = self.gcs_client.bucket(bucket_name)
        gcs_prefix = self.batch_config["gcs_image_prefix"].replace(f"gs://{bucket_name}/", "")

        # Prepare list of files to upload
        files_to_upload = []
        for local_path in image_files:
            blob_name = f"{gcs_prefix}/{os.path.basename(local_path)}"
            files_to_upload.append((local_path, blob_name))

        self.logger.info(f"Uploading {len(files_to_upload)} adversarial images to gs://{bucket_name}/{gcs_prefix}...")
        
        # Use transfer_manager for efficient batch upload
        source_files = [pair[0] for pair in files_to_upload]
        blob_names = [pair[1] for pair in files_to_upload]
        
        results = transfer_manager.upload_many_from_filenames(
            bucket, 
            source_files,
            max_workers=8
        )

        uploaded_files = []
        for (source_file, blob_name), result in zip(files_to_upload, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to upload {source_file}: {result}")
            else:
                self.logger.debug(f"Successfully uploaded {source_file} to {blob_name}")
                uploaded_files.append(f"gs://{bucket_name}/{blob_name}")
        
        self.logger.info(f"Successfully uploaded {len(uploaded_files)} adversarial images from directory")
        return uploaded_files

    def _upload_images_to_gcs(self, image_paths: List[str]) -> List[str]:
        """Uploads regular images to GCS using transfer_manager."""
        if not image_paths:
            self.logger.warning("No images to upload")
            return []

        bucket_name = self.batch_config["gcs_bucket"].replace("gs://", "")
        bucket = self.gcs_client.bucket(bucket_name)
        gcs_prefix = self.batch_config["gcs_image_prefix"].replace(f"gs://{bucket_name}/", "")

        # Prepare list of files to upload (skip existing ones)
        files_to_upload = []
        for local_path in image_paths:
            if os.path.exists(local_path):
                blob_name = f"{gcs_prefix}/{os.path.basename(local_path)}"
                if not bucket.blob(blob_name).exists():
                    files_to_upload.append((local_path, blob_name))

        if not files_to_upload:
            self.logger.info("All images already exist in GCS. Skipping upload.")
            return []

        self.logger.info(f"Uploading {len(files_to_upload)} new images to gs://{bucket_name}/{gcs_prefix}...")
        
        # Use transfer_manager for efficient batch upload
        source_files = [pair[0] for pair in files_to_upload]
        
        results = transfer_manager.upload_many_from_filenames(
            bucket, 
            source_files,
            max_workers=8
        )

        uploaded_files = []
        for (source_file, blob_name), result in zip(files_to_upload, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to upload image {source_file}: {result}")
            else:
                self.logger.debug(f"Successfully uploaded image {source_file} to {blob_name}")
                uploaded_files.append(f"gs://{bucket_name}/{blob_name}")
        
        self.logger.info(f"Image upload complete. Uploaded {len(uploaded_files)} files.")
        return uploaded_files

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
        
        self.logger.info("Starting adversarial job monitoring...")
        while True:
            # Get the current job status
            job = self.model.get_batch_job_status(job_name)
            job_state = job.state
            self.logger.info(f"Adversarial job '{job.name}' state: {job_state.name}")
            
            # Check if job has completed
            if job_state in completed_states:
                if job_state == JobState.JOB_STATE_SUCCEEDED:
                    self.logger.info("✅ Adversarial batch job completed successfully!")
                    return True
                elif job_state == JobState.JOB_STATE_FAILED:
                    self.logger.error("❌ Adversarial batch job failed!")
                    return False
                elif job_state == JobState.JOB_STATE_CANCELLED:
                    self.logger.error("⏹️ Adversarial batch job was cancelled!")
                    return False
                elif job_state == JobState.JOB_STATE_PAUSED:
                    self.logger.warning("⏸️ Adversarial batch job is paused!")
                    return False
                    
            # Wait before checking again
            self.logger.info(f"⏳ Adversarial job still running. Waiting {poll_interval} seconds...")
            time.sleep(poll_interval)

    def _process_batch_results(self, output_gcs_path: str, metadata_path: str, task: BaseTask) -> Dict[str, Any]:
        """Downloads adversarial results from GCS and triggers task evaluation."""
        bucket_name, prefix = output_gcs_path.replace("gs://", "").split("/", 1)
        bucket = self.gcs_client.bucket(bucket_name)
        
        timestamp = int(time.time())
        local_results_dir = Path(f"adv_batch_results/temp_{timestamp}")
        local_results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Downloading adversarial results from {output_gcs_path} to {local_results_dir}")
        
        blobs = self.gcs_client.list_blobs(bucket, prefix=prefix)
        downloaded_files = [blob for blob in blobs if blob.name.endswith(".jsonl")]

        if not downloaded_files:
            raise FileNotFoundError(f"No '.jsonl' result files found in {output_gcs_path}")

        consolidated_results_path = local_results_dir / "consolidated_adversarial_results.jsonl"
        with open(consolidated_results_path, 'w') as outfile:
            for blob in downloaded_files:
                self.logger.debug(f"Downloading and consolidating adversarial result {blob.name}")
                content = blob.download_as_text()
                outfile.write(content)

        self.logger.info(f"Consolidated adversarial results saved to {consolidated_results_path}")

        return task.evaluate_batch_results(str(consolidated_results_path), metadata_path)