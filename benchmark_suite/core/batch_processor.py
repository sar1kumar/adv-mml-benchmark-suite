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
from benchmark_suite.core.adversarial_manager import AdversarialManager


class BatchProcessor:
    """
    Handles the entire batch processing workflow with adversarial support:
    1. Generates adversarial images if enabled
    2. Prepares and uploads input files
    3. Manages image assets on GCS
    4. Creates and monitors Vertex AI Batch Jobs
    5. Downloads and processes results for evaluation
    """
    def __init__(self, config: Dict[str, Any], model, adversarial_manager: AdversarialManager = None):
        if not Client or not transfer_manager or not JobState:
            raise ImportError("Google Cloud libraries are not installed. Please run: pip install google-cloud-storage google-genai")

        self.config = config
        self.model = model
        self.batch_config = model.batch_config
        self.adversarial_manager = adversarial_manager
        self.logger = logging.getLogger("benchmark_suite.batch")
        
        # Initialize GCS client
        self.gcs_client = Client(project=self.model.project_id)
        
    def evaluate_task(self, task_name: str, task: BaseTask) -> Dict[str, Any]:
        """Runs the complete batch evaluation for a single task with adversarial support."""
        try:
            self.logger.info(f"Starting batch evaluation for task: {task_name}")

            # Step 1: Handle adversarial image generation if enabled
            adversarial_mapping = {}
            if self.adversarial_manager and self.adversarial_manager.has_adversarial_task(task_name):
                self.logger.info(f"Generating adversarial images for task: {task_name}")
                original_images = task.get_image_paths()
                adversarial_mapping = self.adversarial_manager.generate_adversarial_images(
                    task_name, original_images
                )
                
                # Update task to use adversarial images
                if hasattr(task, '_update_image_paths'):
                    task._update_image_paths(adversarial_mapping)

            # Determine GCS paths for input files
            gcs_requests_path = self.batch_config.get("gcs_requests_file")
            gcs_metadata_path = self.batch_config.get("gcs_metadata_file")

            # Step 2: Prepare batch input files if not already provided on GCS
            if not gcs_requests_path or not gcs_metadata_path:
                self.logger.info("Preparing local batch input files.")
                local_requests_file, local_metadata_file = self._prepare_local_inputs(task_name, task)
                
                self.logger.info(f"Uploading batch requests file: {local_requests_file}")
                gcs_requests_path = self._upload_to_gcs(local_requests_file)
                gcs_metadata_path = local_metadata_file
            else:
                self.logger.info(f"Using pre-existing GCS requests file: {gcs_requests_path}")

            # Step 3: Upload images to GCS (including adversarial images)
            if self.batch_config.get("upload_images", True):
                self.logger.info("Uploading task images to GCS.")
                
                # Get all image paths (now includes adversarial images if generated)
                all_image_paths = task.get_image_paths()
                self._upload_images_to_gcs(all_image_paths)
            
            # Step 4: Create and submit batch job
            self.logger.info("Creating batch prediction job.")
            job = self.model.create_batch_job(gcs_requests_path)
            self.logger.info(f"Batch job created: {job.name}")
            
            # Step 5: Wait for completion if configured
            if self.batch_config.get("wait_for_completion", True):
                self.logger.info("Waiting for batch job completion.")
                job = self._wait_for_job_completion(job)
                
                if job.state == JobState.JOB_STATE_SUCCEEDED:
                    self.logger.info("Batch job completed successfully.")
                    
                    # Step 6: Download and process results
                    local_results_path = self._download_batch_results(job)
                    results = task.evaluate_batch_results(local_results_path, gcs_metadata_path)
                    
                    # Add adversarial metadata
                    if adversarial_mapping:
                        results["adversarial"] = {
                            "enabled": True,
                            "total_images": len(adversarial_mapping),
                            "adversarial_images": len([p for orig, adv in adversarial_mapping.items() if orig != adv]),
                            "attack_success_rate": len([p for orig, adv in adversarial_mapping.items() if orig != adv]) / len(adversarial_mapping)
                        }
                    
                    # Step 7: Cleanup if configured
                    if self.batch_config.get("cleanup_after_completion", False):
                        self.logger.info("Cleaning up batch job resources.")
                        self._cleanup_batch_resources(job, gcs_requests_path)
                        
                        # Cleanup adversarial images if requested
                        if adversarial_mapping and self.config["tasks"][task_name]["adversarial"].get("cleanup_after_completion", False):
                            self.adversarial_manager.cleanup_adversarial_files(
                                task_name, list(adversarial_mapping.values())
                            )
                    
                    return results
                    
                else:
                    self.logger.error(f"Batch job failed with state: {job.state}")
                    return {"error": f"Batch job failed with state: {job.state}"}
            else:
                self.logger.info(f"Batch job submitted. Job name: {job.name}")
                return {"job_name": job.name, "status": "submitted"}

        except Exception as e:
            self.logger.error(f"Error in batch evaluation for task {task_name}: {str(e)}")
            return {"error": str(e)}

    def _prepare_local_inputs(self, task_name: str, task: BaseTask) -> Tuple[str, str]:
        """Prepares local JSONL request file and metadata for batch processing"""
        
        # Prepare batch data from the task
        batch_data = task.prepare_batch_data()

        # Create output directory
        output_dir = f"batch_inputs_{task_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create requests file
        requests_file = os.path.join(output_dir, f"{task_name}_requests.jsonl")
        metadata_file = os.path.join(output_dir, f"{task_name}_metadata.json")

        # Write batch requests
        metadata = []
        with open(requests_file, 'w') as f:
            for i, example in enumerate(batch_data):
                # Format the request for Vertex AI batch API
                request = task.format_batch_request(example, self.batch_config["gcs_image_prefix"])
                f.write(json.dumps(request) + '\n')

                # Store metadata for result processing
                metadata.append({
                    "index": i,
                    "example_id": example.get("id", i),
                    "original_data": example
                })
        
        # Write metadata file
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Created {len(batch_data)} batch requests")
        return requests_file, metadata_file
    
    def _upload_to_gcs(self, local_file_path: str) -> str:
        """Upload a local file to GCS and return the GCS URI"""
        bucket_name = self.batch_config["gcs_bucket"].replace("gs://", "")
        blob_name = f"batch_inputs/{os.path.basename(local_file_path)}"
        
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(local_file_path)
        gcs_uri = f"gs://{bucket_name}/{blob_name}"

        self.logger.info(f"Uploaded {local_file_path} to {gcs_uri}")
        return gcs_uri

    def _upload_images_to_gcs(self, image_paths: List[str]) -> None:
        """Upload multiple images to GCS efficiently"""
        if not image_paths:
            self.logger.warning("No images to upload")
            return

        bucket_name = self.batch_config["gcs_bucket"].replace("gs://", "")
        image_prefix = self.batch_config["gcs_image_prefix"].replace(f"gs://{bucket_name}/", "")
        
        bucket = self.gcs_client.bucket(bucket_name)

        # Prepare upload list
        upload_tasks = []
        for image_path in image_paths:
            if os.path.exists(image_path):
                blob_name = f"{image_prefix}{os.path.basename(image_path)}"
                upload_tasks.append((image_path, blob_name))
            else:
                self.logger.warning(f"Image not found: {image_path}")
        
        # Use transfer manager for efficient parallel uploads
        results = transfer_manager.upload_many_from_filenames(
            bucket, 
            [task[0] for task in upload_tasks],  # source filenames
            [task[1] for task in upload_tasks],  # destination blob names
            max_workers=8
        )

        successful_uploads = 0
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Failed to upload image: {result}")
            else:
                successful_uploads += 1
        
        self.logger.info(f"Successfully uploaded {successful_uploads}/{len(upload_tasks)} images to GCS")

    def _wait_for_job_completion(self, job, poll_interval: int = None):
        """Wait for batch job to complete with polling"""
        if poll_interval is None:
            poll_interval = self.batch_config.get("poll_interval", 60)
        
        while job.state in [JobState.JOB_STATE_PENDING, JobState.JOB_STATE_RUNNING]:
            self.logger.info(f"Job state: {job.state}. Waiting {poll_interval} seconds...")
            time.sleep(poll_interval)
            job = self.model.get_batch_job(job.name)
        
        return job

    def _download_batch_results(self, job) -> str:
        """Download batch job results from GCS"""
        # Extract GCS path from job output
        output_uri = job.output_info.gcs_output_directory
        
        # Download results file
        bucket_name = output_uri.replace("gs://", "").split("/")[0]
        blob_prefix = "/".join(output_uri.replace("gs://", "").split("/")[1:])
        
        bucket = self.gcs_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=blob_prefix)
        
        # Find the results file (usually predictions.jsonl)
        results_blob = None
        for blob in blobs:
            if blob.name.endswith('.jsonl') or 'predictions' in blob.name:
                results_blob = blob
                break
        
        if not results_blob:
            raise FileNotFoundError("Could not find results file in batch output")
        
        # Download to local file
        local_results_path = f"batch_results_{int(time.time())}.jsonl"
        results_blob.download_to_filename(local_results_path)
        
        self.logger.info(f"Downloaded batch results to {local_results_path}")
        return local_results_path

    def _cleanup_batch_resources(self, job, gcs_requests_path: str):
        """Clean up GCS resources after batch completion"""
        try:
            # Delete the requests file
            bucket_name = gcs_requests_path.replace("gs://", "").split("/")[0]
            blob_name = "/".join(gcs_requests_path.replace("gs://", "").split("/")[1:])
            
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.delete()

            self.logger.info(f"Cleaned up requests file: {gcs_requests_path}")

        except Exception as e:
            self.logger.warning(f"Failed to cleanup batch resources: {e}")