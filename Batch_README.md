# Batch Processing Implementation Guide

This guide explains how to add batch processing capabilities to new datasets in the benchmark suite.

## Overview

Batch processing allows you to evaluate large datasets efficiently using Google Cloud's Vertex AI batch prediction service instead of making individual API calls.

### Benefits of Batch Processing:
- **Cost Effective**: Lower per-token costs compared to realtime API
- **Higher Throughput**: Process thousands of examples simultaneously  
- **Robust**: Built-in retry mechanisms and fault tolerance
- **Scalable**: No rate limits for batch processing

## Architecture
Config → BatchProcessor → GCS Upload → Vertex AI → Evaluation


### Components:
1. **Config File**: Defines batch processing parameters
2. **BatchProcessor**: Orchestrates the entire workflow
3. **Task Methods**: Dataset-specific logic for data preparation and evaluation
4. **GCS**: Stores input files and images
5. **Vertex AI**: Processes the batch job
6. **Evaluation**: Computes metrics from batch results

## Step-by-Step Implementation

### Step 1: Add Required Methods to Your Task Class

Your task class must inherit from `BaseTask` and implement these methods:

#### 1.1 Enable Batch Processing
```python
def supports_batch_processing(self) -> bool:
    """Enable batch processing for this task."""
    return True
```

#### 1.2 Get Image Paths
```python
def get_image_paths(self) -> List[str]:
    """Return a list of all unique local image paths required for the task."""
    if not self.data:
        self.load_data()
    
    all_image_paths = set()
    images_dir = os.path.join(self.data_path, "images")
    
    for example in self.data:
        # Extract image filename from your data structure
        image_filename = example.get("image_path")  # Adjust based on your data
        if image_filename:
            full_path = os.path.join(images_dir, image_filename)
            if os.path.exists(full_path):
                all_image_paths.add(full_path)
    
    return list(all_image_paths)
```

#### 1.3 Prepare Batch Data
```python
def prepare_batch_data(self) -> List[Dict[str, Any]]:
    """Load and structure the dataset for batch processing."""
    if not self.data:
        self.load_data()
    
    batch_data = []
    for idx, example in enumerate(self.data):
        batch_example = {
            "example_id": f"your_task_{idx}",
            "question": example.get("question"),
            "ground_truth": example.get("answer"),  # Adjust field name
            "image_path": example.get("image_path"), # Adjust field name
            "prompt": self.format_prompt(example)
        }
        batch_data.append(batch_example)
    
    return batch_data
```

#### 1.4 Format Batch Request
```python
def format_batch_request(self, example: Dict[str, Any], gcs_image_prefix: str) -> Dict[str, Any]:
    """Format a single example for the Vertex AI batch prediction API."""
    prompt = example["prompt"]
    gcs_image_path = f"{gcs_image_prefix}/{example['image_path']}"
    
    return {
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "file_data": {
                                "file_uri": gcs_image_path,
                                "mime_type": "image/jpeg"
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 1.0,
                "maxOutputTokens": 8192,
                "top_p": 0.95,
                "top_k": 64
            }
        }
    }
```

#### 1.5 Evaluate Batch Results
```python
def evaluate_batch_results(self, batch_results_path: str, metadata_path: str) -> Dict[str, Any]:
    """Process completed batch job results and compute metrics."""
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load and parse batch results
    predictions = []
    with open(batch_results_path, 'r') as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                response_text = result.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                predictions.append(response_text.strip())
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Warning: Could not parse batch result line: {line}. Error: {e}")
                predictions.append("")
    
    # Verify matching numbers
    if len(predictions) != len(metadata):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(metadata)} metadata entries")
    
    # Extract ground truth
    targets = [item["ground_truth"] for item in metadata]
    
    # Compute metrics using your existing method
    metrics = self.compute_metrics(predictions, targets)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(self.output_dir, f"your_task_batch_results_{timestamp}.json")
    os.makedirs(self.output_dir, exist_ok=True)
    
    results = {
        "metadata": {
            "model_name": "gemini-batch",
            "timestamp": datetime.now().isoformat(),
            "dataset": "YourTask",
            "total_examples": len(predictions),
            "batch_processing": True
        },
        "metrics": metrics
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return metrics
```

### Step 2: Create Batch Configuration File

Create a YAML configuration file (e.g., `configs/your_task_batch.yaml`):

```yaml
models:
  gemini-1.5-flash-batch:
    type: "api"
    mode: "batch"  # Important: must be "batch"
    model_name: "gemini-1.5-flash-001"
    project_id: "your-gcp-project-id"  # Replace with your project
    location: "us-central1"
    max_tokens: 8192
    temperature: 1.0

    batch_config:
      gcs_bucket: "gs://your-bucket"  # Replace with your bucket
      upload_images: true
      gcs_image_prefix: "gs://your-bucket/your_task/images"
      output_prefix: "gs://your-bucket/your_task/batch_outputs"
      wait_for_completion: true
      poll_interval: 60

tasks:
  your_task:
    type: "your_task"  # Must match your task name in TaskManager
    data_path: "data/your_task"
    metrics: ["accuracy"]  # Adjust based on your metrics

logging:
  output_dir: "results"
```

### Step 3: Register Your Task

Make sure your task is registered in `TaskManager`:

```python
# In benchmark_suite/core/task_manager.py
_TASK_REGISTRY = {
    "your_task": YourTaskClass,
    # ... other tasks
}
```

### Step 4: Setup and Run

#### Prerequisites:
```bash
# Install required libraries
pip install google-cloud-storage google-genai

# Authenticate with gcloud (one-time setup)
gcloud auth application-default login
gcloud config set project your-gcp-project-id
```

#### Run Batch Evaluation:
```bash
python scripts/run_benchmark.py configs/your_task_batch.yaml
```

## Common Patterns and Examples

### Multiple Images per Example
```python
def format_batch_request(self, example: Dict[str, Any], gcs_image_prefix: str) -> Dict[str, Any]:
    """Handle multiple images per example (like ERQA)."""
    prompt = example["prompt"]
    
    parts = [{"text": prompt}]
    for image_path in example["image_paths"]:  # List of images
        gcs_path = f"{gcs_image_prefix}/{os.path.basename(image_path)}"
        parts.append({
            "file_data": {
                "file_uri": gcs_path,
                "mime_type": "image/jpeg"
            }
        })
    
    return {
        "request": {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 1.0,
                "maxOutputTokens": 8192,
                "top_p": 0.95,
                "top_k": 64
            }
        }
    }
```

### Complex Data Structures
```python
def prepare_batch_data(self) -> List[Dict[str, Any]]:
    """Handle complex nested data structures (like OmniMed)."""
    if not self.data:
        self.load_data()
    
    batch_data = []
    for idx, example in enumerate(self.data):
        # Handle options for multiple choice
        options = {
            "A": example.get("option_A", ""),
            "B": example.get("option_B", ""),
            "C": example.get("option_C", ""),
            "D": example.get("option_D", "")
        }
        
        batch_example = {
            "example_id": f"task_{idx}",
            "question_type": example.get("question_type"),
            "ground_truth": example.get("gt_answer"),
            "options": options,
            "prompt": self.format_prompt(example)
        }
        batch_data.append(batch_example)
    
    return batch_data
```

### Advanced Metrics Computation
```python
def evaluate_batch_results(self, batch_results_path: str, metadata_path: str) -> Dict[str, Any]:
    """Handle per-category metrics (like OmniMed)."""
    # ... load predictions and metadata ...
    
    # Reconstruct additional data for complex metrics
    question_types = []
    datasets = []
    
    if self.data:
        for example in self.data[:len(predictions)]:
            question_types.append(example.get("question_type", "unknown"))
            datasets.append(example.get("dataset", "unknown"))
    
    # Use your existing complex compute_metrics method
    metrics = self.compute_metrics(
        predictions, 
        targets, 
        question_types, 
        datasets, 
        additional_data
    )
    
    return metrics
```

## Troubleshooting

### Common Issues:

1. **Image Upload Fails**
   - Check GCS bucket permissions
   - Verify image file paths exist locally
   - Ensure bucket name doesn't include 'gs://' prefix in some contexts

2. **Job Creation Fails**
   - Verify gcloud authentication: `gcloud auth list`
   - Check project ID is correct
   - Ensure Vertex AI is enabled: `gcloud services enable aiplatform.googleapis.com`

3. **Evaluation Mismatch**
   - Verify field names match between realtime and batch evaluation
   - Check that `ground_truth` field is consistently used
   - Ensure data ordering is preserved

4. **JobState Errors**
   - Update imports: `from google.genai.types import JobState`
   - Use correct state names: `JobState.JOB_STATE_SUCCEEDED`

### Debug Mode:
Add debug logging to trace data flow:

```python
def prepare_batch_data(self) -> List[Dict[str, Any]]:
    batch_data = self._prepare_data()
    print(f"DEBUG: Prepared {len(batch_data)} examples")
    print(f"DEBUG: First example keys: {list(batch_data[0].keys())}")
    return batch_data
```

## Performance Tips

1. **Optimize Image Uploads**: Use `transfer_manager` for parallel uploads
2. **Batch Size**: No strict limits, but 1000-10000 examples work well
3. **Polling Interval**: Use 60-300 seconds for large jobs
4. **Error Handling**: Always include try-catch blocks for JSON parsing

## Example Configurations

### ERQA Batch Config:
```yaml
# configs/erqa_batch.yaml
tasks:
  erqa:
    type: "erqa"
    data_path: "data/erqa"
    metrics: ["accuracy"]
```

### OmniMed Batch Config:
```yaml
# configs/omnimed_batch.yaml  
tasks:
  omnimed_vqa:
    type: "omnimed_vqa"
    data_path: "data/omnimed"
    access_type: "Open-access"
    sample_size: 1000
    metrics: ["accuracy"]
```

### SME Batch Config:
```yaml
# configs/sme_batch.yaml
tasks:
  sme:
    type: "sme"
    data_path: "data/sme"
    split: ["test"]
    sample_size: 500
    metrics: ["accuracy", "mean_iou"]
```

This guide provides everything needed to implement batch processing for any new dataset in the benchmark suite!