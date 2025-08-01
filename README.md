# Multimodal Benchmark Suite

A comprehensive benchmark suite for evaluating multimodal language models across various tasks including FS-MEVQA (SME), VQA-RAD (medical visual question answering), ERQA, and OmniMed-VQA. Features both **realtime** and **batch processing** modes with standard and adversarial evaluation capabilities.

## ✨ Key Features

### 🚀 **Dual Processing Modes**
- **Realtime Mode**: Interactive evaluation with immediate results
- **Batch Mode**: Cost-effective large-scale evaluation using Google Cloud Vertex AI

### 🤖 **Model Support**
- **Gemma** (via Ollama)
- **Gemini** (via Google API & Vertex AI Batch)
- Extensible architecture for adding new models

### 📊 **Benchmark Tasks**
- **SME** (Standard Multimodal Evaluation)
- **VQA-RAD** (Visual Question Answering for Radiology)
- **ERQA** (Embodied Reasoning Visual Question Answering)
- **OmniMed-VQA** (Comprehensive Medical VQA)

### 🛡️ **Adversarial Testing Framework**
- **Text perturbations**: synonyms, typos, negations, policy non-compliance, prompt injections
- **Image perturbations**: noise, contrast, brightness, text overlay, metadata corruption, FGSM
- **Robustness metrics**: performance degradation, consistency ratio, timing analysis

### 🔧 **Advanced Features**
- YAML-based configuration system
- Comprehensive evaluation metrics
- WandB integration for experiment tracking
- Automated dataset preparation
- Google Cloud Storage integration
- Batch job monitoring and management

## 🛠️ Installation

### Option 1: Using UV (Recommended)

1. **Install UV** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and setup**:
```bash
git clone https://github.com/sar1kumar/adv-mml-benchmark-suite.git
cd adv-mml-benchmark-suite

# Create and activate virtual environment with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Using pip

1. **Clone the repository**:
```bash
git clone https://github.com/sar1kumar/adv-mml-benchmark-suite.git
cd adv-mml-benchmark-suite
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. **Set up model access**:

**For Gemma (Ollama)**:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull gemma2:27b
```

**For Gemini (Realtime)**:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**For Gemini (Batch Mode)**:
```bash
# Install and configure gcloud CLI
gcloud auth application-default login
gcloud config set project your-gcp-project-id
```

## 📁 Dataset Preparation

Prepare datasets using the automated scripts:

```bash
# SME Dataset
python scripts/prepare_data.py --task sme --output data/sme

# VQA-RAD Dataset  
python scripts/prepare_data.py --task vqa_rad --output data/vqa_rad

# OmniMed-VQA Dataset
python scripts/prepare_data.py --task omnimed_vqa --output data/omnimed_vqa

# ERQA Dataset
python scripts/prepare_data.py --task erqa --output data/erqa
```

## 🚀 Running Evaluations

### **Realtime Mode**

Perfect for development, testing, and smaller datasets:

```bash
# Run with Ollama (Gemma)
python -m scripts.run_benchmark configs/omni_vqa_ollama.yaml

# Run with Gemini API
python -m scripts.run_benchmark configs/omni_vqa_gemini.yaml
```

### **Batch Mode** 

Ideal for large-scale evaluations with cost efficiency:

#### **Setup Google Cloud**

1. **Create GCS bucket**:
```bash
gsutil mb gs://your-benchmark-suite-bucket
```

2. **Configure batch settings** in `configs/batch/base_batch.yaml`:
```yaml
models:
  gemini-2.5-flash-batch:
    type: "api"
    mode: "batch"
    model_name: "gemini-2.5-flash-001"
    project_id: "your-gcp-project-id"
    location: "us-central1"
    
    batch_config:
      gcs_bucket: "gs://your-benchmark-suite-bucket"
      upload_images: true
      gcs_image_prefix: "gs://your-benchmark-suite-bucket/images"
      output_prefix: "gs://your-benchmark-suite-bucket/batch_outputs"
      wait_for_completion: true
      poll_interval: 60
```

3. **Run batch evaluation**:
```bash
python -m scripts.run_benchmark configs/batch/erqa_batch.yaml
```

#### **Batch Processing Workflow**

When running batch mode, the system:

1. **Creates input folders**:
   - `batch_inputs/` → Contains `request.json` and `metadata.json`
   - `batch_results/` → Contains `consolidated_results.jsonl`

2. **Uploads assets to GCS**:
   - Images → `gs://bucket/images/`
   - Request file → `gs://bucket/requests.jsonl`

3. **Submits to Vertex AI** → Processes batch job

4. **Downloads results** → Evaluates metrics automatically

### **Adversarial Evaluation**

Configure adversarial settings in your YAML:

```yaml
adversarial:
  epsilon: 0.1
  num_trials: 5
  min_semantic_similarity: 0.7
  perturbation_types:
    - synonym
    - typo  
    - negation
    - policy_non_compliance
    - prompt_injections
  image_perturbations:
    - noise
    - contrast
    - brightness
    - text_overlay
    - metadata
    - filtered
    - fgsm
```

Run adversarial benchmark:
```bash
python scripts/run_adversarial_benchmark.py configs/multi_scenario.yaml \
    --output results/adversarial \
    --num-trials 5 \
    --epsilon 0.1
```

## ⚙️ Configuration

### **Realtime Configuration Example**

```yaml
models:
  gemini:
    type: "api"
    mode: "realtime"  # Default mode
    model_name: "gemini-2.0-flash-exp"
    api_key: "${GEMINI_API_KEY}"
    max_tokens: 8192
    temperature: 0.1

tasks:
  erqa:
    type: "erqa"
    data_path: "data/erqa"
    metrics: ["accuracy", "bleu"]
    
logging:
  output_dir: "results"
  
wandb:
  enabled: true
  project: "multimodal-benchmark"
```

### **Batch Configuration Example**

```yaml
models:
  gemini-batch:
    type: "api" 
    mode: "batch"
    model_name: "gemini-2.5-flash-001"
    project_id: "your-gcp-project-id"
    location: "us-central1"
    
    batch_config:
      gcs_bucket: "gs://your-benchmark-suite-bucket"
      upload_images: true
      output_prefix: "gs://your-benchmark-suite-bucket/outputs"
      wait_for_completion: true
      poll_interval: 60

tasks:
  erqa:
    type: "erqa"
    data_path: "data/erqa"
    metrics: ["accuracy"]
```

## 📊 Results Format

### **Standard Results**
```json
{
  "model_name": {
    "task_name": {
      "accuracy": 0.85,
      "bleu": 0.73,
      "total_examples": 1000,
      "total_time": 120.5,
      "mean_inference_time": 0.12
    }
  }
}
```

### **Adversarial Results**
```json
{
  "model_name": {
    "task_name": {
      "clean_performance": {
        "accuracy": 0.85,
        "bleu": 0.83
      },
      "adversarial_performance": {
        "accuracy": {"mean": 0.75, "std": 0.02},
        "bleu": {"mean": 0.73, "std": 0.03}
      },
      "robustness_metrics": {
        "accuracy_degradation": 0.118,
        "accuracy_gap": 0.1,
        "accuracy_consistency": 0.882
      }
    }
  }
}
```

## 🏗️ Architecture

### **Core Components**

- **`benchmark_suite/core/`**
  - `orchestrator.py` → Main evaluation coordinator
  - `batch_processor.py` → Handles batch workflow
  - `model_manager.py` → Model loading and management
  - `task_manager.py` → Task orchestration
  - `config_parser.py` → YAML configuration parsing

- **`benchmark_suite/models/`**
  - `gemini_model.py` → Gemini API & Batch implementation
  - `gemma_model.py` → Ollama integration
  - `base_model.py` → Abstract model interface

- **`benchmark_suite/tasks/`**
  - Task-specific implementations for each benchmark

### **Batch Processing Architecture**
![image.png]

## 🔧 Development

### **Adding New Models**

1. Create model class in `benchmark_suite/models/`:
```python
from .base_model import BaseModel

class NewModel(BaseModel):
    def load_model(self): pass
    def generate(self, prompt, images=None): pass
    def process_image(self, image_path): pass
```

2. Register in model manager and configs

### **Adding New Tasks**

1. Create task class in `benchmark_suite/tasks/`:
```python
from .base_task import BaseTask

class NewTask(BaseTask):
    def load_data(self): pass
    def evaluate(self, model): pass
    def format_prompt(self, example): pass
    
    # For batch support
    def supports_batch_processing(self): return True
    def prepare_batch_data(self): pass
    def format_batch_request(self, example, gcs_prefix): pass
```

### **Batch Processing Support**

To enable batch processing for a new task, implement:

```python
def supports_batch_processing(self) -> bool:
    return True

def get_image_paths(self) -> List[str]:
    """Return all unique image paths for upload"""
    pass

def prepare_batch_data(self) -> List[Dict]:
    """Prepare data for batch processing"""
    pass

def format_batch_request(self, example: Dict, gcs_prefix: str) -> Dict:
    """Format single example for Gemini batch API"""
    pass

def evaluate_batch_results(self, results_file: str, metadata_file: str) -> Dict:
    """Process batch results and compute metrics"""
    pass
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

```bibtex
@article{hu2024omnimedvqa,
  title={OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM},
  author={Hu, Yutao and Li, Tianbin and Lu, Quanfeng and Shao, Wenqi and He, Junjun and Qiao, Yu and Luo, Ping},
  journal={arXiv preprint arXiv:2402.09181},
  year={2024}
}
```
```bibtex
@inproceedings{xue2024few,
  title={Few-Shot Multimodal Explanation for Visual Question Answering},
  author={Xue, Dizhan and Qian, Shengsheng and Xu, Changsheng},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  year={2024}
}
```
```bibtex
@inproceedings{OpenEQA2023,
  title={OpenEQA: Embodied Question Answering in the Era of Foundation Models}, 
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  author={Majumdar, Arjun and Ajay, Anurag and Zhang, Xiaohan and others},
  year={2024}
}
```
```bibtex
@article{Lau2018,
  journal={Scientific Data},
  author={Jason J. Lau and Soumya Gayen and Asma Ben Abacha and Dina Demner-Fushman},
  publisher={Nature Publishing Groups},
  title={A dataset of clinically generated visual questions and answers about radiology images},
  year={2018}
}
```

## 🔍 Troubleshooting

### **Common Issues**

**Batch Mode Setup**:
- Ensure `gcloud auth application-default login` is configured
- Verify GCS bucket permissions and access
- Check `project_id` matches your GCP project

**Memory Issues**:
- Use batch mode for large datasets
- Reduce batch sizes in config
- Monitor resource usage

**API Limits**:
- Implement rate limiting in realtime mode
- Use batch mode for high-throughput scenarios
- Monitor API quotas

### **Performance Tips**

- **Batch Mode**: 3-5x cost reduction for large evaluations
- **Image Optimization**: Resize images before upload to reduce costs
- **Parallel Processing**: Leverage multiple workers for data preparation
- **Caching**: Enable result caching for repeated experiments

---

🚀 **Ready to benchmark your multimodal models?** Start with realtime mode for quick tests, then scale to batch mode for comprehensive evaluations!