# Multimodal Benchmark Suite

A comprehensive benchmark suite for evaluating multimodal language models across various tasks including FS-MEVQA (SME), VQA-RAD (medical visual question answering), and ERQA. Features both standard and adversarial evaluation capabilities.

## Features

- Support for multiple models:
  - Gemma (via Ollama)
  - Gemini (via Google API)
- Multiple benchmark tasks:
  - SME (Standard Multimodal Evaluation)
  - VQA-RAD (visual question answering for radiology)
  - ERQA (embodied reasoning visual question answering)
- Comprehensive evaluation metrics
  - Adversarial testing framework:
    - Text perturbations (synonyms, typos, negations)
    - Image perturbations (noise, contrast, brightness, text overlay, metadata corruption, filtered images, FGSM)
    - Robustness metrics
- Configurable via YAML files
- Dataset Sampling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sar1kumar/adv-mml-benchmark-suite.git
cd adv-mml-benchmark-suite
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up model access:
   - For Gemma: Install [Ollama](https://ollama.ai/) and pull the model:
     ```bash
     ollama pull gemma3:27b
     ```
   - For Gemini: Set your API key as an environment variable:
     ```bash
     export GEMINI_API_KEY="your-api-key-here"
     ```

## Dataset Preparation

1. [SME](https://dl.acm.org/doi/10.1145/3664647.3681597):
```bash
python scripts/prepare_data.py --task sme --output data/sme
```

2. [VQA-RAD](https://www.nature.com/articles/sdata2018251.pdf):
```bash
python scripts/prepare_data.py --task vqa_rad --output data/vqa_rad
```
3. [OmniMed-VQA](https://arxiv.org/pdf/2402.09181v2):
```bash
python scripts/prepare_data.py --task omnimed_vqa --output data/omnimed_vqa
```

4. [ERQA](https://github.com/embodiedreasoning/ERQA):
```bash
python scripts/prepare_data.py --task erqa --output data/erqa
```

## Running Benchmarks

### Standard Evaluation

1. Configure your evaluation in a YAML file (see `configs/` for examples)

2. Run the benchmark:
```bash
python -m scripts.run_benchmark configs/omni_vqa_ollama.yaml
```

### Adversarial Evaluation

1. Configure adversarial settings in your YAML file:

```yaml
adversarial:
  epsilon: 0.1  # Perturbation magnitude
  num_trials: 5  # Number of adversarial trials
  perturbation_types:  # For text tasks
    - synonym
    - typo
    - negation
    - policy - non compilance
    - soft token insertions
    - many shot - think less attacks
    - prompt injections
    - LMP misuse attacks
  image_perturbations:  # For visual tasks
    - noise
    - contrast
    - brightness
    - text_overlay
    - metadata
    - filtered
    - fgsm
```

2. Run adversarial benchmark:
```bash
python scripts/run_adversarial_benchmark.py configs/multi_scenario.yaml \
    --output results/adversarial \
    --num-trials 5 \
    --epsilon 0.1
```

The adversarial evaluation includes:
- Text perturbations:
  - Synonym replacement using WordNet
  - Controlled typo introduction
  - Negation addition/removal
  - Image perturbations (for VQA tasks):
    - Gaussian noise injection
    - Contrast modification
    - Brightness adjustment
    - Text overlay or occlusion
    - Metadata (EXIF) corruption
    - Filtered/manipulated images
    - Gradient-based attack (FGSM)
- Robustness metrics:
  - Performance degradation
  - Absolute performance gap
  - Consistency ratio
  - Mean inference time

## Configuration

Example configuration file (`configs/multi_scenario.yaml`):

```yaml
models:
  gemma:
    type: "ollama"
    model_name: "gemma:7b"
    max_tokens: 2048
    temperature: 0.1
    
  gemini:
    type: "api"
    model_name: "gemini-pro"
    api_key: "${GEMINI_API_KEY}"
    max_tokens: 2048
    temperature: 0.1

tasks:
  sme:
    type: "sme"
    data_path: "data/sme"
    few_shot_k: 5
    metrics: ["accuracy", "f1"]
    
  vqa_rad:
    type: "medical_vqa"
    data_path: "data/vqa_rad"
    split: ["train", "val", "test"]
    metrics: ["accuracy", "bleu"]
    
  embodied_vqa:
    type: "embodied_vqa"
    data_path: "data/embodied_vqa"
    split: ["train", "val"]
    metrics: ["accuracy", "rouge"]

  adversarial:
    epsilon: 0.1
    num_trials: 5
    min_semantic_similarity: 0.7
  perturbation_types: ["synonym", "typo", "negation"]
  image_perturbations: ["noise", "contrast", "brightness", "text_overlay", "metadata", "filtered", "fgsm"]
```

## Results Format

The adversarial evaluation produces a JSON file with the following structure:

```json
{
  "model_name": {
    "task_name": {
      "clean_performance": {
        "accuracy": 0.85,
        "f1": 0.83
      },
      "adversarial_performance": {
        "accuracy": {
          "mean": 0.75,
          "std": 0.02
        },
        "f1": {
          "mean": 0.73,
          "std": 0.03
        }
      },
      "robustness_metrics": {
        "accuracy_degradation": 0.118,
        "accuracy_gap": 0.1,
        "accuracy_consistency": 0.882
      },
      "timing_metrics": {
        "clean": {
          "total_time": 120.5,
          "mean_inference_time": 0.5
        },
        "adversarial": {
          "total_time": 150.2,
          "mean_inference_time": 0.6
        }
      }
    }
  }
}
```

## Adding New Models

1. Create a new model class in `benchmark_suite/models/` that inherits from `BaseModel`
2. Implement required methods:
   - `load_model()`
   - `generate()`
   - `process_image()`
   - `process_video()`

## Adding New Tasks

1. Create a new task class in `benchmark_suite/tasks/` that inherits from `BaseTask`
2. Implement required methods:
   - `load_data()`
   - `evaluate()`
   - `format_prompt()`

## Adding New Adversarial Generators

1. Create a new generator class in `benchmark_suite/adversarial/generators/` that inherits from `BaseAdversarialGenerator`
2. Implement required methods:
   - `generate_text_perturbation()`
   - `generate_image_perturbation()`
   - `generate_multimodal_perturbation()`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

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
        title         = {OpenEQA: Embodied Question Answering in the Era of Foundation Models}, 
        booktitle     = {Conference on Computer Vision and Pattern Recognition (CVPR)},
        author        = {Majumdar, Arjun and Ajay, Anurag and Zhang, Xiaohan and Putta, Pranav and Yenamandra, Sriram and Henaff, Mikael and Silwal, Sneha and Mcvay, Paul and Maksymets, Oleksandr and Arnaud, Sergio and Yadav, Karmesh and Li, Qiyang and Newman, Ben and Sharma, Mohit and Berges, Vincent and Zhang, Shiqi and Agrawal, Pulkit and Bisk, Yonatan and Batra, Dhruv and Kalakrishnan, Mrinal and Meier, Franziska and Paxton, Chris and Sax, Sasha and Rajeswaran, Aravind},
        year          = {2024},
    }
```
```bibtex
@article{Lau2018,
   journal = {Scientific Data},
   author = {Jason J. Lau and Soumya Gayen and Asma Ben Abacha and Dina Demner-Fushman},
   publisher = {Nature Publishing Groups},
   title = {A dataset of clinically generated visual questions and answers about radiology images},
   year = {2018}
}
```

## Notes

### Adversarial Mitigation Strategies
- Image de-noising and reformer through an auto-encoder
- Distinguishing adversarial and benign samples based on certain criteria ("detector"); 
- Integrating adversarial samples in model training ("adversarial training");
- Gradient masking, such as pushing a model gradient to a non-differential or zero, which is effective for gradient-based white-box attacks.
