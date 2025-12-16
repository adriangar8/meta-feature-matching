# Learning Adaptive Visual Descriptors via Contrastive Learning and Continual Adaptation

A comprehensive benchmark comparing traditional handcrafted descriptors (SIFT, ORB, BRISK) with learned deep descriptors trained via contrastive learning, with a focus on continual learning strategies for domain adaptation.

## Overview

This project investigates whether learned descriptors can match or exceed traditional handcrafted descriptors for feature matching, and how continual learning enables adaptation to new visual domains without catastrophic forgetting.

### Key Findings

-   **Learned Descriptors**: Achieve **51.7% accuracy** on illumination changes and **40.9%** on viewpoint changes
-   **Traditional Baseline (SIFT)**: **67.9%** on illumination, **48.0%** on viewpoint
-   **Continual Learning**: Learning without Forgetting (LwF) achieves the best trade-off with only **4.1-5.1% forgetting** (vs 9-21% for naive fine-tuning)
-   **Domain Adaptation**: Successfully enables models to adapt from illumination→viewpoint and vice versa

### Methodology

The project uses:

-   **Contrastive Learning**: Triplet loss with hard negative mining
-   **Architecture**: ResNet-50 (ImageNet pretrained) with 128-dim descriptors
-   **Evaluation**: Patch matching accuracy on HPatches benchmark
-   **Continual Methods**: Naive fine-tuning, Elastic Weight Consolidation (EWC), Learning without Forgetting (LwF)

## Hardware Requirements

This benchmark was developed and tested on:

-   **GPU**: NVIDIA Ada Generation (24GB) - CUDA 13.0
-   **Driver Version**: 580.105.08
-   **CPU**: AMD Ryzen 9 5900X
-   **RAM**: 64GB

**Minimum Requirements:**

-   GPU: 8GB VRAM (NVIDIA with CUDA support recommended)
-   RAM: 16GB
-   Storage: ~5GB (dataset + results)

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
conda create -n feature-matching python=3.9
conda activate feature-matching

# Install PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Option 2: Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The `requirements.txt` includes:

-   PyTorch >= 2.0.0
-   torchvision
-   numpy
-   opencv-python (cv2)
-   matplotlib
-   scikit-learn
-   tqdm
-   wandb (for experiment tracking)
-   pyyaml

## Dataset Setup

### Download HPatches

1. **Download the dataset**:

    ```bash
    # Create dataset directory
    mkdir -p dataset
    cd dataset

    # Download HPatches (1.3GB)
    wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz

    # Extract
    tar -xzf hpatches-sequences-release.tar.gz
    mv hpatches-sequences-release hpatches

    # Clean up
    rm hpatches-sequences-release.tar.gz
    ```

2. **Verify structure**:
    ```
    project-root/
    ├── dataset/
    │   └── hpatches/
    │       ├── i_ajuntament/
    │       ├── i_bologna/
    │       ├── v_artisans/
    │       └── ... (116 sequences total)
    ├── src/
    ├── scripts/
    └── configs/
    ```

### HPatches Dataset Info

-   **Size**: 1.3GB compressed, ~3GB extracted
-   **Sequences**: 116 total (57 illumination, 59 viewpoint)
-   **Images per sequence**: 6 (1 reference + 5 targets with increasing transformation)
-   **Format**: .ppm images with homography ground truth
-   **URL**: http://icvl.ee.ic.ac.uk/vbalnt/hpatches/

## Usage

### Quick Test

Test the pipeline with reduced data:

```bash
python scripts/run_benchmark.py --quick
```

This runs:

-   5 epochs training
-   5,000 triplets (vs 50,000 default)
-   50 distractors (vs 1,000 default)
-   Outputs to `results/quick_test/`

### Full Paper Experiments (~3-4 hours on GPU)

Run the complete benchmark as described in the paper:

```bash
python scripts/run_benchmark.py --config configs/paper_config.yaml
```

This evaluates:

-   Traditional methods (SIFT, ORB, BRISK)
-   Deep learning descriptors (illumination & viewpoint trained)
-   Continual learning (Naive, EWC, LwF)

### Custom Configuration

```bash
# Specify custom dataset path
python scripts/run_benchmark.py --hpatches /path/to/hpatches

# Custom output directory
python scripts/run_benchmark.py --output_dir results/my_experiment

# Use a YAML config
python scripts/run_benchmark.py --config my_config.yaml
```

### Command Line Options

```
--config PATH          Path to YAML configuration file
--quick               Run quick test with reduced parameters
--hpatches PATH       Override HPatches dataset root path
--output_dir PATH     Override output directory
```

### Configuration File Options

Create a custom YAML config (see `configs/paper_config.yaml`):

```yaml
# Dataset
hpatches_root: "dataset/hpatches"

# Experiment settings
seed: 42
eval_traditional: true # Evaluate SIFT, ORB, BRISK
eval_deep: true # Train & evaluate deep descriptors
eval_continual: true # Evaluate continual learning

# Training hyperparameters
deep_epochs: 30 # Epochs for initial training
epochs_source: 30 # Epochs on source domain
epochs_target: 30 # Epochs on target domain
lr: 5e-5 # Learning rate
batch_size: 32 # Batch size

# Data parameters
max_triplets: 50000 # Training triplets to generate
max_pairs_per_seq: 1000 # Eval pairs per sequence
max_distractors: 1000 # Distractors per query

# Continual learning
continual_methods: # Methods to evaluate
    - naive
    - ewc
    - lwf
ewc_lambda: 400 # EWC regularization strength
lwf_lambda: 1.0 # LwF distillation strength

# Output
output_dir: "results/paper_results"
```

## Output Structure

Results are organized as:

```
results/<experiment_name>/
├── results.json              # All metrics (JSON)
├── summary.json             # Summary statistics
├── figures/                 # Visualizations
│   ├── illumination/       # Domain-specific figures
│   └── viewpoint/
├── patches/                 # Example patches
│   ├── illumination/
│   └── viewpoint/
├── global_images/          # Full image visualizations
├── tsne_figures/           # T-SNE embeddings
├── matching_comparisons/   # SIFT vs Learned
└── paper_figures/          # Paper methodology figures
```

### Key Output Files

-   **results.json**: Complete results including:
    -   Traditional methods accuracy (SIFT, ORB, BRISK)
    -   Deep learning performance (per domain)
    -   Continual learning metrics (forgetting rates, target accuracy)
-   **Visualizations**:
    -   T-SNE embedding projections
    -   Matching comparison figures
    -   Training loss curves
    -   Comprehensive accuracy plots

## Benchmark Results

Results from the paper (HPatches test set):

### Traditional Methods

| Method | Illumination | Viewpoint |
| ------ | ------------ | --------- |
| SIFT   | 67.9%        | 48.0%     |
| ORB    | 48.8%        | 25.4%     |
| BRISK  | 37.0%        | 19.6%     |

### Deep Learning (Domain-Specific)

| Training Domain | Illumination | Viewpoint |
| --------------- | ------------ | --------- |
| Illumination    | 51.7%        | 29.5%     |
| Viewpoint       | 44.7%        | 40.9%     |

### Continual Learning (Forgetting Rates)

| Method  | I→V Forgetting | V→I Forgetting |
| ------- | -------------- | -------------- |
| Naive   | 9.0%           | 20.7%          |
| EWC     | 10.4%          | 19.1%          |
| **LwF** | **5.1%**       | **4.1%**       |

## Project Structure

The codebase is organized into modular components:

```
src/
├── data/                   # Data handling
│   ├── hpatches_manager.py    # Dataset loader
│   ├── dataset.py             # PyTorch Dataset
│   └── structures.py          # Data classes
│
├── descriptors/            # Feature extraction
│   └── traditional.py         # SIFT, ORB, BRISK
│
├── evaluation/             # Evaluation logic
│   └── evaluator.py           # Evaluation functions
│
├── training/               # Training procedures
│   ├── trainer.py             # Standard training
│   └── continual.py           # Continual learning (EWC, LwF)
│
├── visualization/          # Plotting & figures
│   ├── tsne_viz.py            # T-SNE visualizations
│   ├── matching_comparison.py # Comparison plots
│   └── methodology_figure.py  # Paper figures
│
├── utils/                  # Utilities
│   ├── preprocessing.py       # Patch normalization
│   ├── config_utils.py        # Config parsing
│   └── seed_utils.py          # Reproducibility
│
└── runner/                 # Main orchestration
    └── benchmark_runner.py    # Benchmark coordinator
```

## Experiment Tracking

The benchmark uses [Weights & Biases](https://wandb.ai) for experiment tracking:

```bash
# Login to wandb (first time only)
wandb login
```

To disable wandb:

```bash
export WANDB_MODE=disabled
python scripts/run_benchmark.py --quick
```

## License

This project is part of academic research at École Polytechnique.

## Contributing

This is a research project. For questions or suggestions:

-   Open an issue describing the problem/suggestion
-   For bugs, include: OS, Python version, GPU info, error message

## Acknowledgments

-   **HPatches Dataset**: Balntas et al., CVPR 2017
-   **Architecture**: ResNet-50 (He et al., CVPR 2016)
-   **Continual Learning**: EWC (Kirkpatrick et al., PNAS 2017), LwF (Li & Hoiem, TPAMI 2018)
-   **Institution**: École Polytechnique - Institut Polytechnique de Paris

## References

Key papers:

-   Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. IJCV.
-   Balntas, V. et al. (2017). HPatches: A benchmark and evaluation of handcrafted and learned local descriptors. CVPR.
-   Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS.
-   Li, Z., & Hoiem, D. (2018). Learning without forgetting. TPAMI.
