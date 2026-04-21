```markdown
# MCPST-FSL: Physics-Informed Multi-Phase Consensus Spatio-Temporal Few-Shot Learning for Traffic Flow Forecasting

[![arXiv](https://img.shields.io/badge/arXiv-2602.01936-b31b1b.svg)](https://arxiv.org/abs/2602.01936)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**MCPST-FSL** is a novel Multi-Phase Consensus Spatio-Temporal framework for few-shot traffic forecasting that reconceptualises traffic prediction as a multi-phase consensus learning problem. It enables robust cross-city prediction with minimal historical data by integrating physics-informed dynamics with meta-learning.

### Key Features

- **Multi-Phase Engine**: Models traffic dynamics through three complementary physics-inspired phases:
  - *Thermodynamic diffusion* — heat-equation-based spatial propagation
  - *Kuramoto synchronisation* — oscillator-based phase coupling across nodes
  - *Spectral decomposition* — graph Laplacian eigenvector embeddings
- **Adaptive Consensus Mechanism**: Dynamically fuses phase-specific predictions with learnable attention weights while enforcing physical consistency
- **Meta-Learning Strategy (FOMAML)**: Two-phase training (source pre-training + target adaptation) for rapid few-shot generalisation to unseen cities
- **Horizon-Specific Prediction**: Dedicated short/medium/long-term heads with built-in uncertainty quantification
- **Theoretical Guarantees**: Bounded approximation errors and generalisation bounds for few-shot adaptation

---

## Architecture

```
Input Features
      │
      ▼
Input Projection (Linear + LayerNorm)
      │
      ▼
StabilizingGCN  ──────────────────────────────────────┐
      │                                                │
      ├──► ThermodynamicModule (Heat Diffusion)        │
      ├──► KuramotoModule (Phase Synchronisation)      │
      └──► SpectralModule (Graph Eigenvectors)         │
                    │                                  │
                    ▼                                  │
            AdaptiveFusion                             │
                    │                                  │
                    ▼                                  │
      MultiScaleTemporalEncoder                        │
      (LSTM × 4 scales + Transformer)                  │
                    │                                  │
                    ▼                                  │
            Final Fusion  ◄────────────────────────────┘
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
HorizonSpecificPredictor    PhysicsFlow
(short/medium/long heads)   (weighted sum of
 + uncertainty estimation    phase predictions)
        │                        │
        └───────────┬────────────┘
                    ▼
           Final Prediction
         (0.7 × learned + 0.3 × physics)
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- CUDA (optional, recommended for large datasets)

### Setup

```bash
# Clone the repository
git clone https://github.com/afofanah/MCPST-FSL.git
cd MCPST-FSL

# Create conda environment
conda create -n mcpst python=3.8
conda activate mcpst

# Install PyTorch (adjust cuda version as needed)
pip install torch==1.12.1+cu113 torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# Install PyTorch Geometric
pip install torch-geometric

# Install remaining dependencies
pip install numpy pyyaml matplotlib scikit-learn tqdm
```

---

## Data Preparation

Download datasets and place them in the `data/` directory:

| Dataset   | Nodes | Edges | Source                                      |
|-----------|-------|-------|---------------------------------------------|
| METR-LA   | 207   | 1,515 | [DCRNN](https://github.com/liyaguang/DCRNN) |
| PEMS-BAY  | 325   | 2,369 | [DCRNN](https://github.com/liyaguang/DCRNN) |
| Chengdu-M | 524   | —     | Available upon request                      |
| Shenzhen  | 627   | —     | Available upon request                      |

Expected directory structure:

```
data/
├── metr-la/
│   ├── dataset.npy       # Node feature matrix [T, N, F]
│   └── matrix.npy        # Adjacency matrix [N, N]
├── pems-bay/
│   ├── dataset.npy
│   └── matrix.npy
├── chengdu_m/
│   ├── dataset.npy
│   └── matrix.npy
└── shenzhen/
    ├── dataset.npy
    └── matrix.npy
```

---

## Usage

### Quick Start

```bash
# Train with default settings (METR-LA, few-shot, 3 target days)
python main.py

# Train on a specific dataset
python main.py --dataset pems-bay

# Train with custom few-shot settings
python main.py --dataset metr-la --target_days 3 --support_size 5 --query_size 10

# Train without few-shot (standard mode)
python main.py --dataset metr-la --few_shot False
```

### Full CLI Options

```bash
python main.py \
  --config config.yaml \           # Path to config file
  --dataset metr-la \              # Dataset: metr-la, pems-bay, chengdu_m, shenzhen
  --seed 42 \                      # Random seed
  --device auto \                  # auto | cpu | cuda
  --source_epochs 250 \            # Phase 1 training epochs
  --target_epochs 250 \            # Phase 2 adaptation epochs
  --target_days 3 \                # Few-shot target days
  --source_lr 0.0003 \             # Source domain learning rate
  --target_lr 0.0001 \             # Target domain fine-tuning LR
  --hidden_dim 64 \                # Hidden dimension
  --pred_len 12 \                  # Prediction horizon (steps)
  --num_diffusion_steps 6 \        # Thermodynamic diffusion steps
  --num_oscillator_steps 10 \      # Kuramoto oscillator steps
  --num_eigen_vectors 8 \          # Spectral eigenvectors
  --adaptation_steps 3 \           # MAML inner-loop steps
  --inner_lr 0.01 \                # Inner-loop learning rate
  --outer_lr 0.0005 \              # Outer-loop (meta) learning rate
  --alpha_flow 2.0 \               # Flow prediction loss weight
  --alpha_spatial 0.08 \           # Spatial loss weight
  --alpha_temporal 0.08 \          # Temporal loss weight
  --alpha_physics 0.03 \           # Physics regularisation weight
  --alpha_uncertainty 0.01         # Uncertainty calibration weight
```

### Configuration File

Key settings in `config.yaml`:

```yaml
training:
  test_dataset: metr-la
  source_epochs: 250
  target_epochs: 250
  target_days: 3
  source_lr: 0.0003
  target_lr: 0.0001
  add_physics_features: true

few_shot:
  enabled: true
  support_size: 5
  query_size: 10
  num_tasks: 1
  adaptation_steps: 3
  inner_lr: 0.01
  outer_lr: 0.0005

model:
  hidden_dim: 64
  pred_len: 12
  num_diffusion_steps: 6
  num_oscillator_steps: 10
  num_eigen_vectors: 8
  dropout: 0.1
  physics:
    alpha_flow: 2.0
    alpha_spatial: 0.08
    alpha_temporal: 0.08
    alpha_physics: 0.03
    alpha_consistency: 0.015
    alpha_uncertainty: 0.01
```

---

## Training Pipeline

MCPST-FSL uses a two-phase training strategy:

**Phase 1 — Source Domain Meta-Learning**
- Trains on source city data using FOMAML
- Inner loop: `adaptation_steps` gradient steps on support set
- Outer loop: meta-update using query set loss
- Validates on target domain every 10 epochs

**Phase 2 — Target Domain Adaptation**
- Fine-tunes on limited target city data (`target_days` of observations)
- Uses lower learning rate (`target_lr`) to preserve meta-learned features
- Validates every 15 epochs

---

## Evaluation

Metrics reported:
- **MAE** — Mean Absolute Error (km/h, denormalised)
- **RMSE** — Root Mean Square Error
- **MAPE** — Mean Absolute Percentage Error (%)

---

## Project Structure

```
MCPST-FSL/
├── data/                    # Dataset files (download separately)
├── datasets/                # Raw dataset archives
├── models/
│   ├── model.py             # MCPST_FSL architecture + PhysicsInformedLoss
│   ├── model_v2.py          # Experimental variant
│   └── motivation.py        # Motivation experiments
├── datasets.py              # Data loading, preprocessing, few-shot sampling
├── train.py                 # EnhancedFewShotTrainer + EnhancedStandardTrainer
├── main.py                  # Entry point + CLI argument parsing
├── utils.py                 # Metrics, logging, early stopping, visualisation
├── plots.py                 # Plotting utilities
├── plots_Revised.py         # Revised plotting utilities
├── results.py               # Results aggregation
└── config.yaml              # Default configuration
```

---

## Memory and Compute

MCPST-FSL includes automatic minibatch chunking for memory-constrained hardware:

| Hardware | Default Minibatch | Recommendation                          |
|----------|-------------------|-----------------------------------------|
| GPU      | 2048 samples      | V100 32GB or better for large datasets  |
| CPU      | 4096 samples      | Tested on Apple Silicon (M-series)      |

Override with `--minibatch_size` for your hardware.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fofanah2026pimcst,
  title={PIMCST: Physics-Informed Multi-Phase Consensus and Spatio-Temporal 
         Few-Shot Learning for Traffic Flow Forecasting},
  author={Fofanah, Abdul Joseph and Wen, Lian and Chen, David},
  journal={arXiv preprint arXiv:2602.01936},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaborations, please open an issue or contact the authors via the paper.
```
