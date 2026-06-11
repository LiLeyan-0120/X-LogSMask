# X-LogSMask: Expand Transformer for Graph-Structured Data

[![Paper](https://img.shields.io/badge/Paper-TPAMI%202025-blue)](https://github.com/LiLeyan-0120/X-LogSMask)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **X-LogSMask**, an explainable multi-head logarithmic structural mask that adapts Transformers for graph-structured data.

**Authors:** Leyan Li, Rennong Yang, Zhenxing Zhang, Liping Hu

---

## Abstract

Transformers have become general-purpose architectures, but their all-to-all self-attention is poorly matched to graph data, whose interactions are sparse, structured and multi-scale. We introduce **X-LogSMask**, an e**x**plainable multi-head **log**arithmic **s**tructural **mask** that injects symmetrically normalized graph topology directly into attention logits. The logarithmic transform converts structural connectivity into a topology-aware gating signal, suppressing unsupported node interactions while preserving feature-dependent attention.

By assigning different powers of the normalized adjacency matrix to different attention heads, X-LogSMask gives each head a defined structural radius and supports multi-hop information propagation within a single layer. Across **20 node-, edge- and graph-level benchmarks**, Transformers equipped with X-LogSMask achieve **state-of-the-art performance on 13 datasets** and remain competitive in a lightweight one-layer configuration.

---

## Key Features

- **Topologically Constrained Message Passing**: Suppresses attention between non-adjacent nodes via an additive structural mask
- **Explicit Graph Inductive Bias**: Injects graph topology directly into the attention matrix
- **Explainable Multi-head Mechanism**: Assigns distinct powers of normalized adjacency to different heads, encoding multi-hop structural information within a single layer
- **Lightweight Architecture**: Achieves competitive performance with only 1-layer configuration

---

## Repository Structure

```
X-LogSMask/
├── Edge/
│   ├── GNN-Transformer-edge_regression/    # Edge regression tasks
│   │   ├── main.py
│   │   ├── config.yaml
│   │   └── script/
│   └── GNN-Transformer-link_prediction/    # Link prediction tasks
│       ├── main.py
│       ├── config.yaml
│       └── script/
├── Graph/
│   └── GNN-Transformer/                    # Graph-level classification
│       ├── main.py
│       ├── config.yaml
│       └── script/
├── Node/
│   └── GNN-Transformer/                    # Node-level classification
│       ├── main.py
│       ├── config.yaml
│       └── script/
├── ViT_XLogSMask/                          # Vision Transformer extension
│   ├── train.py
│   ├── evaluate.py
│   ├── configs/
│   ├── models/
│   └── utils/
├── LICENSE
└── README.md
```

---

## Results

### Node-Level Classification (Accuracy %)

| Method | Cora | Citeseer | Pubmed | Computers | Photo | CS | Physics | WikiCS | Avg. Rank |
|:------:|:----:|:--------:|:------:|:---------:|:-----:|:--:|:-------:|:------:|:---------:|
| GCN | 81.60 | 71.60 | 78.80 | 89.65 | 92.70 | 92.92 | 96.18 | 77.47 | 10.6 |
| GraphSAGE | 82.68 | 71.93 | 79.41 | 91.20 | 94.59 | 93.91 | 96.49 | 74.77 | 8.1 |
| GAT | 83.00 | 72.50 | 79.00 | 90.78 | 93.87 | 93.61 | 96.17 | 76.91 | 8.6 |
| GraphGPS | 82.84 | **72.73** | 79.94 | 91.19 | 95.06 | 93.93 | 97.12 | 78.66 | 5.1 |
| Polynormer | 83.25 | 72.31 | 79.24 | **93.68** | 96.46 | 95.53 | 97.27 | 80.10 | 3.8 |
| **X-LogSMask (1-layer)** | 80.80 | 71.00 | 79.20 | 91.21 | 95.69 | 96.56 | 97.65 | **80.36** | 5.9 |
| **X-LogSMask (ours)** | 82.70 | 71.70 | 79.60 | 92.01 | **96.86** | **96.62** | **97.68** | **80.36** | **3.3** |

### Edge-Level Link Prediction (MRR)

| Method | Cora | Citeseer | Avg. Rank |
|:------:|:----:|:--------:|:---------:|
| GCN | 32.50 | 50.01 | 7.0 |
| LPFormer | 39.42 | 65.42 | 3.0 |
| **X-LogSMask** | **59.9** | **71.5** | **1.0** |

### Edge-Level Regression

| Method | epic-games-plr MAE | air-traffic-2019 MAE | air-traffic-2015 MAE | Avg. Rank |
|:------:|:------------------:|:--------------------:|:--------------------:|:---------:|
| eGCN | 0.1178 | 0.3983 | 0.1673 | 4.3 |
| **X-LogSMask** | **0.0149** | **0.1065** | **0.0989** | **1.0** |

### Graph-Level Classification

| Method | NCI1 | D&D | PROTEINS | MUTAG | COLLAB | IMDB-B | MOLHIV | Avg. Rank |
|:------:|:----:|:---:|:--------:|:-----:|:------:|:------:|:------:|:---------:|
| GraphGPS | **84.21** | - | 75.77 | 85.00 | 81.40 | **77.40** | 78.80 | 2.0 |
| **X-LogSMask (1-layer)** | 81.27 | **81.20** | 75.68 | **88.89** | 79.00 | 76.00 | 77.68 | 4.0 |
| **X-LogSMask (ours)** | 82.24 | **81.20** | **80.63** | **88.89** | 80.80 | 77.00 | **78.91** | **1.7** |

### Vision Transformer Extension (CIFAR-10)

| Run | ViT | ViT + X-LogSMask | Improvement |
|:---:|:---:|:----------------:|:-----------:|
| T1 | 82.18% | 86.07% | +3.89% |
| T2 | 80.96% | 85.08% | +4.12% |
| T3 | 81.44% | 84.15% | +2.71% |

---

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.7.1
- PyTorch Geometric >= 1.6.3

### Setup

```bash
# Clone the repository
git clone https://github.com/LiLeyan-0120/X-LogSMask.git
cd X-LogSMask

# Create conda environment
conda create --name xlogsmask python=3.8.6
conda activate xlogsmask

# Install dependencies
pip install numpy==1.20.0
pip install scipy==1.6.0
pip install pyyaml==5.4.1
pip install scikit-learn==0.24.1
pip install optuna

# Install PyTorch
conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch

# Install PyTorch Geometric
pip install torch-geometric==1.6.3
```

---

## Usage

### Node-Level Classification

```bash
cd Node/GNN-Transformer
python main.py
```

### Graph-Level Classification

```bash
cd Graph/GNN-Transformer
python main.py
```

### Edge-Level Link Prediction

```bash
cd Edge/GNN-Transformer-link_prediction
python main.py
```

### Edge-Level Regression

```bash
cd Edge/GNN-Transformer-edge_regression
python main.py
```

### Vision Transformer (CIFAR-10)

```bash
cd ViT_XLogSMask
python train.py --config configs/cifar10_config.yaml
```

---

## Method Overview

X-LogSMask constructs a structural mask from the symmetrically normalized adjacency matrix and injects it into attention logits:

1. **Symmetric Normalization**: $\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}(\mathbf{A} + \mathbf{I})\mathbf{D}^{-1/2}$

2. **Logarithmic Structural Mask**: $\mathbf{M}_{ij} = \log(\tilde{\mathbf{A}}_{ij} + \epsilon)$

3. **Multi-head Decomposition**: Assign $\tilde{\mathbf{A}}^k$ to the $k$-th head for multi-hop attention

4. **Attention Computation**: $\alpha_{ij}^{(r)} \propto \exp\left(\frac{\mathbf{q}_i^{(r)}{\mathbf{k}_j^{(r)}}^{\mathrm{T}} + \mathbf{M}_{ij}^{(r)}}{\sqrt{d_h}}\right)$

---

## Datasets

### Node-Level
Cora, Citeseer, Pubmed, Computers, Photo, CS, Physics, WikiCS

### Edge-Level
- **Link Prediction**: Cora, Citeseer
- **Edge Regression**: epic-games-plr, air-traffic-2019-rlr, air-traffic-2015-rlr

### Graph-Level
NCI1, D&D, PROTEINS, MUTAG, COLLAB, IMDB-B, MOLHIV

### Vision
CIFAR-10

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{li2025xlogsmask,
  title={X-LogSMask: Expand Transformer for Graph-Structured Data},
  author={Li, Leyan and Yang, Rennong and Zhang, Zhenxing and Hu, Liping},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
