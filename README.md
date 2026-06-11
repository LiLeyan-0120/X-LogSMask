# X-LogSMask: Expand Transformer for Graph-Structured Data

PyTorch implementation of **X-LogSMask**, a Transformer-based framework for graph-structured data. This repository provides implementations across multiple graph learning tasks including node classification, graph classification, edge regression, and link prediction, as well as a Vision Transformer variant with X-LogSMask for image classification.

The codebase prioritizes methodological clarity and reproducibility over achieving state-of-the-art benchmark numbers.

## Repository Structure

```
GNN-Pytorch-main/
├── Node/GNN-Transformer/                      # Node classification
├── Graph/GNN-Transformer/                     # Graph classification
├── Edge/
│   ├── GNN-Transformer-edge_regression/       # Edge regression
│   └── GNN-Transformer-link_prediction/       # Link prediction
├── ViT_XLogSMask/                             # Vision Transformer with X-LogSMask (CIFAR-10)
├── Dataset/                                   # Shared dataset utilities
├── LICENSE
└── README.md
```

---

## Node Classification

### Datasets

| Dataset  | Nodes | Edges | Node Features | Classes | Train | Valid | Test |
| :------: | :---: | :---: | :-----------: | :-----: | :---: | :---: | :--: |
|   Cora   | 2708  | 5429  |     1433      |    7    |  140  |  500  | 1000 |
| Citeseer | 3327  | 4732  |     3703      |    6    |  120  |  500  | 1000 |
|  Pubmed  | 19717 | 44338 |      500      |    3    |  60   |  500  | 1000 |

Additional datasets supported via configuration: Computers, CS, Photo, Physics, WikiCS.

### Results (Accuracy)

| Method | Paper | Cora | Citeseer | Pubmed |
| :----: | :---: | :---: | :------: | :----: |
| GNN-Transformer | This work | 0.827 | -- | -- |

### Usage

```bash
cd Node/GNN-Transformer
python main.py --config config.yaml --dataset cora
```

For hyperparameter search:

```bash
python look_for_param.py --config config.yaml --dataset cora
```

---

## Graph Classification

### Datasets

|   Dataset   | Graphs | Avg. Nodes | Avg. Edges | Node Features | Classes | Train | Valid | Test |
| :---------: | :----: | :--------: | :--------: | :-----------: | :-----: | :---: | :---: | :--: |
|     DD      |  1178  |   284.32   |   715.66   |      89       |    2    |  826  |  117  | 235  |
|    NCI1     |  4110  |   29.87    |   32.30    |      37       |    2    | 2877  |  411  | 822  |
|  PROTEINS   |  1113  |   39.06    |   72.82    |       4       |    2    |  780  |  111  | 222  |
|    MUTAG    |   188  |   17.93    |   19.79    |       7       |    2    |  --   |  --   | --   |
|   COLLAB    |  5000  |   74.49    |  2457.78   |     492       |    3    |  --   |  --   | --   |
| IMDB-BINARY |  1000  |   19.77    |   96.53    |     136       |    2    |  --   |  --   | --   |

Additional datasets supported via configuration: ZINC, ZINC_full, MOLHIV.

### Results (Accuracy)

| Method | Paper | DD | NCI1 | PROTEINS |
| :----: | :---: | :--: | :--: | :------: |
| GNN-Transformer | This work | 0.824 | -- | -- |

### Usage

```bash
cd Graph/GNN-Transformer
python main.py --config config.yaml --dataset DD
```

---

## Edge Regression

Predicts continuous edge attributes using a Transformer encoder applied to node pairs and their local graph context.

### Datasets

|       Dataset       | Node Features | Edge Features |
| :-----------------: | :-----------: | :-----------: |
|   epic-games-plr   |      573      |      512      |
| air-traffic-2019-rlr |     128      |      20       |
| air-traffic-2015-rlr |     128      |      20       |

### Usage

```bash
cd Edge/GNN-Transformer-edge_regression
python main.py --config config.yaml --dataset epic-games-plr
```

---

## Link Prediction

Performs link prediction using the Transformer encoder, evaluated with MRR and Hit Rate metrics.

### Datasets

| Dataset  | Node Features | Metrics |
| :------: | :-----------: | :-----: |
|   Cora   |     1433      |   MRR   |
| Citeseer |     3703      |   MRR   |
|  Pubmed  |      500      |   MRR   |
|  Collab  |      128      |  HR@50  |
|   PPA    |      58       |  HR@100 |
| Citation2|      128      |   MRR   |
|   DDI    |      128      |  HR@20  |

### Usage

```bash
cd Edge/GNN-Transformer-link_prediction
python main.py --config config.yaml --dataset cora
```

---

## Vision Transformer with X-LogSMask

A standalone Vision Transformer (ViT) variant that incorporates X-LogSMask for image classification on CIFAR-10. This module demonstrates the generalizability of the X-LogSMask mechanism beyond graph-structured data.

### Structure

```
ViT_XLogSMask/
├── configs/                  # Configuration files
├── models/
│   ├── vit.py                # Standard ViT implementation
│   └── vit_xlogsmask.py      # ViT with X-LogSMask
├── utils/                    # Utility functions
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
└── requirements.txt          # Dependencies
```

### Usage

Install dependencies:

```bash
cd ViT_XLogSMask
pip install -r requirements.txt
```

Train:

```bash
python train.py --config configs/cifar10_config.yaml
```

Evaluate:

```bash
python evaluate.py --config configs/cifar10_config.yaml --model_path path/to/model.pth
```

---

## Environment Setup

### Core Dependencies (Graph Models)

|    Package     | Version | Installation                                                        |
| :------------: | :-----: | :------------------------------------------------------------------ |
|     Python     |  3.8.6  | `conda create --name gnn python=3.8.6`                              |
|     NumPy      | 1.20.0  | `pip install numpy==1.20.0`                                         |
|     SciPy      |  1.6.0  | `pip install scipy==1.6.0`                                          |
|     PyYAML     |  5.4.1  | `pip install pyyaml==5.4.1`                                         |
|  scikit-learn  | 0.24.1  | `pip install scikit-learn==0.24.1`                                  |
|    PyTorch     |  1.7.1  | `conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch`          |
| PyTorch Geometric | 1.6.3 | [Installation Guide](https://github.com/rusty1s/pytorch_geometric#installation) |

### ViT_XLogSMask Dependencies

Install separately for the Vision Transformer module:

```bash
cd ViT_XLogSMask
pip install -r requirements.txt
```

Requires: `torch>=1.10.0`, `torchvision>=0.11.0`, `numpy>=1.19.5`, `tqdm>=4.6.0`, `PyYAML>=5.4.1`

---

## License

This project is licensed under the terms specified in the [LICENSE](./LICENSE) file.
