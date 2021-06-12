# RAHMeN: Relation-aware embeddings for Attributed Heterogeneous Multiplex Networks

This repository contains the source code for the experiments conducted and models described in the RAHMeN paper.

## Prerequisites

- PyTorch 1.8.1 [Link](https://pytorch.org/)
- DGL 0.6.1 [Link](https://www.dgl.ai/)

## Getting Started

First install PyTorch and DGL for your CUDA environment. Install remaining dependencies by

```bash
pip install -r requirements.txt
```

## Datasets

The DGL graphs with node features and val/test edge sets for each dataset are provided in the [data](/data) directory.

- The Amazon and YouTube datasets along with their train/val/test splits and feature sets are from [Cen et al. (2019)](https://github.com/THUDM/GATNE).
- The Twitter dataset is sampled from [Source](https://snap.stanford.edu/data/higgs-twitter.html).
- The Tissue-PPI dataset is sampled from [Source](http://snap.stanford.edu/ohmnet/).

For the Twitter dataset, the 3-layer reply, retweet, and mention network is sampled for all nodes which participate in at least one reply. We then extract the largest strongly connected component in this network for our experiments.
10% of edges are randomly sampled for a test set and 5% of edges for a validation set. An equivalent number of negative edges is added to each test and validation set.

For the Tissue-PPI data, two datasets are generated for the transductive and inductive experiments. Both datasets are derived from the 10-layer network consisting of the ten largest tissue layers in the original data.
For the transductive experiments, each edge was randomly split into 5 cross-validation folds. For the inductive experiment, 15% of nodes were masked from the graph.
20% of remaining edges were sampled as a validation set and removed from the training graph. For evaluation, 50% of the edges incident on the removed nodes were added to the training graph,
and the remaining 50% of edges, along with an equivalent number of randomly sampled negative edges were used as a test set.

## Training and Evaluation

You can use `python -m main_rahmen <dataset> <out_model_name>` to train and evaluate RAHMeN on the provided dataset.
Dataset options are
- amazon
- twitter
- youtube
- tissue_ppi

To train and evaluate a model on the inductive tissue ppi dataset, you can add the `--inductive` argument.

## Results

### Amazon
|  | ROC-AUC | F1 |
|---|---|---|
node2vec | 94.47 | 87.88 |
DeepWalk | 94.20 | 87.38 |
MNE | 90.28 | 83.25 |
R-graphSAGE | 94.88 | 89.39 |
R-GCN | 94.96 | 90.08 |
GATNE | 96.25 | 91.36 |
HAN | 95.28 | 90.43 |
RAHMeN | **96.78** | **92.39** |

### Twitter
|  | ROC-AUC | F1 |
|---|---|---|
node2vec | 72.58 | 71.94 |
DeepWalk | 76.88 | 72.42 |
MNE | OOT | OOT |
R-graphSAGE | 74.31 | 70.77 |
R-GCN | 92.75 | 85.85 |
GATNE | 92.94 | 86.20 |
HAN | **94.81** | **88.44** |
RAHMeN | **94.58** | **88.31** |

### YouTube
|  | ROC-AUC | F1 |
|---|---|---|
node2vec | 71.21 | 65.36 |
DeepWalk | 71.11 | 65.52 |
MNE | 82.30 | 75.03 |
R-graphSAGE | 87.02 | 79.93 |
R-GCN | 80.21 | 73.36 |
GATNE | 84.47 | 76.83 |
HAN | 80.43 | 73.43 |
RAHMeN | **88.64** | **80.58** |

### Tissue PPI
|  | ROC-AUC | F1 |
|---|---|---|
node2vec | 51.30 | 64.04 |
DeepWalk | 58.48 | 67.16 |
MNE | OOT | OOT |
R-graphSAGE | 66.61 | 61.59 |
R-GCN | 84.19 | 75.98 |
GATNE | 79.83 | 71.78 |
HAN | 93.05 | 85.98 |
RAHMeN | **94.88** | **87.99** |