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
node2vec | | |
DeepWalk | | |
MNE | | |
R-graphSAGE | | |
R-GCN | | |
GATNE | | |
HAN | | |
RAHMeN | | |

### Twitter
|  | ROC-AUC | F1 |
|---|---|---|
node2vec | | |
DeepWalk | | |
MNE | | |
R-graphSAGE | | |
R-GCN | | |
GATNE | | |
HAN | | |
RAHMeN | | |

### YouTube
|  | ROC-AUC | F1 |
|---|---|---|
node2vec | | |
DeepWalk | | |
MNE | | |
R-graphSAGE | | |
R-GCN | | |
GATNE | | |
HAN | | |
RAHMeN | | |

### Tissue PPI
|  | ROC-AUC | F1 |
|---|---|---|
node2vec | | |
DeepWalk | | |
MNE | | |
R-graphSAGE | | |
R-GCN | | |
GATNE | | |
HAN | | |
RAHMeN | | |