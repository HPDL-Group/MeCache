# MeCache: Communication-Efficient Multi-GPU Heterogeneous Graph Neural Network Training

This repository contains the code for the paper **"MeCache: Communication-Efficient Multi-GPU Heterogeneous Graph Neural Network Training"**, which has been accepted for [IPDPS&#39;26](https://placeholder.com).

## Overview

MeCache is a system designed to alleviate the CPU–GPU communication bottleneck in multi-GPU training of heterogeneous graph neural networks (HGNNs).
It introduces two key innovations:

- Meta Reduction: A semantic-aware feature dimensionality reduction strategy that applies type-specific compression to node features, significantly reducing data transfer volume while preserving model accuracy.
- Fusion Cache: A fine-grained, data-level caching mechanism that implements differentiated cache policies for read-only features, embeddings, and optimizer states, guided by a cost model for optimal GPU memory allocation.

## Requirements

- Python 3.7
- CUDA 11.7
- PyTorch 1.13
- DGL 1.1
- Additional dependencies:
  - numpy
  - networkx

## Installation

1. Clone the repository:

```bash
git clone https://github.com/HPDL-Group/MeCache.git
cd MeCache
```

2. Install other dependencies:

```bash
pip install -r requirements.txt
```

## Repo Structure

- `MeCache.pdf` includes the original paper, with the last two pages serving as the Artifact Description, providing detailed instructions for reproducing the computational results reported in the paper.
- `load_graph.py` contain the dataloader of datasets.
- `sparse_optim.py` and `gpu_cache.py` contain the implementation of the GPU caching strategies.
- `model.py` contains the implementation of the HGNN models.
- `run_all_acc.py`, `run_all_cost_model_dgl_fc.py`, `run_all_cost_model.py`, `run_all_dgl_fc.py`, `run_all_dgl_mr.py`, `run_all_preprocess.py`, `run_all_speed.py` and `run_cost_model_for_fig10.py` contain evaluation scripts used in the paper.
- `run_reduction.sh`, `run_no_reduction.sh` and `train_dist.py` contains the training script.
- The `scripts` directory contains the plotting scripts used to generate the figures and tables in the paper. See the `README.md` file in each subdirectory for details.

## Datasets

The code supports various heterogeneous graph datasets including those from

- IGBH-Small, IGBH-Medium, IGBH-Large: `https://github.com/IllinoisGraphBenchmark/IGB-Datasets`
- ogbn-mag, MAG240M: `https://github.com/snap-stanford/ogb`

## Running Experiments

To run MeCache on a dataset (take ogbn-mag as an example), follow the steps below:

1. Prepare the environment: prepare `ipconfig.txt` file with the ip addresses of the machine in the cluster
2. Apply Meta Reduction to the graph (optional):

```bash
python -u ./scripts/run/preprocess.py --dataset ogbn-mag --out-dir preprocess --root /datasets/gnn/dgldata --reduction-level 32,8
```

The features of the ogbn-mag dataset will be reduced and saved in the `preprocess` directory.If Meta Reduction is applied, use `run_reduction.sh` and specify the corresponding `REDUCTION_LEVEL` parameter.
Otherwise, use `run_no_reduction.sh`.

3. Profile hotness for GPU cache (optional):

```bash
python -u ./scripts/run/cost_model.py --dataset ogbn-mag --model rgcn --root /datasets/gnn/dgldata --worker 4 --ntypes-w-feats paper --reduction-level 32,8 --budget 24
```

Enable the GPU cache in `run_reduction.sh` or `run_no_reduction.sh` by setting `CACHE_METHOD` to `miss_penalty`. Default is `none`.

4. Train the model:

```bash
./scripts/run/run_reduction.sh rgcn ogbn-mag paper 349 1024 paper miss_penalty 64 64 5,10,15 3 0.5 32,8 /datasets/gnn/dgldata ip_config.txt 0.01 0.06 4 --use_node_projs --no-test
```

Example command with key parameters explained in the script comments.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

<!-- ## Cite Our Paper

placeholder -->
