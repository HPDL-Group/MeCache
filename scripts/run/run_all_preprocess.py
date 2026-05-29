import os
import itertools

datasets = ['ogbn-mag', 'igb-full-small', 'igb-full-medium', 'mag240m', 'igb-full-large']
root={
    'ogbn-mag': '/datasets/gnn/dgldata',
    'igb-full-small': '/datasets/gnn/dataset/IGB',
    'igb-full-medium': '/datasets/gnn/dataset/IGB',
    'igb-full-large': '/datasets/gnn/dataset/IGB',
    'mag240m': '/datasets/gnn/mag240m'
}
reduction_ratio={
    'ogbn-mag': '32,8',
    'igb-full-small': '128,8',
    'igb-full-medium': '128,8',
    'igb-full-large': '128,8',
    'mag240m': '128,8',
}

# python -u preprocess.py --dataset igb-full-medium --out-dir preprocess --root /datasets/gnn/IGB --reduction-level 128,8
for dataset, in itertools.product(datasets):
    cmd = f"python -u ./preprocess.py --dataset {dataset} --out-dir preprocess --root {root[dataset]} --reduction-level {reduction_ratio[dataset]}"
    print(cmd)
    # os.system(cmd)
