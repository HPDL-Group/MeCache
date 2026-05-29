import os
import itertools

num_workers=['4','2']
models = ['rgat']
datasets = ['ogbn-mag', 'mag240m']
root={
    'ogbn-mag': '/datasets/gnn/dgldata',
    'igb-full-small': '/datasets/gnn/dataset/IGB',
    'igb-full-medium': '/datasets/gnn/dataset/IGB',
    'igb-full-large': '/datasets/gnn/dataset/IGB',
    'mag240m': '/datasets/gnn/mag240m'
}
ntypes_w_feats = {
    'ogbn-mag': ['paper'],
    'igb-full-small': ['paper','author','institute','conference','fos','journal'],
    'igb-full-medium': ['paper','author','institute','conference','fos','journal'],
    'igb-full-large': ['paper','author','institute','conference','fos','journal'],
    'mag240m': ['paper']
}
budget={
    'igb-full-medium': {'rgat':6,'rgcn':14},
    'igb-full-small': {'rgat':2,'rgcn':9},
    'igb-full-large': {'rgat':8,'rgcn':9},
    'mag240m': {'rgat':13,'rgcn':18},
    'ogbn-mag': {'rgat':16,'rgcn':20},
}

# python -u cost_model.py --dataset mag240m --model rgcn --root /datasets/gnn/mag240m --budget 6 --worker 4 --reduction-level 128,8 --ntypes-w-feats paper
for num_worker, dataset, model in itertools.product(num_workers, datasets, models):
    print(f"Running {model} on {dataset}")
    cmd = f"python -u ./cost_model.py --dataset {dataset} --model {model} --root {root[dataset]} --worker {num_worker}"
    if len(ntypes_w_feats[dataset]) > 0:
        cmd += f" --ntypes-w-feats {','.join(ntypes_w_feats[dataset])}"
    cmd += f" --budget {budget[dataset][model]}"
    print(cmd)
    # os.system(cmd)
