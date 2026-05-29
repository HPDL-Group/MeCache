import os
import itertools

num_workers=['4','2','1']
models = ['rgat','rgcn']
datasets = ['ogbn-mag', 'igb-full-small', 'igb-full-medium', 'igb-full-large', 'mag240m']
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
reduction_ratio={
    'ogbn-mag': '32,8',
    'igb-full-small': '128,8',
    'igb-full-medium': '128,8',
    'igb-full-large': '128,8',
    'mag240m': '128,8',
}
budget={
    'igb-full-medium': {'rgat':24,'rgcn':24},
    'igb-full-small': {'rgat':24,'rgcn':24},
    'igb-full-large': {'rgat':8,'rgcn':9},
    'mag240m': {'rgat':12,'rgcn':14},
    'ogbn-mag': {'rgat':24,'rgcn':24},
}

# python -u cost_model.py --dataset mag240m --model rgcn --root /datasets/gnn/mag240m --budget 6 --worker 4 --reduction-level 128,8 --ntypes-w-feats paper
for num_worker, dataset, model in itertools.product(num_workers, datasets, models):
    print(f"Running {model} on {dataset}")
    cmd = f"python -u ./cost_model.py --dataset {dataset} --model {model} --root {root[dataset]} --worker {num_worker}"
    if len(ntypes_w_feats[dataset]) > 0:
        cmd += f" --ntypes-w-feats {','.join(ntypes_w_feats[dataset])}"
    cmd += f" --reduction-level {reduction_ratio[dataset]} --budget {budget[dataset][model]}"
    print(cmd)
    # os.system(cmd)
