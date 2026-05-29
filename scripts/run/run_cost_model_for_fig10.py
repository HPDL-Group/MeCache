import os
import itertools

num_workers=['2']
models = ['rgcn']
datasets = ['mag240m']
patterns = ['R', 'O', 'E', 'EO']
root={
    'mag240m': '/datasets/gnn/mag240m'
}
ntypes_w_feats = {
    'mag240m': ['paper']
}
reduction_ratio={
    'mag240m': '128,8',
}
budget={
    'mag240m': {'rgat':4,'rgcn':4},
}

# python -u cost_model.py --dataset mag240m --model rgcn --root /datasets/gnn/mag240m --budget 6 --worker 4 --ntypes-w-feats paper
for num_worker, dataset, model, pattern in itertools.product(num_workers, datasets, models, patterns):
    print(f"Running {model} on {dataset}")
    cmd = f"python -u ./cost_model.py --dataset {dataset} --model {model} --root {root[dataset]} --worker {num_worker}"
    if len(ntypes_w_feats[dataset]) > 0:
        cmd += f" --ntypes-w-feats {','.join(ntypes_w_feats[dataset])}"
    cmd += f" --reduction-level {reduction_ratio[dataset]} --patterns {pattern} --budget {budget[dataset][model]}"
    print(cmd)
    # os.system(cmd)
