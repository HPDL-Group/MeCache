import os
import itertools

num_workers=['4']
models = ['rgat','rgcn']
datasets = ['ogbn-mag', 'igb-full-small', 'igb-full-medium', 'mag240m']
cache_methods = ['miss_penalty']
embedding_sizes = 64
hidden_sizes ={
    'ogbn-mag': 64,
    'igb-full-small': 256,
    'igb-full-medium': 256,
    'mag240m': 256,
}
ip_config={
    'ogbn-mag': 'ip_config_gn70.txt',
    'igb-full-small': 'ip_config_gn70.txt',
    'igb-full-medium': 'ip_config_gn70.txt',
    'mag240m': 'ip_config_gn80.txt',
}
root={
    'ogbn-mag': '/datasets/gnn/dgldata',
    'igb-full-small': '/datasets/gnn/dataset/IGB',
    'igb-full-medium': '/datasets/gnn/dataset/IGB',
    'mag240m': '/datasets/gnn/mag240m'
}
predict_category = 'paper'
number_of_classes = {
    'ogbn-mag': 349,
    'igb-full-small': 19,
    'igb-full-medium': 19,
    'mag240m': 153
}
batch_size = {
    'ogbn-mag': 1024,
    'igb-full-small': 1024,
    'igb-full-medium': 1024,
    'mag240m': 1024
}
ntypes_w_feats = {
    'ogbn-mag': ['paper'],
    'igb-full-small': ['paper','author','institute','conference','fos','journal'],
    'igb-full-medium': ['paper','author','institute','conference','fos','journal'],
    'mag240m': ['paper']
}
fan_out={
    'ogbn-mag': '25,25',
    'igb-full-small': '25,20',
    'igb-full-medium': '25,20',
    'mag240m': '5,10,15',
}
epochs={
    'ogbn-mag': 10,
    'igb-full-small': 30,
    'igb-full-medium': 30,
    'mag240m': 1000,
}
reduction_ratio={
    'ogbn-mag': '32,8',
    'igb-full-small': '128,8',
    'igb-full-medium': '128,8',
    'mag240m': '128,8',
}
lr = {
    'ogbn-mag': 1e-2,
    'igb-full-small': 1e-2,
    'igb-full-medium': 1e-2,
    'mag240m': 1e-4
}
sp_lr = {
    'ogbn-mag': 6e-2,
    'igb-full-small': 1e-2,
    'igb-full-medium': 1e-2,
    'mag240m': 1e-5
}
dropout=0.5
use_node_projs='--use_node_projs'

# ./run_gn70.sh rgcn igb-full-small paper 19 1024 paper,author,institute,conference,fos,journal miss_penalty 256 64 5,10,15 30 0.5 8,8 "/datasets/gnn/dgldata" ip_config_gn72.txt 1e-2 1e-5 4
for num_worker, dataset, model, cache_method in itertools.product(num_workers, datasets, models, cache_methods):
    print(f"Running {model} on {dataset}")
    cmd = f"./run_reduction.sh {model} {dataset} {predict_category}"
    cmd += f" {number_of_classes[dataset]} {batch_size[dataset]}"
    if len(ntypes_w_feats[dataset]) > 0:
        cmd += f" {','.join(ntypes_w_feats[dataset])}"
    cmd += f" {cache_method} {hidden_sizes[dataset]} {embedding_sizes} {fan_out[dataset]} {epochs[dataset]} "
    cmd += f"{dropout} {reduction_ratio[dataset]} {root[dataset]} {ip_config[dataset]} {lr[dataset]} {sp_lr[dataset]} {num_worker} "
    cmd += f"{use_node_projs}"
    print(cmd)
    # os.system(cmd)
