import os
import itertools

num_workers=['4','2']
models = ['rgat']
datasets = ['ogbn-mag', 'mag240m']
cache_methods = ['miss_penalty']
embedding_sizes = 64
hidden_sizes ={
    'ogbn-mag': 64,
    'igb-full-small': 256,
    'igb-full-medium': 256,
    'igb-full-large': 256,
    'mag240m': 256,
}
ip_config={
    'ogbn-mag': 'ip_config_gn70.txt',
    'igb-full-small': 'ip_config_gn70.txt',
    'igb-full-medium': 'ip_config_gn70.txt',
    'igb-full-large': 'ip_config_gn70.txt',
    'mag240m': 'ip_config_gn80.txt',
}
root={
    'ogbn-mag': '/datasets/gnn/dgldata',
    'igb-full-small': '/datasets/gnn/dataset/IGB',
    'igb-full-medium': '/datasets/gnn/dataset/IGB',
    'igb-full-large': '/datasets/gnn/dataset/IGB',
    'mag240m': '/datasets/gnn/mag240m'
}
predict_category = 'paper'
number_of_classes = {
    'ogbn-mag': 349,
    'igb-full-small': 19,
    'igb-full-medium': 19,
    'igb-full-large': 19,
    'mag240m': 153
}
batch_size = {
    'ogbn-mag': 1024,
    'igb-full-small': 1024,
    'igb-full-medium': 1024,
    'igb-full-large': 1024,
    'mag240m': 1024
}
ntypes_w_feats = {
    'ogbn-mag': ['paper'],
    'igb-full-small': ['paper','author','institute','conference','fos','journal'],
    'igb-full-medium': ['paper','author','institute','conference','fos','journal'],
    'igb-full-large': ['paper','author','institute','conference','fos','journal'],
    'mag240m': ['paper']
}
fan_out={
    'ogbn-mag': '5,10,15',
    'igb-full-small': '5,10,15',
    'igb-full-medium': '5,10,15',
    'igb-full-large': '5,10,15',
    'mag240m': '5,10,15',
}
epochs={
    'ogbn-mag': 3,
    'igb-full-small': 3,
    'igb-full-medium': 3,
    'igb-full-large': 3,
    'mag240m': 3,
}
reduction_ratio={
    'ogbn-mag': '32,8',
    'igb-full-small': '128,8',
    'igb-full-medium': '128,8',
    'igb-full-large': '128,8',
    'mag240m': '128,8',
}
lr = {
    'ogbn-mag': 1e-2,
    'igb-full-small': 1e-2,
    'igb-full-medium': 1e-2,
    'igb-full-large': 1e-2,
    'mag240m': 1e-4
}
sp_lr = {
    'ogbn-mag': 6e-2,
    'igb-full-small': 1e-2,
    'igb-full-medium': 1e-2,
    'igb-full-large': 1e-2,
    'mag240m': 1e-5
}
dropout=0.5
no_test='--no-test'
use_node_projs='--use_node_projs'

# ./run_gn70.sh rgcn igb-full-small paper 19 1024 paper,author,institute,conference,fos,journal miss_penalty 256 64 5,10,15 30 0.5 8,8 "/datasets/gnn/dgldata" ip_config_gn72.txt 1e-2 1e-5 4
for num_worker, dataset, model, cache_method in itertools.product(num_workers, datasets, models, cache_methods):
    print(f"Running {model} on {dataset}")
    cmd = f"./run_no_reduction.sh {model} {dataset} {predict_category}"
    cmd += f" {number_of_classes[dataset]} {batch_size[dataset]}"
    if len(ntypes_w_feats[dataset]) > 0:
        cmd += f" {','.join(ntypes_w_feats[dataset])}"
    cmd += f" {cache_method} {hidden_sizes[dataset]} {embedding_sizes} {fan_out[dataset]} {epochs[dataset]} "
    cmd += f"{dropout} {reduction_ratio[dataset]} {root[dataset]} {ip_config[dataset]} {lr[dataset]} {sp_lr[dataset]} {num_worker} "
    cmd += f"{use_node_projs} {no_test}"
    print(cmd)
    # os.system(cmd)
