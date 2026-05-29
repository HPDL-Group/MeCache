import argparse
import json
import os
import sys
import time
import torch
import dgl

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.load_dataset import load_dataset
from src import write_reductioned_feat,load_reductioned_feat,generate_reductioned_feat,write_mag_reductioned_feat

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ogbn-mag")
parser.add_argument("--out-dir", type=str, default="preprocess")
parser.add_argument("--root", type=str, default="/datasets/gnn/IGB")
parser.add_argument(
    "--reduction-level", type=str, default=None   # now only support 128,8, need preprocessing
)
parser.add_argument("--use_l2", action="store_true", help="use l2pca")
args = parser.parse_args()

def save_meta_data(graph, path):
    """
    save graph meta data as dataset.json
    """
    sorted_ntypes = sorted(graph.ntypes)
    sorted_etypes = [f"{etype[0]}:{etype[1]}:{etype[2]}" for etype in sorted(graph.canonical_etypes)]
    ntypes = {
        ntype: i for i, ntype in enumerate(sorted_ntypes)
    }
    etypes = {
        etype: i for i, etype in enumerate(sorted_etypes)
    }
    # ntype: [[start, end], ...]
    node_map = {}
    for ntype in sorted_ntypes:
        node_map[ntype] = []
    # etype: [[start, end], ...]
    edge_map = {}
    for etype in sorted_etypes:
        edge_map[etype] = []

    num_nodes=0
    num_edges=0
    for ntype in sorted_ntypes:
        start = num_nodes
        if ntype in hg.ntypes:
            end = start + graph.number_of_nodes(ntype)
        else:
            end = start
        node_map[ntype].append([start, end])
        num_nodes = end
    
    for etype in sorted_etypes:
        start = num_edges
        etype = tuple(etype.split(':'))
        if etype in graph.canonical_etypes:
            end = start + graph.number_of_edges(etype=etype)
        else:
            end = start
        etype = f"{etype[0]}:{etype[1]}:{etype[2]}"
        edge_map[etype].append([start, end])
        num_edges = end

    # generate {args.dataset}.json
    data = {
        "graph_name": args.dataset,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_parts": 1,
        "node_map": node_map,
        "edge_map": edge_map,
        "ntypes": ntypes,
        "etypes": etypes,
    }
    json_name=os.path.join(path, f"{args.dataset}.json")
    with open(json_name, "w") as f:
        json.dump(data, f, indent=4)
    print(f"json dumped at {json_name}")


if __name__=="__main__":
    print(args)
    json_path = os.path.join(args.out_dir, args.dataset)
    os.makedirs(json_path, exist_ok=True)
    start_time = time.time()

    # load dataset
    hg, num_classes, target_node_type, list_of_metapaths, reverse_edge_type_prefix = load_dataset(args.dataset, args.root)
    print("Load graph time: {:.4f} s".format(time.time()-start_time))
    
    # save meta data as json
    save_meta_data(hg,json_path)
    
    # level reduction
    if args.reduction_level != None:
        start = time.time()
        reduction_level= [int(level) for level in args.reduction_level.split(",")]
        dir_name=args.dataset+"_"+args.reduction_level.replace(",","-")
        reduction_path = os.path.join(args.out_dir, args.dataset, dir_name)
        os.makedirs(reduction_path, exist_ok=True)
        if 'igb-full-large' in args.dataset:
            print("generate feature for igb-full-large")
            generate_reductioned_feat(hg,target_node_type,reduction_level,reduction_path)
        elif 'mag240m' in args.dataset:
            print("compute reduced feature for mag240m")
            write_mag_reductioned_feat(hg,target_node_type,reduction_level,reduction_path)
        else:
            write_reductioned_feat(hg,target_node_type,reduction_level,reduction_path,args.use_l2)
        print("Reduction time: {:.4f}s".format(time.time()-start))
    
    print("Preprocess time: {:.4f}s".format(time.time()-start_time))

    # feats=load_reductioned_feat(args.out_dir)

    # for ntype, feat in feats.items():
    #     print(ntype,feat)

    # nids={'author':torch.tensor([10,20,30]),
    #     'paper':torch.tensor([400000,400001,600000,600001]),
    #     'institute':torch.tensor([10,20,30]),
    #     'fos':torch.tensor([10,20,30])}
    # for ntype, nid in nids.items():
    #     ret,idx=feats[ntype].get(nid)
    #     print(ret,idx)

# python -u preprocess.py --dataset igb-full-small --out-dir preprocess --root /datasets/gnn/IGB --reduction-level 128,8
# python -u preprocess.py --dataset mag240m --out-dir preprocess --root /datasets/gnn/mag240m --reduction-level 128,8
# python -u preprocess.py --dataset igb-full-medium --out-dir preprocess --root /datasets/gnn/IGB --reduction-level 128,8
# python -u preprocess.py --dataset ogbn-mag --out-dir preprocess --root /datasets/gnn/dgldata --reduction-level 32,8
# python -u preprocess.py --dataset igb-full-large --out-dir preprocess --root /datasets/gnn/dataset/IGB --reduction-level 128,8
# python -u preprocess.py --dataset donor --out-dir preprocess --root /datasets/gnn/dgldata --reduction-level 128,8
