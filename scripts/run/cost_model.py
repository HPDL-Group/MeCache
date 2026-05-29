import argparse
import os
import numpy as np
import math
import time
from collections import defaultdict

import torch as th
from tqdm import tqdm
import pickle
import heapq
import dgl
from dgl.distributed import load_partition_book
from src.load_dataset import load_dataset
from src import load_reductioned_feat,Feat

READONLY=0
LEARNABLE=1
OPTIMIZER=2

ntypes_w_feats_example = {
    'ogbn-mag': 'paper',
    'igb-full-small': 'paper,author,institute,conference,fos,journal',
    'igb-full-medium': 'paper,author,institute,conference,fos,journal',
    'mag240m': 'paper'
}
# srun --reservation=256 python -u cost_model.py --dataset mag240m --model rgcn --root /datasets/gnn/mag240m --budget 6 --worker 4 --ntypes-w-feats paper
# srun --reservation=256 -w gn80 python -u cost_model.py --dataset mag240m --model rgat --root /datasets/gnn/mag240m --budget 15 --worker 4 --ntypes-w-feats paper
# srun --reservation=256 -w gn80 python -u cost_model.py --dataset mag240m --model rgat --root /datasets/gnn/mag240m --budget 12 --worker 4 --reduction-level 128,8 --ntypes-w-feats paper
# srun --reservation=256 -w gn80 python -u cost_model.py --dataset igb-full-large --model rgat --root /datasets/gnn/dataset/IGB --budget 14 --worker 4 --reduction-level 128,8 --ntypes-w-feats paper,author,institute,conference,fos,journal
# srun --reservation=256 -w gn80 python -u cost_model.py --dataset igb-full-large --model rgcn --root /datasets/gnn/dataset/IGB --budget 19 --worker 4 --reduction-level 128,8 --ntypes-w-feats paper,author,institute,conference,fos,journal
# python -u cost_model.py --dataset igb-full-small --model rgcn --root /datasets/gnn/IGB --budget 4 --worker 4 --reduction-level 128,8 --ntypes-w-feats paper,author,institute,conference,fos,journal
# srun python -u cost_model.py --dataset igb-full-small --model rgcn --root /datasets/gnn/dataset/IGB --budget 19 --worker 4 --reduction-level 128,8 --ntypes-w-feats paper,author,institute,conference,fos,journal
# srun python -u cost_model.py --dataset mag240m --model rgat --root /datasets/gnn/mag240m --budget 12 --worker 4 --reduction-level 128,8 --ntypes-w-feats paper
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ogbn-mag")
parser.add_argument("--root", type=str, default="datasets")
parser.add_argument("--out-dir", type=str, default="cache")
parser.add_argument("--budget", type=float, default=4.0)
parser.add_argument("--embed-dim", type=int, default=64)
parser.add_argument("--fan_out", type=str, default="5,10,15")
parser.add_argument("--model", type=str, default="rgcn")
parser.add_argument("--worker", type=int, default=4)
parser.add_argument(
    "--reduction-level", type=str, default=None
)
parser.add_argument(
    "--preprocess_dir", type=str, default='preprocess', help="The path to preprocessed data and json"
)
parser.add_argument(
    "--ntypes-w-feats", type=str, nargs="*", default=[],
    help="Node types with features"
)
parser.add_argument(
    "--pattern", type=str, default='',
    help="option for Figure 10. MeCache-R prefers caching read-only features, MeCache-O prefers caching optimizer states, MeCache-E prefers caching embeddings, and MeCache-EO prefers caching both embeddings and optimizer states.\
        for example, '--pattern EO' stands for MeCache-EO"
)
args = parser.parse_args()
print(args)
# 输入图，输出name类型字典：type={name:type}
def get_type_dict(g,ntypes_w_feats,embed_dim):
    type_dict=defaultdict(int)
    for ntype in g.ntypes:
        if ntype not in ntypes_w_feats:
            g.nodes[ntype].data["feat"] = th.randn(
                g.num_nodes(ntype), embed_dim)
            type_dict[ntype]=LEARNABLE
            type_dict[f"{ntype}-optimizer"]=OPTIMIZER
        else:
            type_dict[ntype]=READONLY
    return type_dict

def prepare_graph(dataset, preprocess_dir, root, reduction_level=None):
    reduced_feat=None
    load_feat=True
    if reduction_level!=None:
        start = time.time()
        dir_name=dataset+"_"+reduction_level.replace(",","-")
        reduced_feat=load_reductioned_feat(os.path.join(preprocess_dir, dataset,dir_name))
        print(f"Load_reductioned_feat time: {time.time() - start:.2f}s")
        load_feat=False
        for ntype, feats in reduced_feat.items():
            for dim in feats.reduction_level:
                print(ntype, dim, feats.feat[dim].dtype, feats.shape(dim))

    start = time.time()
    g, _, target_node_type, _, _ = load_dataset(
        dataset, root, load_feat=load_feat)
    print(g)
    print(f"Load {dataset} time: {time.time() - start:.2f}s")

    train_nid = {target_node_type: g.nodes[target_node_type].data["train_mask"].nonzero(as_tuple=True)[0]}
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    data_loader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=2048 * 8,    # 2048 * 8
        shuffle=True,
        drop_last=True,
        use_uva=False
    )

    return reduced_feat, g, data_loader

def load_data(g, reduced_feat, node_type_dict, input_idx, ntype_read_cost, ntype_write_cost, ntype_accecc_freq, embed_dim, device):
    for ntype in g.ntypes:
        if ntype in input_idx:
            th.cuda.synchronize()
            t0 = time.perf_counter()
            if reduced_feat!=None and node_type_dict[ntype]!=LEARNABLE:
                feats, _=reduced_feat[ntype].get(input_idx[ntype])
                feats = {dim:feat.to(device,non_blocking=True).float() for dim,feat in feats.items()}
            else: #reduced_feat==None or node_type_dict[ntype]==LEARNABLE
                g.nodes[ntype].data["feat"][input_idx[ntype]].to(device)
            th.cuda.synchronize()
            ntype_read_cost[ntype] += time.perf_counter() - t0
            ntype_accecc_freq[ntype] += len(input_idx[ntype])

            if node_type_dict[ntype]==LEARNABLE:
                # simulate write
                new_feat = th.randn(len(input_idx[ntype]), embed_dim, device=device)
                th.cuda.synchronize()
                t0 = time.perf_counter()
                g.nodes[ntype].data["feat"][input_idx[ntype]] = new_feat.cpu()
                th.cuda.synchronize()
                ntype_write_cost[ntype] += time.perf_counter() - t0

# profiler 输入Feat 和 embedding table，输出字典，包括feat大小，读取代价，写回代价
# 输出三个字典：data_shape={name:shape}, read_cost={name:cost}, write_cost={name:cost}
def get_cost_and_hotness(g, data_loader, node_type_dict, embed_dim, reduced_feat=None, epochs=1):
    start=time.time()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # get ntype_shape
    ntype_shape=defaultdict(int)
    for ntype in g.ntypes:
        if node_type_dict[ntype]==LEARNABLE:
            ntype_shape[ntype]=g.nodes[ntype].data["feat"].shape
            ntype_shape[f"{ntype}-optimizer"]=g.nodes[ntype].data["feat"].shape
        else:
            if reduced_feat==None:
                ntype_shape[ntype]=g.nodes[ntype].data["feat"].shape
            else:
                for dim in reduced_feat[ntype].reduction_level:
                    if reduced_feat[ntype].shape(dim)[0]!=0:
                        ntype_shape[ntype]=reduced_feat[ntype].shape(dim)
                        break
                    
    # get ntype_read_cost, ntype_write_cost, ntype_hotness
    ntype_read_cost= defaultdict(float)
    ntype_write_cost= defaultdict(float)
    ntype_accecc_freq=defaultdict(int)
    node_accecc_freq = {ntype: th.zeros(g.number_of_nodes(ntype), dtype=th.int64, device=device) for ntype in g.ntypes}
    num_batches = len(data_loader)
    print("Number of batches: ", num_batches)
    for _ in range(epochs):
        for input_nodes, _, _ in tqdm(data_loader):
            for ntype, nid in input_nodes.items():
                unique_seeds = th.unique(nid)
                if len(unique_seeds) != len(nid):
                    print("Seed nodes have duplicates!")
                node_accecc_freq[ntype][nid] += 1

            load_data(g, reduced_feat, node_type_dict,input_nodes,
                        ntype_read_cost, ntype_write_cost, ntype_accecc_freq, 
                        embed_dim, device)
    
    print(f"Profile time: {time.time() - start:.2f} s")

    print("ntype_read_time",ntype_read_cost)
    print("ntype_write_time",ntype_write_cost)
    print("ntype_accecc_freq",ntype_accecc_freq)
    for ntype in g.ntypes:
        ntype_read_cost[ntype]/=ntype_accecc_freq[ntype]
        ntype_write_cost[ntype]/=ntype_accecc_freq[ntype]
    node_accecc_freq={ntype:counter.cpu()/epochs for ntype,counter in node_accecc_freq.items()}
    return ntype_shape, ntype_read_cost,ntype_write_cost,node_accecc_freq

# now replace `get_reIO` with `new_get_reIO`
# since node_read_freq and node_write_freq is sightly different from node_accecc_freq
def get_reIO(worker, node_type_dict, ntype_read_cost, ntype_write_cost,node_accecc_freq):
    read_IO = {}
    for ntype,t in node_type_dict.items():
        if t==READONLY:
            read_IO[ntype]=ntype_read_cost[ntype]*node_accecc_freq[ntype]
        elif t==LEARNABLE:
            # 新的代价: 算上 all-2-all的代价, 额外代价为(n-1)*freq*a2a_cost, 假设a2a_cost=write_cost, 则后一项变为原来的1/n
            read_IO[ntype]=2*ntype_read_cost[ntype]*node_accecc_freq[ntype]\
                +ntype_write_cost[ntype]*node_accecc_freq[ntype]/worker
            read_IO[ntype]=read_IO[ntype]/worker
        # 这个地方optimizer cost用2倍embedding cost 表示，因此乘2
        elif t==OPTIMIZER:
            ntype_name=ntype.split('-')[0]
            read_IO[ntype]=2*ntype_read_cost[ntype_name]*node_accecc_freq[ntype_name]\
                +2*ntype_write_cost[ntype_name]*node_accecc_freq[ntype_name]
            read_IO[ntype]=read_IO[ntype]/2/worker
        else:
            print("Unknown node type",ntype,t)
    return read_IO

def new_get_reIO(worker, node_type_dict, ntype_read_cost, ntype_write_cost, node_read_freq, node_write_freq):
    read_IO = {}
    for ntype,t in node_type_dict.items():
        if t==READONLY:
            read_IO[ntype]=ntype_read_cost[ntype]*node_read_freq[ntype]
        elif t==LEARNABLE:
            # 新的代价: 算上 all-2-all的代价, 额外代价为(n-1)*freq*a2a_cost, 假设a2a_cost=write_cost, 则后一项变为原来的1/n
            read_IO[ntype]=ntype_read_cost[ntype]*node_read_freq[ntype]\
                +ntype_read_cost[ntype]*node_write_freq[ntype]\
                +ntype_write_cost[ntype]*node_write_freq[ntype]/worker
            read_IO[ntype]=read_IO[ntype]/worker
        # 这个地方optimizer cost用2倍embedding cost 表示，因此乘2
        elif t==OPTIMIZER:
            ntype_name=ntype.split('-')[0]
            read_IO[ntype]=2*ntype_read_cost[ntype_name]*node_write_freq[ntype_name]\
                +2*ntype_write_cost[ntype_name]*node_write_freq[ntype_name]
            read_IO[ntype]=read_IO[ntype]/2/worker
        else:
            print("Unknown node type",ntype,t)
    return read_IO

def process_benefits(benefit, weight, target):
    # 对每个ntype的benefit进行降序排序，并记录对应的idx
    sorted_benefits = {}
    sorted_idxs = {}
    # result=[]
    for ntype in benefit:
        # 假设benefit[ntype]是一个一维张量
        sorted_values, sorted_indices = th.sort(benefit[ntype], descending=True)
        sorted_benefits[ntype] = sorted_values.tolist()  # 转换为列表便于处理
        sorted_idxs[ntype] = sorted_indices.tolist()
    
    # 初始化最大堆，每个元素为(-value, ntype, current_index)
    heap = []
    for ntype in sorted_benefits:
        if sorted_benefits[ntype]:
            current_value = sorted_benefits[ntype][0]
            # 使用负数实现最大堆
            heapq.heappush(heap, (-current_value, ntype, 0))
    
    accumulated = 0
    selected_idxs = {ntype: [] for ntype in benefit}
    
    # 逐步取出最大值，直到满足条件或堆为空
    while heap and accumulated < target:
        neg_value, ntype, index = heapq.heappop(heap)
        current_value = -neg_value
        current_weight = weight[ntype]
        accumulated += current_weight
        # 记录对应的原始索引
        selected_idxs[ntype].append(sorted_idxs[ntype][index])
        # result.append((ntype,sorted_idxs[ntype][index],current_value,current_weight))
        
        # 将该ntype的下一个元素推入堆中（如果存在）
        next_index = index + 1
        if next_index < len(sorted_benefits[ntype]):
            next_value = sorted_benefits[ntype][next_index]
            heapq.heappush(heap, (-next_value, ntype, next_index))
    
    # for item in result:
    #     max_ntype, max_idx, max_bene,max_weight=item
    #     print(f"Node Type: {max_ntype}, Index: {max_idx}, Value: {max_bene}, weight: {max_weight}")
    print("target",target,"accumulated",accumulated)
    # 返回是否达到目标及选中的索引
    return selected_idxs

def get_cached_nodes(worker, budget, node_type_dict,ntype_shape,read_IO, pattern=''):
    print("pattern",pattern)
    start=time.time()
    benefits = {}
    learnable_dtype_size = 4
    readonly_dtype_size = 2 if args.dataset == 'mag240m' else 4
    print("readonly_dtype_size",readonly_dtype_size, "learnable_dtype_size", learnable_dtype_size)
    dtype_size_dict={}
    for ntype,t in node_type_dict.items():
        if t==READONLY:
            dtype_size_dict[ntype]=worker*readonly_dtype_size*ntype_shape[ntype][1]
            benefits[ntype]=read_IO[ntype]/dtype_size_dict[ntype]
            if 'R' in pattern:
                benefits[ntype]*=100000000000000
                print(f"Boosting benefit for {ntype} due to pattern R")
        # 先加一个/worker，表示需要(worker-1)/worker的gpu间同步开销
        elif t==LEARNABLE:
            dtype_size_dict[ntype]=worker*learnable_dtype_size*ntype_shape[ntype][1]
            benefits[ntype]=read_IO[ntype]/dtype_size_dict[ntype]
            if 'E' in pattern:
                benefits[ntype]*=100000000000000
                print(f"Boosting benefit for {ntype} due to pattern E")
        elif t==OPTIMIZER:
            dtype_size_dict[ntype]=learnable_dtype_size*2*ntype_shape[ntype][1]
            benefits[ntype]=read_IO[ntype]/dtype_size_dict[ntype]
            if 'O' in pattern:
                benefits[ntype]*=100000000000000
                print(f"Boosting benefit for {ntype} due to pattern O")
        else:
            print("Unknown node type",ntype,t)
    print("dtype_size_dict",dtype_size_dict)
    print("benefit",benefits)
    selected_idxs=process_benefits(benefits, dtype_size_dict, worker*budget)
    print(f"Calculating cached nodes time: {time.time() - start:.2f} s")
    return selected_idxs
    

def get_cached_nodes1(worker, budget, node_type_dict, ntype_shape, read_IO):
    start=time.time()
    benefit = {}
    learnable_dtype_size = 4
    readonly_dtype_size = 2 if args.dataset == 'mag240m' else 4
    print("readonly_dtype_size",readonly_dtype_size, "learnable_dtype_size", learnable_dtype_size)
    dtype_size_dict=defaultdict(int)
    for ntype,t in node_type_dict.items():
        if t==READONLY:
            dtype_size_dict[ntype]=worker*readonly_dtype_size*ntype_shape[ntype][1]
            benefit[ntype]=read_IO[ntype]/dtype_size_dict[ntype]
        elif t==LEARNABLE:
            dtype_size_dict[ntype]=worker*learnable_dtype_size*ntype_shape[ntype][1]
            benefit[ntype]=read_IO[ntype]/dtype_size_dict[ntype]
        elif t==OPTIMIZER:
            dtype_size_dict[ntype]=learnable_dtype_size*2*ntype_shape[ntype][1]
            benefit[ntype]=read_IO[ntype]/dtype_size_dict[ntype]
        else:
            print("Unknown node type",ntype,t)
    
    print("dtype_size_dict",dtype_size_dict)
    print("benefit",benefit)

    # sort and get topk
    # 合并张量并生成ID列表
    new_tensor = []
    idx = []
    weight_list=[]
    for ntype, old_tensor in benefit.items():
        # 获取当前张量的长度
        length = old_tensor.size(0)
        # 生成ID列表，格式为 "{id}_{ntype}"
        current_ids = [f"{i},{ntype}" for i in range(length)]
        # 将当前张量的数据和ID添加到新列表中
        new_tensor.append(old_tensor)
        idx.extend(current_ids)
        # weight list
        weight_list.append(th.full((length,), dtype_size_dict[ntype]))

    # 合并所有张量为一个新张量
    new_tensor = th.cat(new_tensor)
    weight_list = th.cat(weight_list)
    # descending=True, to highest to lowest
    sorted_beni, sorted_indices = th.sort(new_tensor,descending=True)
    sorted_weight=weight_list[sorted_indices]
    weight_cumsum=th.cumsum(sorted_weight, dim=0)
    cache_idx_length=th.searchsorted(weight_cumsum, worker*budget, side='right') - 1
    sorted_idx = [idx[i] for i in sorted_indices][:cache_idx_length]

    print("budget:",worker*budget/1024/1024/1024, "total", th.sum(sorted_weight)/1024/1024/1024)
    print("sorted_weight.shape",sorted_weight.shape,"sorted_idx.shape", len(sorted_idx))
    # print("cached weight:",sum(sorted_weight[:cache_idx_length]/1024/1024/1024))
    # print("sorted_beni: ",sorted_beni)
    # print("sorted_weight: ",sorted_weight)
    # print(f"sorted_idx: {sorted_idx[:10]} ... {sorted_idx[-10:]}")

    # get cache idx
    cached_nodes_dict=defaultdict(list)
    for idx in sorted_idx:
        list_idx,ntype=idx.split(',')
        cached_nodes_dict[ntype].append(int(list_idx))

    print(f"Calculating cached nodes time: {time.time() - start:.2f} s")

    for ntype,t in node_type_dict.items():
        if ntype not in cached_nodes_dict.keys():
            cached_nodes_dict[ntype]=[]
    return cached_nodes_dict

def save_cached_node(cached_nodes_dict, dataset,method,model,worker,out_dir):
    dir = f"{out_dir}/drgnn/{dataset}_{method}_{model}_{worker}"
    os.makedirs(dir, exist_ok=True)
    _sum=0
    for ntype, cached_nodes in cached_nodes_dict.items():
        _sum+=len(cached_nodes)
        count=np.array(cached_nodes)
        print(f"cached {len(count)} {ntype} nodes")
        np.save(os.path.join(dir, f'{ntype}.npy'), count)
    print("npy dumped!",dir)
    
    print("Cache size ratio:",{ntype:len(cached_nodes)/_sum for ntype,cached_nodes in cached_nodes_dict.items()})

# 假设每个顶点在从每个训练节点被采样到的概率为p，每个训练节点的采样独立同分布
# batch_size为b，gpu数量为worker，总数量为batch中含有的训练节点独立同分布
# 显然每个顶点在某个batch中被采样的次数服从伯努利分布，概率为1 - (1 - p)^b
# 那么节点d epoch中总出现频率服从二项分布，Y∼Binomial(m,q)，m=n/b，q=1 - (1 - p)^b
# 现在已经知道读的node_read_freq, batchsize 为b, 总batch数量为m
# 求node_write_freq，即batch size为worker*b下的情况
# 对每个节点，新的频率为(m/w)*(1-(1-x/m)^w)
def get_node_write_freq(node_read_freq, m, worker):
    node_write_freq = {}
    for node_type, freq_tensor in node_read_freq.items():
        # Convert to float for precise calculation
        x = freq_tensor.float()
        # Calculate the ratio x/m
        ratio = x / m
        # Compute (1 - ratio)^worker
        power_val = (1 - ratio) ** worker
        # Compute new frequency: (m/worker) * (1 - power_val)
        new_freq_float = (m / worker) * (1 - power_val)
        # Round to nearest integer and convert to long
        new_freq = th.round(new_freq_float).long()
        node_write_freq[node_type] = new_freq
    #     print(node_type,ratio,power_val,new_freq_float,new_freq)
    # exit()
    return node_write_freq


# 2个部件，profiler和cost model
# 输入图，输出name类型字典：type={name:type}
# 输入 embedding table, 输入feat，输入
# profiler 输入Feat 和 embedding table，输出字典，包括feat大小，读取代价，写回代价
# 输出三个字典：data_shape={name:shape}, read_cost={name:cost}, write_cost={name:cost}
# cost model 输入worker数量，
if __name__ == "__main__":
    ntypes_w_feats = args.ntypes_w_feats[0].split(",") if len(args.ntypes_w_feats) > 0 else []
    budget = args.budget * 1024 * 1024 * 1024 # budget GB
    fanouts = [int(fanout) for fanout in args.fan_out.split(",")]
    embed_dim=args.embed_dim
    dataset=args.dataset
    preprocess_dir=args.preprocess_dir
    root=args.root
    reduction_level=args.reduction_level
    worker=args.worker
    model=args.model
    out_dir=args.out_dir
    pattern=args.pattern

    file_name=os.path.join(f"{out_dir}/drgnn", f'{dataset}_{reduction_level}.pkl')
    # if not os.path.isfile(file_name):
    if not os.path.isfile(file_name):
        print(f"{file_name} not exist! Start profilling......")

        reduced_feat, g, data_loader=prepare_graph(dataset, preprocess_dir, root, reduction_level)

        node_type_dict=get_type_dict(g,ntypes_w_feats,embed_dim)
        print("node_type_dict: ", node_type_dict)
        
        ntype_shape, ntype_read_cost,ntype_write_cost,node_accecc_freq = \
            get_cost_and_hotness(g, data_loader, node_type_dict, embed_dim, reduced_feat,epochs=5)

        num_batches = len(data_loader)
        # pickle dump
        with open(file_name,'wb') as f:
            pickle.dump([node_type_dict,ntype_shape,ntype_read_cost,ntype_write_cost,node_accecc_freq,num_batches],f, protocol=4)
            print(f'{file_name} dumped!')

    start=time.time()
    with open(file_name,'rb') as f:
        print(f"{file_name} exist! Start reading......")
        node_type_dict,ntype_shape,ntype_read_cost,ntype_write_cost,node_accecc_freq,num_batches=pickle.load(f)
    
    print("ntype_shape",ntype_shape)
    print("ntype_read_cost",ntype_read_cost)
    print("ntype_write_cost",ntype_write_cost)
    print("num_batches",num_batches)
    print("node_accecc_freq",node_accecc_freq)
    node_write_freq=get_node_write_freq(node_accecc_freq, num_batches, worker)
    print("node_write_freq",node_write_freq)
    # read_IO = get_reIO(worker, node_type_dict, ntype_read_cost,ntype_write_cost,node_accecc_freq)
    read_IO = new_get_reIO(worker, node_type_dict, ntype_read_cost,ntype_write_cost,node_accecc_freq,node_write_freq)

    print("node_type_dict: ", node_type_dict)
    print("ntype_shape: ", ntype_shape)
    print("read_IO: ", read_IO)

    print(f"Load and calculating read_IO time: {time.time() - start:.2f} s")
    
    
    cached_nodes_dict=get_cached_nodes(worker, budget, node_type_dict,ntype_shape,read_IO,pattern)
    # print("cached_nodes_dict",cached_nodes_dict)

    save_cached_node(cached_nodes_dict, dataset, "miss_penalty", model, worker, out_dir)

