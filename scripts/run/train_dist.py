import argparse
import os
import gc
import socket
import sys
import time
from contextlib import contextmanager
from typing import List
from datetime import timedelta

import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from dgl.distributed import load_partition_book
from dgl.distributed.graph_partition_book import NodePartitionPolicy
from collections import defaultdict
from src import get_model, DistEmbedLayer, GPUCache
from src.utils import nvidia_smi_usage,gpu_capacity
from src.distributed import copy_graph_to_shared_mem, get_graph_from_shared_mem,initialize
from src.load_dataset import load_dataset
from src.optim import SparseAdam
from memory_profiler import profile, memory_usage
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

parser = argparse.ArgumentParser(description="GCN")
parser.add_argument("--graph_name", type=str, help="graph name")
parser.add_argument("--root", type=str, help="dataset root")
parser.add_argument("--model", type=str, help="model name")
parser.add_argument("--id", type=int, help="the partition id")
parser.add_argument(
    "--ip_config", type=str, help="The file for IP configuration"
)
parser.add_argument(
    "--preprocess_dir", type=str, default='preprocess', help="The path to preprocessed data and json"
)
parser.add_argument(
    "--n_classes", type=int, default=0, help="the number of classes"
)
parser.add_argument(
    "--predict_category", type=str, help="predict category"
)
parser.add_argument(
    "--backend",
    type=str,
    default="nccl",
    help="pytorch distributed backend",
)
parser.add_argument(
    "--num_gpus",
    type=int,
    default=-1,
    help="the number of GPU device. Use -1 for CPU training",
)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--num_hidden", type=int, default=16)
parser.add_argument("--embed_dim", type=int, default=64)
# parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--fan_out", type=str, default="25,20")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--batch_size_eval", type=int, default=512)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--eval_every", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument(
    "--local_rank", type=int, help="get rank of the process"
)
parser.add_argument(
    "--standalone", action="store_true", help="run in the standalone mode"
)
parser.add_argument(
    "--pad-data",
    default=False,
    action="store_true",
    help="Pad train nid to the same length across machine, to ensure num "
    "of batches to be the same.",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Random seed used in the training and validation phase.",
)
parser.add_argument(
    "--dgl-sparse",
    action="store_true",
    help="Whether to use DGL sparse embedding",
)
parser.add_argument(# origin 0.06
    "--sparse-lr", type=float, default=1e-5, help="sparse lr rate"
)
parser.add_argument(
    "--ntypes-w-feats", type=str, nargs="*", default=[],
    help="Node types with features"
)
parser.add_argument(
    "--cache-method", type=str, default='none', help='cache method'
)
parser.add_argument(
    "--no-sampling", action="store_true", help="no sampling"
)
parser.add_argument(
    "--no-test", action="store_true", help="no test"
)
parser.add_argument(
    "--reduction-level", type=str, default=None   # now only support 128,8, need preprocessing
)
parser.add_argument(
    "--use_node_projs", action="store_true", help='project the reduced feature to args.embed_dim'
)
# parser.add_argument(
#     "--cache-update-every", type=int, default=-1   # -1 stand for static cache
# )
args = parser.parse_args()

def set_seed(seed):
    # dgl.seed(seed)
    # th.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # th.use_deterministic_algorithms(True)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)

def create_all_local_groups():
    """create all local groups."""
    ranks = list(range(dist.get_world_size()))
    local_world_size = args.local_world_size
    local_groups = []
    for i in range(len(ranks) // local_world_size):
        local_ranks = ranks[i * local_world_size : (i + 1) * local_world_size]
        local_groups.append(dist.new_group(local_ranks, timeout=timedelta(seconds=18000)))
    
    return local_groups

def load_partition(part_id: int, graph_name: str, local_group: dist.ProcessGroup) -> dgl.DGLGraph:
    local_root_rank = args.machine_rank * args.local_world_size
    if args.local_rank == 0:
        assert local_root_rank == args.rank
        # g = dgl.load_graphs(os.path.join(args.part_dir, f"part{part_id}/graph.dgl"))[0][0]
        if args.reduction_level==None and args.graph_name!='mag240m':
            load_feat=True
        else:
            load_feat=False
        start=time.time()
        g, _, _, _, _ = load_dataset(args.graph_name, args.root, load_feat=load_feat)
        print(f"load dataset takes: {time.time()-start} sec")
        new_g = copy_graph_to_shared_mem(g, graph_name, local_root_rank, local_group)
        del g
    else:
        new_g = get_graph_from_shared_mem(graph_name, local_root_rank, local_group)

    print(f"Rank {args.rank}: loaded graph")
    dist.barrier()
    return new_g

# def load_partition(part_id: int, graph_name: str, local_group: dist.ProcessGroup) -> dgl.DGLGraph:
#     local_root_rank = args.machine_rank * args.local_world_size
#     if args.reduction_level==None and args.graph_name!='mag240m':
#         load_feat=True
#     else:
#         load_feat=False
#     start=time.time()
#     g, _, _, _, _ = load_dataset(args.graph_name, args.root, load_feat=load_feat)
#     print(f"load dataset takes: {time.time()-start} sec")

#     print(f"Rank {args.rank}: loaded graph")
#     dist.barrier()
#     return g

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, embed_layer, g, dataloader, loss_func, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        all_preds = []
        all_labels = []
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # fetch features/labels
            # move to target device
            if not isinstance(blocks[-1].dstdata[dgl.NID], dict):
                ntype = g.ntypes[0]
                input_nodes = {ntype: input_nodes.cpu()}
            batch_inputs = embed_layer(input_nodes)
            blocks = [block.to(device) for block in blocks]
            batch_labels = g.nodes[args.predict_category].data["label"][seeds[args.predict_category]].type(th.LongTensor).to(device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

            all_preds.append(batch_pred)
            all_labels.append(batch_labels)

    all_preds = th.cat(all_preds, dim=0)
    all_labels = th.cat(all_labels, dim=0)
    model.train()
    return compute_acc(all_preds, all_labels), loss_func(all_preds, all_labels)

def run(args, device, data, local_group: dist.ProcessGroup):
    # Unpack data
    train_nid, test_nid, n_classes, g = data
    shuffle = True
    comm_device=device
    if th.distributed.get_backend()=='gloo':
        comm_device=th.device('cpu')

    if args.no_sampling:
        # NB: this is not the best practice to disable sampling
        fanouts = [-1, -1] 
    else:
        fanouts = [int(fanout) for fanout in args.fan_out.split(",")]
    print("fanouts:", fanouts)
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(int(args.num_layers))
    dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    test_dataloader = dgl.dataloading.DataLoader(
        g,
        test_nid,
        sampler,
        batch_size=args.batch_size_eval,
        shuffle=False,
        drop_last=True
    )

    pb = load_partition_book(os.path.join(args.preprocess_dir, args.graph_name, args.graph_name + ".json"), args.machine_rank)[0]
    print(f"args.ntypes_w_feats: {args.ntypes_w_feats}")
    args.ntypes_w_feats = args.ntypes_w_feats[0].split(",") if len(args.ntypes_w_feats) > 0 else []

    print("recycled:",gc.collect())
    print(f"mem_usage={memory_usage()[0]/1024}GB")
    embed_layer = DistEmbedLayer(
        device,
        g,
        args.embed_dim,
        args.ntypes_w_feats,
        args.graph_name,
        dgl_sparse_emb=args.dgl_sparse,
        feat_name="feat",
        partition_book=pb,
        predict_category=args.predict_category,
        cache_method=args.cache_method,
        reduction_level=args.reduction_level,
        use_node_projs=args.use_node_projs,
        args=args
    )
    # embed_layer.broadcast()
    if not args.use_node_projs:
        feat_shape_dict=embed_layer.get_feat_shape_dict()
    else:
        feat_shape_dict=None
    # Define model and optimizer
    # model = get_model(args.model, g, args.predict_category, args.embed_dim, args.num_hidden, n_classes, 
    #                   args.num_layers, args.dropout)
    model = get_model(args.model, g, args.predict_category, args.embed_dim, args.num_hidden, n_classes, 
                len(fanouts), args.dropout,feat_shape_dict)

    # if os.path.exists('model.pth'):
    #     model.load_state_dict(th.load('model.pth'))
    # else:
    #     th.save(model.state_dict(), 'model.pth')
    
    # if os.path.exists('embed_layer.pth'):
    #     embed_layer.load_state_dict(th.load('embed_layer.pth'))
    # else:
    #     th.save(embed_layer.state_dict(), 'embed_layer.pth')
    # if args.rank==0:
    #     th.save(embed_layer.node_projs.state_dict(), 'embed_layer.pth')

    model = model.to(device)

    model.broadcast()

    # if args.rank==0:
    #     print("model.state_dict()",model.state_dict())
    #     print("embed_layer.state_dict()",embed_layer.state_dict())

    if args.num_gpus == -1:
         model = th.nn.parallel.DistributedDataParallel(
            model,
            # NB: the part of the model is distributed in GPUs of the local machine 
            process_group=local_group, 
            find_unused_parameters=False,
        )
    else:
        model = th.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # NB: the part of the model is distributed in GPUs of the local machine 
            process_group=local_group, 
            find_unused_parameters=False,
        )
    print(f"Rank {args.rank}: created model")

    # activation_book=get_feas_by_hook(model)

    # If there are dense parameters in the embedding layer
    # or we use Pytorch saprse embeddings.
    # embed_layer = embed_layer.to(device)
    if len(embed_layer.node_projs) > 0 or not args.dgl_sparse:
        print(f"Rank {args.rank}: before embed_layer DDP")
        embed_layer = embed_layer.to(device)
        if args.num_gpus == -1:
            embed_layer = th.nn.parallel.DistributedDataParallel(
                embed_layer, 
                find_unused_parameters=False, 
                process_group=local_group if args.graph_name in ["igb-het", "donor"] else None
            )
        else:
            embed_layer = th.nn.parallel.DistributedDataParallel(
                embed_layer, device_ids=[device], output_device=device, 
                find_unused_parameters=False, 
                process_group=local_group if args.graph_name in ["igb-het", "donor"] else None
            )
        print(f"Rank {args.rank}: after embed_layer DDP")

    print("recycled:",gc.collect())
    print(f"mem_usage={memory_usage()[0]/1024}GB")
    if isinstance(embed_layer, nn.parallel.DistributedDataParallel):
        embed_layer_module = embed_layer.module
    else:
        embed_layer_module = embed_layer

    if args.dgl_sparse and len(embed_layer_module.node_embeds.keys()) > 0:
        emb_params=list(embed_layer_module.node_embeds.values())
        if args.cache_method!='none':
            optimizer_cache={}
            def init_emb(shape, dtype):
                print("init emb shape: ", shape)
                arr = th.zeros(shape, dtype=dtype)
                return arr
            for param in emb_params:
                cache_name=param.name
                # cache_name = node_emb_field_of_study
                ntype=cache_name.split('node_emb_')[-1]
                cache_nodes = np.load(f"cache/drgnn/{args.graph_name}_{args.cache_method}_{args.model}_{args.num_gpus}/{ntype}-optimizer.npy")
                # optimizer使用非复制cache，不写直达 cache_nodes % self._local_world_size == self._gpu_id
                local_cache_nodes = cache_nodes[cache_nodes % int(os.environ['LOCAL_WORLD_SIZE']) == dist.get_rank()]
                # print(f"Rank {dist.get_rank()} local_cache_nodes = {local_cache_nodes}")
                # local_cache_nodes=cache_nodes[:int(len(cache_nodes)/int(os.environ['LOCAL_WORLD_SIZE']))]
                # local_cache_nodes=cache_nodes
                part_policy = NodePartitionPolicy(pb, ntype=ntype)
                gpu_cache = GPUCache(len(local_cache_nodes), part_policy.get_size(), args.embed_dim, th.float32, device, write_through=False)
                gpu_cache.init_cache(local_cache_nodes, g, ntype, init_func=init_emb)
                optimizer_cache[cache_name]=gpu_cache
            # print('optimizer_cache',optimizer_cache)
        else:
            optimizer_cache=None
        emb_optimizer = SparseAdam(
            emb_params, lr=args.sparse_lr, cache=optimizer_cache
        )
        print(f"Rank {args.rank} optimize DGL sparse embedding: {embed_layer_module.node_embeds.keys()}")
        all_params = list(model.parameters()) + list(embed_layer_module.node_projs.parameters())
    else:
        emb_optimizer = None
        all_params = list(model.parameters()) + list(embed_layer_module.node_projs.parameters())

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)

    optimizer = optim.Adam(all_params, lr=args.lr)
    print(f"Rank {args.rank} optimize DGL dense embedding: {embed_layer_module.node_projs.keys()}")

    # Training loop
    print("recycled:",gc.collect())
    iter_tput = []
    epoch = 0
    print(f"Rank {args.rank}: start training")
    for epoch in range(args.num_epochs):
        # log history embedding before every epoch
        # if dist.get_rank()==0:
        #     write_his_embed(embed_layer_module.node_embeds, epoch, args, './log/init_emb')
        tic = time.time()

        sample_time = 0
        feat_copy_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        emb_update_time = 0
        # cache_update_time = 0
        # cache_all_reduce_time=0
        # cache_update_count = 0

        emb_update_breakdown = th.zeros(5, dtype=th.float32)
        emb_update_tot_time = th.zeros(1, dtype=th.float32)

        tot_num_seeds = 0
        tot_num_inputs = 0
        tot_num_w_feat = 0 
        tot_num_wo_feat = 0 
        tot_num_inputs_split = defaultdict(int)
        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph
        # as a list of blocks.
        step_time = []
        
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # if args.rank==0:
            #     print("seeds",seeds)
                # print("input_nodes",{k:v.shape for k,v in input_nodes.items()},'\n',input_nodes)
            # if args.rank == 0:
            #     print(f'step {step} starts! mem usage={memory_usage()[0]/1024}GB')
            tic_step = time.time()
            sample_time += tic_step - start

            if isinstance(blocks[-1].dstdata[dgl.NID], dict):
                num_seeds = len(blocks[-1].dstdata[dgl.NID][args.predict_category])
                num_inputs = sum([len(v) for _, v in blocks[0].srcdata[dgl.NID].items()])
                for ntype, v in blocks[0].srcdata[dgl.NID].items():
                    # print("test unique",len(v),len(input_nodes[ntype]))
                    tot_num_inputs_split[ntype]+=len(v)
            else:
                # to dict of node types
                ntype = g.ntypes[0]
                input_nodes = {ntype: input_nodes.cpu()}
                if args.rank==0 and epoch==0:
                    print('input_nodes', input_nodes)
                num_seeds = len(blocks[-1].dstdata[dgl.NID])
                num_inputs = len(blocks[0].srcdata[dgl.NID])
            tot_num_seeds += num_seeds
            tot_num_inputs += num_inputs
            # if args.rank==0:
            #     print(f"step {step}, num_inputs {num_inputs}")
            # move to target device
            # fetch features/labels
            # if th.distributed.get_rank()==1:
            #     print("out!:")
            #     for ntype in input_nodes:
            #         print(ntype,input_nodes[ntype].device)
            # print(f"rank {th.distributed.get_rank()} {ntype} {input_nodes[ntype]}")
            # if th.distributed.get_rank()==1:
            #     print("in!:")
            #     print(f"rank {th.distributed.get_rank()} {ntype} {id(input_nodes[ntype])} {input_nodes[ntype].device} {input_nodes[ntype]}")
            # print("out")
            # for ntype in input_nodes:
            #     node_ids, inv = th.unique(input_nodes[ntype], return_inverse=True, sorted=False)  
            #     print(len(node_ids),len(inv))
            batch_inputs = embed_layer(input_nodes)
            start = time.time()
            blocks = [block.to(device) for block in blocks]
            batch_labels = g.nodes[args.predict_category].data["label"][seeds[args.predict_category]].type(th.LongTensor).to(device)
            feat_copy_time += time.time() - start + embed_layer_module._fetch_feat_time
            tot_num_w_feat += embed_layer_module.num_gen[0]
            tot_num_wo_feat += embed_layer_module.num_gen[1]
            
            # Compute loss and prediction
            start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            forward_end = time.time()
            # if step==0 or step ==1:
            #     viz_graph(batch_pred).render(f"./log/tensorboard/model_rank{args.rank}_step{step}", format="png")

            optimizer.zero_grad()
            if args.dgl_sparse and emb_optimizer is not None:
                emb_optimizer.zero_grad()

            loss.backward()
            # if step==0 or step ==1:
            #     grad_dict={}
            #     for name, parms in model.named_parameters():
            #         grad_dict[name]=parms.grad
            #         # print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
            #         # ' -->grad_value:',parms.grad)
            #     # activation={}
            #     # for hook_tool in activation_book:
            #     #     for key,value in hook_tool.activations.items():
            #     #         activation[key]=value
            #     th.save([batch_inputs,batch_pred,grad_dict,model.state_dict(),batch_labels.clone(),loss.clone(),seeds[args.predict_category].clone(),[block.to('cpu') for block in blocks]],f'./log/init_emb/batch_inputs_{args.rank}_{step}')
            compute_end = time.time()
            forward_time += forward_end - start
            backward_time += compute_end - forward_end

            optimizer.step()
            update_time += time.time() - compute_end
            update_end = time.time()
            if args.dgl_sparse and emb_optimizer is not None:
                emb_optimizer.step(dist=False, group=local_group)
                emb_update_breakdown[0] += emb_optimizer._send_grad_time
                emb_update_breakdown[1] += emb_optimizer._pull_time
                emb_update_breakdown[2] += emb_optimizer._push_time
                emb_update_breakdown[3] += emb_optimizer._h2d_d2h_time
                emb_update_breakdown[4] += emb_optimizer._comp_time
                emb_update_tot_time[0] += emb_optimizer._tot_time
            elif emb_optimizer is not None:
                emb_optimizer.step()
            emb_update_time += time.time() - update_end
            emb_update_end = time.time()
            # update cache, extra data transfer and comm for higher cache hit rate 
            # if args.cache_method!='none' and args.cache_update_every != -1 and step % args.cache_update_every==0:
            #     cache_counter=embed_layer_module.all_reduce_counter()
            #     cache_all_reduce_time += time.time() - emb_update_end
            #     cache_all_reduce_end = time.time()
            #     cache_update_count += embed_layer_module.update_cache()
            #     emb_optimizer.update_cache(cache_counter)
            #     cache_update_time += time.time() - cache_all_reduce_end

            step_t = time.time() - tic_step
            step_time.append(step_t)
            iter_tput.append(num_seeds / step_t)
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = (
                    nvidia_smi_usage()
                    if th.cuda.is_available()
                    else 0
                )
                tput = np.mean(iter_tput[3:])
                step_t = np.mean(step_time[-args.log_every :])

                stats = th.tensor(
                    [loss.item(), acc.item(), tput, step_t]).to(device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                stats = stats / dist.get_world_size()
                loss, acc, tput, step_t = stats[0].item(), stats[1].item(), stats[2].item(), stats[3].item()

                if args.rank == 0:
                    print(
                        "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                        "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                        "{:.1f} GB | time {:.3f} s".format(
                            args.rank,
                            epoch,
                            step,
                            loss,
                            acc,
                            tput,
                            gpu_mem_alloc,
                            step_t
                        )
                    )
                gpu_capacity()
            start = time.time()

        toc = time.time()
        tot_time = toc - tic

        # all_time_tensor = th.tensor(
        #     [tot_time, sample_time, feat_copy_time, forward_time, backward_time, update_time, emb_update_time],
        #     device=device,
        # )
        # dist.all_reduce(all_time_tensor, op=dist.ReduceOp.SUM)
        # all_time_tensor /= dist.get_world_size()
        # tot_time, sample_time, feat_copy_time, forward_time, backward_time, update_time, emb_update_time = all_time_tensor.tolist()

        # if args.rank == 0: 
        if True: 
            print(
                "Part {}, Epoch Time(s): {:.4f}, sample: {:.4f}, feat_copy: {:.4f}, "
                "forward: {:.4f}, backward: {:.4f}, model update: {:.4f}, emb update: {:.4f} #seeds: {}, "
                "#inputs: {}, #tot_num_w_feat: {}, #tot_num_wo_feat: {}".format(
                    args.rank,
                    tot_time,
                    sample_time,
                    feat_copy_time,
                    forward_time,
                    backward_time,
                    update_time,
                    emb_update_time,
                    tot_num_seeds,
                    tot_num_inputs,
                    tot_num_w_feat,
                    tot_num_wo_feat,
                )
            )

        print(f"Part {args.rank}, gpu cache read hit rate: {embed_layer_module.cache_read_hit_rate}")
        if args.dgl_sparse and emb_optimizer is not None:
            print(f"Part {args.rank}, gpu cache write hit rate: {emb_optimizer.cache_write_hit_rate}")
        # synchronize emb breakdown
        if args.dgl_sparse and emb_optimizer is not None:
            # dist.all_reduce(emb_update_breakdown, op=dist.ReduceOp.SUM)
            # emb_update_breakdown = emb_update_breakdown / dist.get_world_size()
            send_grad_time, pull_time, push_time, h2d_d2h_time, comp_time = emb_update_breakdown.tolist()
            print("Part {}, send_grad: {:.4f}, pull: {:.4f}, push: {:.4f}, h2d_d2h: {:.4f}, comp: {:.4f}".format(
                args.rank, send_grad_time, pull_time, push_time, h2d_d2h_time, comp_time
            ))

            all_emb_update_tot_time = [th.zeros(1, dtype=th.float32,device=comm_device) for _ in range(dist.get_world_size())]
            emb_update_tot_time=emb_update_tot_time.to(comm_device)
            dist.all_gather(all_emb_update_tot_time, emb_update_tot_time)
            all_emb_update_tot_time = [f"{t.item():.4f}" for t in all_emb_update_tot_time]
            print(f"Part {args.rank}, emb_update_cnt: {emb_optimizer.update_cnt}")
            emb_optimizer.update_cnt={ntype:0 for ntype in emb_optimizer.update_cnt.keys()}
            print("emb_update_tot_time: {}".format(all_emb_update_tot_time))
        print(f"Part {args.rank}, feature_retrieval_cnt: {dict(tot_num_inputs_split)}")
        epoch += 1

        if epoch % args.eval_every == 0 and epoch != 0 and args.no_test!=True:
            start = time.time()
            test_acc, test_loss = evaluate(model, embed_layer, g, test_dataloader, loss_fcn, device)
            # allreduce
            accs = th.tensor([test_acc, ]).to(device)
            dist.all_reduce(accs, op=dist.ReduceOp.SUM)
            accs = accs / dist.get_world_size()
            test_acc = accs[0].item()
            
            if args.rank == 0:
                print(
                    "Test Acc {:.4f}, Test Loss {:.4f}, time: {:.4f}".format(
                        test_acc, test_loss, time.time() - start
                    )
                )

    # if dist.get_rank()==0:
    #         write_his_embed(embed_layer_module.node_embeds, epoch, args, './log/init_emb')

def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    initialize(args.ip_config)

    dist.init_process_group(backend=args.backend, timeout=timedelta(seconds=18000))
    print(f"get world size {dist.get_world_size()}")

    args.rank = dist.get_rank()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    args.machine_rank = int(os.environ["GROUP_RANK"])

    # add environ LOCAL_WORLD_SIZE and GROUP_RANK
    # os.environ["LOCAL_WORLD_SIZE"]=f"{th.cuda.device_count()}"
    # args.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    # os.environ["GROUP_RANK"]=f"{args.rank//args.local_world_size}"
    # args.machine_rank = int(os.environ["GROUP_RANK"])

    print(f"rank {args.rank}, local rank {args.local_rank}, machine rank {args.machine_rank}")

    set_seed(args.seed + args.local_rank)

    local_groups = create_all_local_groups()
    local_group = local_groups[args.machine_rank]

    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = args.rank % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
        th.cuda.set_device(device)
        print("rank", args.rank, "device", th.cuda.current_device())
    n_classes = args.n_classes

    g = load_partition(args.machine_rank, args.graph_name, local_group=local_group)

    train_mask = g.nodes[args.predict_category].data['train_mask']
    nodes = g.nodes(args.predict_category)
    train_nid = {args.predict_category: nodes[train_mask][args.local_rank::args.local_world_size]}
    if args.graph_name == "mag240m" or "mag240m-pca":
        # NB: MAG240M's test set does not have labels
        test_mask = g.nodes[args.predict_category].data['val_mask']
    else:
        test_mask = g.nodes[args.predict_category].data['test_mask']
    test_nid = {args.predict_category: nodes[test_mask][args.local_rank::args.local_world_size]}
    print("train_nid.shape = ",train_nid[args.predict_category].shape)
    print("test_nid.shape = ",test_nid[args.predict_category].shape)

    n_classes = args.n_classes

    data = train_nid, test_nid, n_classes, g
    run(args, device, data, local_group)
    print("parent ends")

if __name__=="__main__":
    print(args)
    main(args)
