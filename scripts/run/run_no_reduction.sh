#!/bin/bash

# ./run_gn70.sh rgcn igb-full-small paper 19 1024 paper,author,institute,conference,fos,journal miss_penalty 256 64 5,10,15 30 0.5 8,8 "/datasets/gnn/dgldata" ip_config_gn72.txt 1e-2 1e-5 4
# ./run_reduction.sh hgt igb-full-small paper 19 1024 paper,author,institute,conference,fos,journal none 64 64 25,20 30 0.5 128,8 "/datasets/gnn/dataset/IGB" ip_config_gn71.txt 1e-2 1e-5 4 --use_node_projs
# ./run_reduction.sh rgcn mag240m paper 153 1024 paper miss_penalty 256 64 5,10,15 2 0.5 128,8 "/datasets/gnn/mag240m" ip_config_gn80.txt 1e-4 1e-5 4 --use_node_projs


MODEL=${1:-"rgcn"}
DATASET=${2:-"igb-part-small-pca"}
PREDICT_CATEGORY=${3:-"paper"}
N_CLASSES=${4:-"19"}
BATCH_SIZE=${5:-"1024"}
NTYPES_W_FEATS=${6:-"paper"}
CACHE_METHOD=${7:-"none"}
NUM_HIDDEN=${8:-"256"}
EMBEDDING_SIZE=${9:-"64"}
FANOUT=${10:-"25,20"}
EPOCH=${11:-"10"}
DROPOUT=${12:-"0.2"}
# NUM_LAYERS=${13:-"2"}
REDUCTION_LEVEL=${13:-"128,8"}
ROOT=${14:-"/datasets/gnn/dgldata"}
IP_CONFIG=${15:-"ip_config_gn72.txt"}
LR=${16:-"1e-2"}
SP_LR=${17:-"1e-5"}
GPU=${18:-"4"}
NODE_PROJS=${19:-" "}
NO_TEST=${20:-" "}

cmd="python3 train_dist.py --graph_name ${DATASET} --root ${ROOT} \
--model ${MODEL} --ip_config ${IP_CONFIG} --num_epochs ${EPOCH} \
--batch_size ${BATCH_SIZE} --n_classes ${N_CLASSES} --predict_category ${PREDICT_CATEGORY} \
--eval_every 1 --fan_out ${FANOUT} --num_hidden ${NUM_HIDDEN} --embed_dim ${EMBEDDING_SIZE} \
--preprocess_dir preprocess --num_gpus ${GPU} \
--dgl-sparse --cache-method ${CACHE_METHOD} \
--dropout ${DROPOUT} --batch_size_eval ${BATCH_SIZE} \
--lr ${LR} --sparse-lr ${SP_LR} ${NODE_PROJS} ${NO_TEST}"

# not freebase
if [ "$DATASET" != "freebase" ]; then
    cmd="${cmd} --ntypes-w-feats ${NTYPES_W_FEATS}"
fi
echo $cmd

# output_dir="drgnn_${MODEL}_${DATASET}_${CACHE_METHOD}_${EMBEDDING_SIZE}_${REDUCTION_LEVEL}.log"
output_dir="drgnn_${MODEL}_${DATASET}_${CACHE_METHOD}_${EMBEDDING_SIZE}_no_reduction.log"

if [ "$NO_TEST" != " " ]; then
    output_dir="log/speed/${output_dir} "
else
    output_dir="log/acc/${output_dir} "
fi
echo $output_dir

module add cuda/11.7

# 1 machine
python3 ./launch.py \
    --workspace /gf3/home/jgqj/test_code/hydro \
    --num_trainers ${GPU} \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config preprocess/${DATASET}/${DATASET}.json \
    --ip_config ${IP_CONFIG} \
    "${cmd}" > "${output_dir}"
