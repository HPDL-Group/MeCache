#!/bin/bash
#SBATCH -N 1
#SBATCH -J dist_dgl
#SBATCH -o ./log/slurm/job.%j.out

# 记得运行该脚本的环境和实际运行的环境一样

module add cuda/11.7
srun sleep 10d
