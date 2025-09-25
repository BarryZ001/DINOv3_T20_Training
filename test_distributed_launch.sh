#!/bin/bash

# 燧原T20 8卡分布式连接测试脚本
# 用于验证分布式环境配置是否正确

set -eu -o pipefail

echo "🧪 燧原T20 8卡分布式连接测试"
echo "=================================="

# 🔧 燧原GCU核心环境变量配置 - 基于官方最佳实践
export ENFLAME_CLUSTER_PARALLEL=true
export ENFLAME_ENABLE_EFP=true
export TOPS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 🔧 燧原分布式训练环境变量 - 官方推荐配置
export OMP_NUM_THREADS=5
export ECCL_ASYNC_DISABLE=false
export ENABLE_RDMA=true
export ECCL_MAX_NCHANNELS=2
export ENFLAME_UMD_FLAGS="mem_alloc_retry_times=1"
export ECCL_RUNTIME_3_0_ENABLE=true
export ENFLAME_PT_EVALUATE_TENSOR_NEEDED=false

# 🔧 禁用CUDA相关设备，强制使用GCU
export CUDA_VISIBLE_DEVICES=""

# 🔧 分布式训练配置
export MASTER_ADDR="localhost"
export MASTER_PORT="29501"
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0

echo "🔧 环境变量配置:"
echo "ENFLAME_CLUSTER_PARALLEL=$ENFLAME_CLUSTER_PARALLEL"
echo "TOPS_VISIBLE_DEVICES=$TOPS_VISIBLE_DEVICES"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"

# 🔧 使用现代化的torchrun启动方式 (替代已弃用的torch.distributed.launch)
echo ""
echo "🚀 启动8卡分布式测试..."
echo "启动命令: torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT test_distributed_gcu.py"
echo ""

torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT test_distributed_gcu.py

echo ""
echo "✅ 分布式连接测试完成!"
