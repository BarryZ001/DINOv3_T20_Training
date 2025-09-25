#!/bin/bash

# 燧原T20 DeepSpeed训练启动脚本 (基于官方最佳实践优化版)
# 严格遵循燧原官方分布式训练规范

set -eu -o pipefail

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
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0

# 🔧 DeepSpeed GCU兼容性环境变量 - 禁用所有CUDA特定组件
export DS_BUILD_FUSED_ADAM=0
export DEEPSPEED_DISABLE_FUSED_ADAM=1
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=0
export DS_BUILD_AIO=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_FUSED_LAMB=0
export DS_BUILD_TRANSFORMER=0

# 训练参数
CONFIG_FILE="configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
WORK_DIR="work_dirs/dinov3_mmrs1m_t20_8card"
DEEPSPEED_CONFIG="deepspeed_config.json"

# 创建工作目录
mkdir -p ${WORK_DIR}

# 🔧 生成燧原GCU优化的DeepSpeed配置文件 - 基于官方最佳实践
cat > ${DEEPSPEED_CONFIG} << EOF
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 1,
    "steps_per_print": 100,
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.05
        }
    },
    
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": 100000,
            "warmup_num_steps": 1000,
            "warmup_max_lr": 1e-4,
            "warmup_min_lr": 1e-6
        }
    },
    
    "zero_optimization": {
        "stage": 0
    },
    
    "fp16": {
        "enabled": false
    },
    
    "bf16": {
        "enabled": false
    },
    
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": true,
    "disable_fused_adam": true
}
EOF

echo "🚀 启动燧原T20 8卡分布式DeepSpeed训练..."
echo "📁 配置文件: ${CONFIG_FILE}"
echo "📁 工作目录: ${WORK_DIR}"
echo "📁 DeepSpeed配置: ${DEEPSPEED_CONFIG}"
echo "🔧 使用燧原官方torch.distributed.launch方式启动"

# 🔧 使用燧原官方推荐的torch.distributed.launch启动方式
# 这是燧原GCU分布式训练的标准启动方法，参考官方llama2示例
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -u -m torch.distributed.launch $DISTRIBUTED_ARGS \
    scripts/train_dinov3_deepspeed_8card_gcu.py \
    --config ${CONFIG_FILE} \
    --work-dir ${WORK_DIR} \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --launcher pytorch \
    --distributed-backend eccl

echo "✅ 训练完成!"