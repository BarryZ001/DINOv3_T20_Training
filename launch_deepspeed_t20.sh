#!/bin/bash

# 燧原T20 DeepSpeed训练启动脚本 (生产版)
# 使用标准DeepSpeed命令启动8卡分布式训练

set -e

# 环境配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 燧原GCU环境变量
export TOPS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ECCL_DEBUG=0

# 训练参数
CONFIG_FILE="configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
WORK_DIR="work_dirs/dinov3_mmrs1m_t20_8card"
DEEPSPEED_CONFIG="deepspeed_config.json"

# 创建工作目录
mkdir -p ${WORK_DIR}

# 生成DeepSpeed配置文件
cat > ${DEEPSPEED_CONFIG} << EOF
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 1,
    
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
        "type": "WarmupCosineLR",
        "params": {
            "total_num_steps": 100000,
            "warmup_num_steps": 1000,
            "warmup_max_lr": 1e-4,
            "warmup_min_lr": 1e-6
        }
    },
    
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": false,
    "steps_per_print": 100
}
EOF

echo "启动DeepSpeed训练..."
echo "配置文件: ${CONFIG_FILE}"
echo "工作目录: ${WORK_DIR}"
echo "DeepSpeed配置: ${DEEPSPEED_CONFIG}"

# 使用标准DeepSpeed命令启动训练
deepspeed --num_gpus=8 \
    --master_port=${MASTER_PORT} \
    train_dinov3_deepspeed_8card_gcu.py \
    --config ${CONFIG_FILE} \
    --work-dir ${WORK_DIR} \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --launcher deepspeed

echo "训练完成!"