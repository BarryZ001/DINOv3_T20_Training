#!/bin/bash

# DINOv3 8卡分布式训练启动脚本 - 燧原T20 GCU
# 使用方法: ./scripts/start_8card_training.sh configs/train_dinov3_mmrs1m_t20_gcu_8card.py

set -e

# 检查参数
if [ $# -eq 0 ]; then
    echo "❌ 错误: 请提供配置文件路径"
    echo "使用方法: $0 <config_file>"
    echo "示例: $0 configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
    exit 1
fi

CONFIG_FILE=$1

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "🚀 启动DINOv3 8卡分布式训练"
echo "📁 配置文件: $CONFIG_FILE"
echo "🔥 计算环境: 燧原T20 GCU - 8卡分布式"

# 设置环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8
export NPROC_PER_NODE=8

# 启动训练
deepspeed --num_gpus=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_dinov3_deepspeed_8card_gcu.py \
    $CONFIG_FILE

echo "✅ 训练启动完成"