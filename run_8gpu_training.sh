#!/bin/bash

# DINOv3 8卡分布式训练启动脚本 - 燧原T20 GCU版本
# 修复数据流水线问题后的正式训练脚本

echo "🚀 启动DINOv3 8卡分布式训练 - 燧原T20 GCU"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 未找到，请确保Python 3已安装"
    exit 1
fi

# 检查训练脚本是否存在
SCRIPT_PATH="scripts/train_dinov3_deepspeed_8card_gcu.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 训练脚本未找到: $SCRIPT_PATH"
    exit 1
fi

# 检查配置文件是否存在
CONFIG_PATH="configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ 配置文件未找到: $CONFIG_PATH"
    exit 1
fi

# 设置T20 GCU环境变量
export GCU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用GCU而非CUDA
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8
export NPROC_PER_NODE=8

echo "📋 训练配置:"
echo "   - 计算设备: 燧原T20 GCU x 8"
echo "   - 配置文件: $CONFIG_PATH"
echo "   - 训练脚本: $SCRIPT_PATH"
echo "   - 世界大小: $WORLD_SIZE"
echo "   - 主节点: $MASTER_ADDR:$MASTER_PORT"
echo "   - 数据流水线: 已修复 (PackSegInputs + segmentation task)"

# 使用deepspeed启动8卡GCU分布式训练
echo "🔥 使用DeepSpeed启动8卡GCU分布式训练..."

deepspeed --num_gpus=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $SCRIPT_PATH \
    --config $CONFIG_PATH

echo "✅ 8卡GCU分布式训练启动完成"