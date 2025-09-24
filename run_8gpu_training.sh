#!/bin/bash

# 8卡分布式训练启动脚本
# 使用DeepSpeed进行DINOv3训练

echo "🚀 启动8卡分布式DINOv3训练..."

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

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8
export NCCL_DEBUG=INFO

echo "📋 训练配置:"
echo "   - GPU数量: 8"
echo "   - 配置文件: $CONFIG_PATH"
echo "   - 训练脚本: $SCRIPT_PATH"
echo "   - 世界大小: $WORLD_SIZE"
echo "   - 主节点: $MASTER_ADDR:$MASTER_PORT"

# 使用deepspeed启动8卡分布式训练
echo "🔥 使用DeepSpeed启动8卡分布式训练..."

deepspeed --num_gpus=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $SCRIPT_PATH \
    --config $CONFIG_PATH \
    --steps 100

echo "✅ 8卡分布式训练启动完成"