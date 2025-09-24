#!/bin/bash

# DINOv3 + MMRS-1M 8卡分布式训练启动脚本
# 使用标准化命令避免混淆

echo "=== DINOv3 Training Script ==="
echo "开始启动训练..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 检查训练脚本是否存在
if [ ! -f "scripts/train_dinov3_deepspeed_8card_gcu.py" ]; then
    echo "错误: 训练脚本不存在"
    exit 1
fi

# 运行训练脚本
echo "使用Python运行训练脚本..."
python scripts/train_dinov3_deepspeed_8card_gcu.py

echo "训练脚本执行完成"