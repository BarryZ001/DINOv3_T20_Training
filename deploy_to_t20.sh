#!/bin/bash

# DINOv3 T20 Training Project Deployment Script
# 用于将项目上传到T20服务器

SERVER_IP="your_t20_server_ip"
SERVER_USER="your_username"
REMOTE_PATH="/home/$SERVER_USER/DINOv3_T20_Training"

echo "🚀 开始部署DINOv3训练项目到T20服务器..."

# 创建项目压缩包
echo "📦 创建项目压缩包..."
tar -czf dinov3_training.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='checkpoints' \
    --exclude='datasets' \
    .

# 上传到服务器
echo "📤 上传到T20服务器..."
scp dinov3_training.tar.gz $SERVER_USER@$SERVER_IP:~/

# 在服务器上解压
echo "📂 在服务器上解压项目..."
ssh $SERVER_USER@$SERVER_IP << 'REMOTE_COMMANDS'
    # 备份旧版本（如果存在）
    if [ -d "DINOv3_T20_Training" ]; then
        mv DINOv3_T20_Training DINOv3_T20_Training_backup_$(date +%Y%m%d_%H%M%S)
    fi
    
    # 解压新版本
    tar -xzf dinov3_training.tar.gz
    mv . DINOv3_T20_Training
    cd DINOv3_T20_Training
    
    # 创建必要的目录
    mkdir -p checkpoints datasets logs
    
    echo "✅ 项目部署完成！"
    echo "📍 项目路径: $(pwd)"
    echo "🔧 请确保已安装torch-gcu和相关依赖"
REMOTE_COMMANDS

# 清理本地临时文件
rm dinov3_training.tar.gz

echo "�� 部署完成！"
echo "💡 使用方法："
echo "   1. SSH登录T20服务器"
echo "   2. cd DINOv3_T20_Training"
echo "   3. 准备数据集到datasets/目录"
echo "   4. 运行训练: python scripts/train_dinov3_deepspeed_8card_gcu.py configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
