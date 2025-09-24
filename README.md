# DINOv3 T20 Training Project

这是一个专门用于在燧原T20服务器上训练DINOv3模型的精简项目。

## 项目结构

```
DINOv3_T20_Training/
├── configs/                    # 训练配置文件
│   └── train_dinov3_mmrs1m_t20_gcu_8card.py
├── scripts/                    # 训练脚本
│   └── train_dinov3_deepspeed_8card_gcu.py
├── mmseg_custom/              # 自定义MMSeg模块
│   ├── datasets/              # 自定义数据集
│   ├── models/                # 自定义模型
│   ├── transforms/            # 数据变换
│   ├── losses/                # 损失函数
│   ├── hooks/                 # 训练钩子
│   └── metrics/               # 评估指标
├── checkpoints/               # 模型检查点
├── datasets/                  # 数据集存放目录
├── docs/                      # 文档
└── requirements.txt           # 依赖包列表
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 准备数据集（将MMRS1M数据集放在datasets/目录下）

3. 在T20服务器上运行训练：

**方法1：使用默认配置文件（推荐）**
```bash
# 使用DeepSpeed启动8卡分布式训练
deepspeed --num_gpus=8 scripts/train_dinov3_deepspeed_8card_gcu.py
```

**方法2：指定配置文件**
```bash
# 使用DeepSpeed启动并指定配置文件
deepspeed --num_gpus=8 scripts/train_dinov3_deepspeed_8card_gcu.py \
    --config configs/train_dinov3_mmrs1m_t20_gcu_8card.py \
    --work-dir ./work_dirs/dinov3_training \
    --steps 10
```

**方法3：直接运行（单卡测试）**
```bash
python scripts/train_dinov3_deepspeed_8card_gcu.py \
    --config configs/train_dinov3_mmrs1m_t20_gcu_8card.py \
    --work-dir ./work_dirs/dinov3_training \
    --steps 10
```

## 特性

- 专注于DINOv3模型训练
- 支持燧原T20 GCU分布式训练
- 使用DeepSpeed进行高效训练
- 精简的项目结构，避免不必要的复杂性

## 注意事项

- 本项目专门为T20服务器优化
- 需要预先安装torch-gcu等T20专用包
- 确保ECCL环境正确配置
