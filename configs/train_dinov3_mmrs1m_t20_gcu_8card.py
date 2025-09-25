# DINOv3 + MMRS-1M 8卡分布式训练配置文件 - 燧原T20 GCU版本
# 阶段一：基础模型训练，使用DINOv3-ViT-L/16作为backbone
# 针对MMRS-1M多模态遥感数据集进行优化
# 专门适配燧原T20 GCU计算环境 - 8卡分布式训练

# 导入自定义模块
custom_imports = dict(
    imports=[
        'mmseg_custom.datasets',
        'mmseg_custom.transforms',
        'mmseg_custom.models'  # 只导入自定义模型，避免与mmseg官方模块冲突
    ],
    allow_failed_imports=False
)

# 基础配置
work_dir = './work_dirs/dinov3_mmrs1m_t20_gcu_8card'
exp_name = 'dinov3_mmrs1m_t20_gcu_8card'

# 数据集配置
dataset_type = 'MMRS1MDataset'
data_root = '/workspace/data/mmrs1m/data'  # T20服务器路径
local_data_root = '/Users/barryzhang/myDev3/MapSage_V5/data'  # 本地开发路径

# 图像配置
img_size = (512, 512)
crop_size = (512, 512)
num_classes = 7  # MMRS-1M的类别数

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

# 数据预处理器
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],  # ImageNet统计值
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

# DINOv3-ViT-L/16 模型配置
model = dict(
    type='CustomEncoderDecoder',  # 使用自定义的EncoderDecoder
    data_preprocessor=data_preprocessor,
    
    # DINOv3 backbone
    backbone=dict(
        type='DINOv3ViT',
        arch='large',
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        with_cls_token=True,
        output_cls_token=False,
        interpolate_mode='bicubic',
        out_indices=(23,),  # 输出最后一层
        final_norm=True,
        drop_path_rate=0.1,
        init_cfg=None  # 移除预训练权重配置，使用默认初始化
    ),
    
    # 解码头
    decode_head=dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=-1,
        img_size=img_size,
        embed_dims=1024,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        num_conv=2,
        upsampling_method='bilinear',
        num_upsample_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=None  # 可以根据数据集类别分布调整
        )
    ),
    
    # 辅助头
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4
        )
    ),
    
    # 训练和测试配置
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 数据处理管道
# 🔧 修复的训练管道 - 确保tensor形状一致性
train_pipeline = [
    dict(type='CustomLoadImageFromFile'),
    dict(type='CustomLoadAnnotations'),
    dict(
        type='CustomResize',
        img_scale=img_size,
        keep_ratio=False,  # 🔥 关键修复：禁用keep_ratio确保尺寸一致
        backend='pillow'
    ),
    dict(type='CustomRandomFlip', prob=0.5),
    dict(type='CustomNormalize', **img_norm_cfg),
    dict(
        type='CustomPad', 
        size=crop_size, 
        pad_val=0, 
        seg_pad_val=255,
        pad_to_square=False  # 确保填充到指定尺寸
    ),
    # 🔥 关键修复：移除ImageToTensor，让PackSegInputs处理所有格式化
    # PackSegInputs会自动处理numpy到tensor的转换并确保正确的批次维度
    dict(
        type='PackSegInputs', 
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction')
    )
]

# 验证管道
val_pipeline = [
    dict(type='CustomLoadImageFromFile'),
    dict(type='CustomLoadAnnotations'),
    dict(
        type='CustomResize',
        img_scale=img_size,
        keep_ratio=True
    ),
    dict(type='CustomNormalize', **img_norm_cfg),
    # 使用标准的PackSegInputs替代CustomCollect
    dict(type='PackSegInputs', meta_keys=('img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor'))
]

# 测试管道
test_pipeline = val_pipeline

# 数据加载器配置 (生产版 - 恢复高性能设置)
train_dataloader = dict(
    batch_size=8,  # 每个GCU的batch size
    num_workers=8,  # 增加worker数量以提升数据加载速度
    persistent_workers=True,  # 开启持久化worker，减少开销
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        task_type='segmentation',
        modality='optical',
        instruction_format=True,
        pipeline=train_pipeline
    )
)

# 验证数据加载器
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        task_type='segmentation',
        modality='optical',
        instruction_format=True,
        pipeline=val_pipeline
    )
)

# 测试数据加载器
test_dataloader = val_dataloader

# 评估器
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore']
)
test_evaluator = val_evaluator

# 优化器配置 (生产版 - 恢复高性能设置)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,  # 恢复正常学习率
        weight_decay=0.05,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    loss_scale='dynamic'
)

# 学习率调度器 (生产版)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000,  # warmup步数
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=100000,  # 总训练步数
        by_epoch=False,
        begin=1000,
        end=100000,
    )
]

# 训练循环配置 (生产版)
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=100000,  # 恢复正常训练步数
    val_begin=1,
    val_interval=5000,  # 每5000步验证一次
    dynamic_intervals=[(90000, 1000)]  # 最后阶段增加验证频率
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 默认钩子
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=500,
        show=False,
        wait_time=0.01,
        backend_args=None
    )
)

# DeepSpeed配置集成 - 修复优化器配置以避免CUDA特定的FusedAdam
deepspeed_config = dict(
    train_batch_size=8,  # 单卡测试时使用8，8卡训练时DeepSpeed会自动调整为64
    train_micro_batch_size_per_gpu=8,
    gradient_accumulation_steps=1,
    
    # 🔧 关键修复：强制禁用FusedAdam，确保GCU环境兼容性
    # 这是解决IndexError: list index out of range的根本方案
    disable_fused_adam=True,  # 强制禁用CUDA专用的FusedAdam优化器
    
    # 🔧 关键修复：使用标准DeepSpeed格式明确指定AdamW优化器
    # 避免DeepSpeed使用CUDA专用的FusedAdam，解决GCU环境下的IndexError
    optimizer={
        "type": "AdamW",  # 标准PyTorch AdamW，兼容GCU硬件
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.05
        }
    },
    
    scheduler={
        "type": "WarmupDecayLR",  # 使用 DeepSpeed 支持的调度器名称
        "params": {
            "total_num_steps": 100000,  # 必需参数：总训练步数
            "warmup_num_steps": 1000,   # warmup 步数
            "warmup_max_lr": 1e-4,      # warmup 最大学习率
            "warmup_min_lr": 1e-6       # warmup 最小学习率
        }
    },
    
    # fp16 配置已完全移除，强制使用 float32 精度
    # 这样可以避免 DeepSpeed 在 GCU 环境下创建 torch.float16 优化器
    # 从而解决 "incompatible tensor type" 错误
    
    # ZeRO优化配置 - 降级到Stage 0以确保最大GCU兼容性
    zero_optimization={
        "stage": 0,  # 降级到 Stage 0，禁用所有内存分区功能，确保最大兼容性
        # Stage 0 使用标准数据并行，类似于PyTorch的DistributedDataParallel
        # 虽然没有内存优化，但与GCU/XLA后端兼容性最好
    },
    
    gradient_clipping=1.0,
    wall_clock_breakdown=False,
    steps_per_print=100
)

# 环境配置 (生产版 - 燧原T20 GCU)
env_cfg = dict(
    cudnn_benchmark=False,  # GCU环境禁用cudnn
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='eccl'),  # 使用燧原ECCL后端
    resource_limit=4096
)

# 模型包装器配置 - 避免分布式训练设备不匹配问题
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=False,
    broadcast_buffers=False,
    device_ids=None,  # 关键：设置为None避免设备不匹配错误
    output_device=None  # 关键：设置为None让DDP使用模型当前设备
)

# 移除device_cfg配置，避免与脚本中的设备管理冲突
# 所有设备配置都由训练脚本动态处理，确保模型正确移动到GCU设备
# device_cfg = dict(
#     type='gcu',  # 指定使用GCU设备
#     device_ids=[0, 1, 2, 3, 4, 5, 6, 7],  # 使用所有8个GCU设备
# )

# 可视化配置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='TensorboardVisBackend',
        save_dir='./work_dirs/dinov3_mmrs1m_t20_gcu_8card/tf_logs'
    )
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 日志配置
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None  # 从头开始训练
resume = False

# 随机性配置
randomness = dict(seed=42)

# 自动学习率缩放 - 8卡分布式训练
auto_scale_lr = dict(
    enable=True,
    base_batch_size=16  # 8 GCUs * 2 batch_size = 16
)

# 多模态配置
multimodal_config = dict(
    modalities=['optical', 'sar', 'infrared'],
    modality_weights=[0.6, 0.3, 0.1],  # 不同模态的权重
    cross_modal_learning=True,
    modal_specific_augmentation=True
)

# 指令配置
instruction_config = dict(
    enable_instruction_tuning=True,
    instruction_templates=[
        "What is the category of this remote sensing image?",
        "Classify this satellite image.",
        "Identify the land cover type in this image.",
        "What type of terrain is shown in this remote sensing data?"
    ],
    response_format='single_word'
)

# 蒸馏配置
distillation_config = dict(
    enable=False,  # 第一阶段不使用蒸馏
    teacher_model=None,
    distill_loss_weight=0.5,
    temperature=4.0
)

# EMA配置 - 移除硬编码设备配置，让训练脚本动态设置# EMA配置 - 8卡训练暂时禁用以避免设备冲突
model_ema_config = dict(
    enable=False,  # 暂时禁用EMA以避免设备冲突
    momentum=0.9999
    # device配置由训练脚本动态设置
)

# 混合精度训练由--amp标志控制，移除配置文件中的fp16设置避免冲突
# fp16 = dict(loss_scale=512.0)  # 已移除，使用训练脚本的--amp标志替代

# 梯度累积配置 (生产版)
custom_hooks = [
    dict(
        type='GradientAccumulationHook',
        accumulation_steps=1,  # 恢复正常梯度累积
        priority='ABOVE_NORMAL'
    )
]

# 梯度累积
accumulative_counts = 1  # 8卡训练不需要梯度累积

print(f"🚀 DINOv3 + MMRS-1M 燧原T20 GCU 8卡分布式训练配置已加载")
print(f"📊 数据集: {dataset_type}")
print(f"🏗️ 模型: DINOv3-ViT-L/16 + VisionTransformerUpHead")
print(f"💾 工作目录: {work_dir}")
print(f"🔄 最大迭代数: {train_cfg['max_iters']}")
batch_size = 2  # 从train_dataloader配置中获取
print(f"📈 批次大小: {batch_size} x 8 cards = {batch_size * 8}")
print(f"🔥 计算环境: 燧原T20 GCU - 8卡分布式训练")
# 设备配置 - 在分布式训练中由训练脚本动态设置
# 这里设置为None，让训练脚本根据local_rank动态分配设备
device = None  # 由训练脚本动态设置为gcu:{local_rank}

print(f"⚙️ 设备配置: 动态分配 - 训练脚本将根据local_rank设置为gcu:{{local_rank}}")