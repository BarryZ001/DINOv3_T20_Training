#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv3 + MMRS-1M 8卡分布式训练脚本 (基于DeepSpeed)
使用DeepSpeed框架进行GCU环境下的分布式训练
"""
import argparse
import os
import sys
import time
import warnings
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置GCU环境变量
os.environ.setdefault('PYTORCH_GCU_ALLOC_CONF', 'backend:topsMallocAsync')
os.environ.setdefault('TORCH_ECCL_AVOID_RECORD_STREAMS', 'false')
os.environ.setdefault('TORCH_ECCL_ASYNC_ERROR_HANDLING', '3')

# 导入必要的库
try:
    import torch
    print(f"✅ PyTorch版本: {torch.__version__}")
    
    import torch_gcu  # 燧原GCU支持
    print(f"✅ torch_gcu可用: {torch_gcu.is_available()}")
    if torch_gcu.is_available():
        print(f"✅ GCU设备数: {torch_gcu.device_count()}")
    else:
        raise RuntimeError("torch_gcu不可用，请检查安装")
    
    import deepspeed
    print(f"✅ DeepSpeed版本: {deepspeed.__version__}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

try:
    from mmengine.config import Config
    from mmengine.registry import MODELS, DATASETS
    print("✅ MMEngine导入成功")
except ImportError as e:
    print(f"❌ MMEngine导入失败: {e}")
    sys.exit(1)

try:
    import mmseg
    from mmseg.models import *
    from mmseg.apis import init_segmentor
    print("✅ MMSegmentation导入成功")
except ImportError as e:
    print(f"❌ MMSegmentation导入失败: {e}")
    sys.exit(1)

# 导入自定义模块
try:
    import mmseg_custom.models
    import mmseg_custom.datasets  # 这会注册MMRS1MDataset到MMEngine的DATASETS
    import mmseg_custom.transforms
    print("✅ 自定义模块导入成功")
        
except ImportError as e:
    print(f"⚠️ 自定义模块导入失败: {e}")
    # 尝试手动导入关键组件
    try:
        from mmseg_custom.datasets.mmrs1m_dataset import MMRS1MDataset
        from mmseg_custom.datasets.loveda_dataset import LoveDADataset
        print("✅ 手动导入数据集类成功")
    except ImportError as e2:
        print(f"❌ 手动导入数据集失败: {e2}")
        sys.exit(1)

def setup_gcu_environment():
    """设置GCU环境 - 已废弃，使用main函数中的简化版本"""
    # 这个函数已被废弃，现在直接在main函数中使用与成功demo相同的方式
    pass

def make_deepspeed_config(config_path="/tmp/ds_config.json"):
    """创建DeepSpeed配置文件 - 优化为8卡分布式训练"""
    cfg = {
        "train_batch_size": 64,  # 8卡总batch size (8 * 8 = 64)
        "train_micro_batch_size_per_gpu": 8,  # 每个GPU的micro batch size
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": False},  # GCU环境下暂时不使用fp16
        "zero_optimization": {
            "stage": 2,  # 使用ZeRO-2优化，适合8卡训练
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 100
            }
        },
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
        "comms_logger": {"enabled": True},  # 启用通信日志以监控8卡通信
        "tensorboard": {"enabled": True, "output_path": "./tensorboard_logs"},
        "flops_profiler": {"enabled": False}
    }
    
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    
    print(f"📝 DeepSpeed 8卡分布式配置文件: {config_path}")
    return config_path

def load_and_validate_config(config_path, work_dir=None):
    """加载和验证配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    print(f"📝 加载配置文件: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # 设置工作目录
    if work_dir is not None:
        cfg.work_dir = work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = './work_dirs/dinov3_deepspeed_8card_gcu'
    
    print(f"📁 工作目录: {cfg.work_dir}")
    
    # 确保工作目录存在
    os.makedirs(cfg.work_dir, exist_ok=True)
    os.makedirs(f"{cfg.work_dir}/logs", exist_ok=True)
    
    # 验证关键配置
    if not hasattr(cfg, 'model'):
        raise ValueError("配置文件缺少model配置")
    
    if not hasattr(cfg, 'train_dataloader'):
        raise ValueError("配置文件缺少train_dataloader配置")
    
    print("✅ 配置文件验证通过")
    return cfg

# 删除custom_collate_fn - 让MMEngine配置文件处理数据格式问题

def build_model_and_dataset(cfg, device_name):
    """构建模型和数据集"""
    print(f"📊 构建数据集: {cfg.train_dataloader.dataset.type}")
    
    # 使用MMEngine的统一构建器构建训练数据集
    train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(f"✅ 训练数据集大小: {len(train_dataset)}")
    
    # 构建验证数据集（如果存在）
    val_dataset = None
    if hasattr(cfg, 'val_dataloader') and cfg.val_dataloader is not None:
        val_dataset = DATASETS.build(cfg.val_dataloader.dataset)
        print(f"✅ 验证数据集大小: {len(val_dataset)}")
    
    # 构建模型
    print(f"🏗️ 构建模型: {cfg.model.type}")
    model = MODELS.build(cfg.model)
    print(f"✅ 模型构建完成")
    
    # 设置设备
    if device_name.startswith('xla'):
        device = torch_gcu.device(device_name)
        # 对于GCU设备，直接使用device_name字符串
        model = model.to(device_name)
    else:
        device = torch.device(device_name)
        model = model.to(device)
    
    print(f"✅ 模型已移动到设备: {device_name}")
    
    return model, train_dataset, val_dataset

def setup_signal_handlers():
    """设置信号处理器，防止死循环"""
    import signal
    
    def signal_handler(signum, frame):
        print(f"\n⚠️ 收到信号 {signum}，正在优雅退出...")
        import sys
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """主函数"""
    # 设置信号处理器
    setup_signal_handlers()
    
    # 创建参数解析器，正确处理DeepSpeed的--local_rank参数
    parser = argparse.ArgumentParser(description='DINOv3 + MMRS-1M 8卡分布式训练')
    parser.add_argument('--config', type=str, 
                       default="configs/train_dinov3_mmrs1m_t20_gcu_8card.py",
                       help='配置文件路径')
    parser.add_argument('--work-dir', type=str, default=None,
                       help='工作目录')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='DeepSpeed自动添加的local rank参数')
    parser.add_argument('--steps', type=int, default=1000,
                       help='训练步数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 解析参数
    args = parser.parse_args()
    
    print(f"📝 使用配置文件: {args.config}")
    print(f"📝 工作目录: {args.work_dir}")
    print(f"📝 Local Rank: {args.local_rank}")
    print(f"📝 训练步数: {args.steps}")

    print("🚀 启动DINOv3 + MMRS-1M 8卡分布式训练")
    print("=" * 60)
    
    # 1. 设置GCU环境和分布式训练 - 使用与成功demo相同的方式
    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", "8"))  # 默认8卡
    
    # 设置设备
    device_name = f"xla:{local_rank}"
    print(f"[PID {os.getpid()}] GCU环境 - local_rank={local_rank}, world_size={world_size}, device={device_name}")
    
    # 设置GCU设备
    if torch_gcu is not None:
        torch_gcu.set_device(local_rank)
        print(f"✅ 设置GCU设备: {local_rank}")
    else:
        print("⚠️ torch_gcu不可用，跳过设备设置")
    
    # 删除手动分布式初始化 - 让DeepSpeed完全接管分布式环境
    print(f"📱 训练模式 - world_size={world_size}, local_rank={local_rank}")
    
    # 2. 加载配置
    cfg = load_and_validate_config(args.config, args.work_dir)
    
    # 构建模型和数据集
    model, train_dataset, val_dataset = build_model_and_dataset(cfg, device_name)
    
    # ==================== 数据加载调试代码 ====================
    print("\n🔍 === 开始数据加载调试 ===")
    # Import the new pseudo_collate function from mmengine.dataset
    # and alias it as collate for convenience.
    from mmengine.dataset import pseudo_collate as collate
    from torch.utils.data import DataLoader
    
    # 使用与训练时完全相同的参数手动创建一个 DataLoader
    debug_dataloader = DataLoader(
        train_dataset,
        batch_size=2,  # 使用配置文件中的 batch size
        shuffle=True,
        num_workers=2,  # 使用少量 worker 以便调试
        collate_fn=collate
    )
    
    # 迭代几个批次并检查内容
    for i, batch in enumerate(debug_dataloader):
        print(f"\n--- 正在检查 Batch #{i} ---")
        print(f"Batch keys: {list(batch.keys())}")
        
        if 'data_samples' not in batch:
            print("❌ 错误: 批处理数据中没有 'data_samples' 键！")
            print(f"实际的键: {list(batch.keys())}")
            continue
        
        print(f"Batch size: {len(batch['data_samples'])}")
        has_labels_count = 0
        
        # 首先检查 data_samples 的类型和结构
        data_samples = batch['data_samples']
        print(f"  data_samples 类型: {type(data_samples)}")
        print(f"  data_samples 内容: {data_samples}")
        
        # 如果 data_samples 是字符串或其他非可迭代类型，跳过处理
        if isinstance(data_samples, str):
            print(f"  ❌ data_samples 是字符串，无法迭代: {data_samples}")
            continue
        elif not hasattr(data_samples, '__iter__'):
            print(f"  ❌ data_samples 不可迭代: {type(data_samples)}")
            continue
        
        # 尝试获取长度
        try:
            samples_length = len(data_samples)
            print(f"  data_samples 长度: {samples_length}")
        except:
            print(f"  ❌ 无法获取 data_samples 长度")
            continue
        
        # 如果是空的，跳过
        if samples_length == 0:
            print(f"  ⚠️ data_samples 为空")
            continue
        
        # 尝试迭代处理样本
        try:
            for j, sample in enumerate(data_samples):
                print(f"  样本 #{j}:")
                print(f"    类型: {type(sample)}")
                print(f"    属性: {dir(sample) if hasattr(sample, '__dict__') else 'N/A'}")
                
                if hasattr(sample, 'gt_sem_seg'):
                    if sample.gt_sem_seg is not None:
                        has_labels_count += 1
                        if hasattr(sample.gt_sem_seg, 'data'):
                            print(f"    ✅ 包含标签, 形状: {sample.gt_sem_seg.data.shape}")
                        else:
                            print(f"    ✅ 包含标签, 类型: {type(sample.gt_sem_seg)}")
                    else:
                        print(f"    ⚠️ gt_sem_seg 为 None")
                else:
                    print(f"    ❌ 缺少 gt_sem_seg 属性")
                    
                # 检查其他可能的标签字段
                if hasattr(sample, 'gt_semantic_seg'):
                    print(f"    发现 gt_semantic_seg: {type(sample.gt_semantic_seg)}")
                
                # 打印样本的所有属性
                if hasattr(sample, '__dict__'):
                    print(f"    所有属性: {list(sample.__dict__.keys())}")
        except Exception as e:
            print(f"  ❌ 迭代 data_samples 时出错: {e}")
            continue
        
        print(f"批次 #{i} 中有效标签数量: {has_labels_count}/{samples_length}")
        
        if has_labels_count == 0:
            print(f"❌❌❌ 致命错误: Batch #{i} 中所有样本都缺少有效标签！这就是导致 torch.stack 失败的原因。")
            print("让我们检查第一个样本的详细信息:")
            try:
                if samples_length > 0:
                    sample = data_samples[0]
                    print(f"样本详细信息: {sample}")
            except Exception as e:
                print(f"❌ 访问第一个样本时出错: {e}")
        
        if i >= 3:  # 只检查前几个批次
            break
    
    print("🔍 === 数据加载调试结束 ===\n")
    # ========================================================
    
    # 4. 创建DeepSpeed配置
    ds_config_path = make_deepspeed_config()

    # 5. 初始化DeepSpeed引擎 (核心变化)
    #    - 不再手动创建 DataLoader 
    #    - 不再手动初始化 torch.distributed 
    #    - 将 train_dataset 直接交给 DeepSpeed 
    print("🔧 初始化DeepSpeed引擎...")
    # Import the new pseudo_collate function from mmengine.dataset
    # and alias it as collate for convenience.
    from mmengine.dataset import pseudo_collate as collate
    
    # 创建优化器 - 使用与成功demo相同的Adam优化器
    print("🔧 创建优化器...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("✅ 优化器创建完成")
    
    # 燧原文档要求：确保模型已经to到device上，然后再使用deepspeed.initialize
    print(f"📍 确认模型设备状态: {next(model.parameters()).device}")
    
    # 检查是否有MPI环境，如果没有则使用单GPU模式
    try:
        print("🔧 尝试初始化DeepSpeed引擎...")
        # 尝试初始化DeepSpeed
        engine, optimizer, train_dataloader, _ = deepspeed.initialize(
            config=ds_config_path,
            model=model,  # 确保 model 已经在 device 上
            optimizer=optimizer,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            collate_fn=collate  # 使用mmcv的collate函数处理DataContainer
        )
        print("✅ DeepSpeed引擎及DataLoader初始化完成")
        use_deepspeed = True
    except Exception as e:
        print(f"⚠️ DeepSpeed初始化失败: {e}")
        print(f"⚠️ 错误类型: {type(e).__name__}")
        print(f"⚠️ 错误详情: {str(e)}")
        print("🔄 回退到单GPU训练模式...")
        
        # 清理可能的DeepSpeed状态
        try:
            import deepspeed
            deepspeed.init_distributed()
        except:
            pass
            
        # 回退到手动创建DataLoader
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.train_dataloader.get('batch_size', 2),
            shuffle=True,
            num_workers=cfg.train_dataloader.get('num_workers', 2),
            pin_memory=False,  # GCU环境下不使用pin_memory
            collate_fn=collate  # 使用mmcv的collate函数处理DataContainer
        )
        engine = None
        use_deepspeed = False
    
    # 7. 显示训练信息
    print(f"📊 训练信息:")
    print(f"   - 配置文件: {args.config}")
    print(f"   - 工作目录: {cfg.work_dir}")
    print(f"   - 设备: {device_name}")
    print(f"   - 世界大小: {world_size}")
    print(f"   - 本地rank: {local_rank}")
    print(f"   - 训练步数: {args.steps}")
    
    # 8. 开始训练 - 使用DeepSpeed优化的训练循环
    try:
        print("🚀 开始训练...")
        print("=" * 60)
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练循环 - DeepSpeed返回的train_dataloader可以直接迭代
        for step, batch in enumerate(train_dataloader):
            if step >= args.steps:
                break
                
            try:
                # 检查批次数据的有效性
                if batch is None:
                    print(f"⚠️ Step {step}: batch为None，跳过")
                    continue
                    
                # DeepSpeed的DataLoader会自动处理数据到设备的移动
                # 但对于XLA后端，手动确认一下更保险
                if use_deepspeed:
                    # 安全地获取inputs和data_samples
                    if isinstance(batch, dict):
                        inputs = batch.get('inputs')
                        data_samples = batch.get('data_samples', [])
                    else:
                        print(f"⚠️ Step {step}: 意外的batch格式: {type(batch)}")
                        continue
                    
                    if inputs is None:
                        print(f"⚠️ Step {step}: inputs为None，跳过")
                        continue
                    
                    inputs = inputs.to(engine.device)
                    if data_samples:
                        data_samples = [s.to(engine.device) for s in data_samples]
                    
                    # 前向传播
                    outputs = engine(inputs, data_samples, mode='loss')
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                    else:
                        print(f"⚠️ Step {step}: 无法获取loss，outputs: {type(outputs)}")
                        continue
                    
                    # 打印信息
                    print(f"[{local_rank}] step={step} loss={loss.item():.6f} device={loss.device}")
                    
                    # 反向传播和更新
                    engine.backward(loss)
                    engine.step()
                    print(f"[{local_rank}] step={step} backward+step ✅")
                else:
                    # 单GPU模式回退处理
                    if isinstance(batch, dict):
                        inputs = batch.get('inputs', batch.get('img'))
                        data_samples = batch.get('data_samples', [])
                        
                        if inputs is not None:
                            inputs = inputs.to(device_name)
                    else:
                        inputs = batch[0].to(device_name) if len(batch) > 0 else None
                        data_samples = batch[1] if len(batch) > 1 else []
                    
                    if inputs is None:
                        print(f"⚠️ Step {step}: inputs为None，跳过")
                        continue
                    
                    optimizer.zero_grad()
                    outputs = model(inputs, data_samples, mode='loss')
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs
                    
                    print(f"[{local_rank}] step={step} loss={loss.item():.6f} device={loss.device}")
                    
                    loss.backward()
                    optimizer.step()
                    print(f"[{local_rank}] step={step} backward+step ✅")
                
                # 添加all-reduce测试（仅在分布式环境下）
                if torch.distributed.is_initialized():
                    test_tensor = torch.tensor([local_rank + 1.0], device=device_name)
                    torch.distributed.all_reduce(test_tensor, op=torch.distributed.ReduceOp.SUM)
                    expected_sum = sum(range(world_size)) + world_size
                    print(f"[{local_rank}] all_reduce sum result: {test_tensor.item()} (should be {expected_sum})")
                else:
                    print(f"[{local_rank}] 分布式环境未初始化，跳过all_reduce测试")
                
                # 添加短暂延迟，与成功demo保持一致
                time.sleep(0.5)
                
            except Exception as step_error:
                print(f"❌ Step {step} 训练出错: {step_error}")
                print(f"❌ 错误类型: {type(step_error).__name__}")
                print(f"❌ 错误详情: {str(step_error)}")
                
                # 如果是关键错误，停止训练
                if isinstance(step_error, (RuntimeError, KeyError, AttributeError)):
                    print("💥 遇到关键错误，停止训练")
                    break
                else:
                    print("⚠️ 非关键错误，继续训练")
                    continue
        
        # 计算训练时间
        end_time = time.time()
        training_time = end_time - start_time
        
        print("=" * 60)
        print("✅ 训练完成!")
        print(f"⏱️ 总训练时间: {training_time:.2f}秒 ({training_time/3600:.2f}小时)")
        print(f"📁 模型保存在: {cfg.work_dir}")
        
        # 保存模型
        if local_rank == 0:
            save_path = f"{cfg.work_dir}/final_model.pth"
            if use_deepspeed and engine is not None:
                torch.save(engine.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"💾 模型已保存: {save_path}")
        
        print("🎉 脚本执行完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()