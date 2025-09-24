#!/usr/bin/env python3
"""
DeepSpeed 初始化测试脚本
用于定位 DeepSpeed 分布式训练初始化阶段的死循环问题
"""

import os
import sys
import time
import signal
import torch

# 尝试导入可能在Mac环境下不可用的模块
try:
    import torch_gcu
    torch_gcu_available = True
except ImportError:
    torch_gcu_available = False
    print("⚠️ torch_gcu 在当前环境下不可用")

try:
    from mmengine.config import Config
    mmengine_config_available = True
except ImportError:
    mmengine_config_available = False
    print("⚠️ mmengine.config 在当前环境下不可用")

try:
    from mmengine.dataset import pseudo_collate as collate
    mmengine_dataset_available = True
except ImportError:
    mmengine_dataset_available = False
    print("⚠️ mmengine.dataset 在当前环境下不可用")

def signal_handler(signum, frame):
    print(f"\n🚨 收到信号 {signum}，正在退出...")
    sys.exit(1)

# 设置信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def test_basic_torch_distributed():
    """测试基础的 torch.distributed 初始化"""
    print("\n=== 测试1: 基础 torch.distributed 初始化 ===")
    
    try:
        # 检查torch.distributed是否可用
        if not hasattr(torch, 'distributed'):
            print("❌ torch.distributed 在当前环境下不可用")
            return False
            
        # 设置环境变量
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        
        print("🔧 环境变量设置完成")
        print(f"   MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
        print(f"   MASTER_PORT: {os.environ.get('MASTER_PORT')}")
        print(f"   RANK: {os.environ.get('RANK')}")
        print(f"   WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
        
        # 检查是否已经初始化
        if torch.distributed.is_initialized():
            print("⚠️ torch.distributed 已经初始化")
            torch.distributed.destroy_process_group()
            print("🔄 已销毁现有进程组")
        
        print("🚀 开始初始化 torch.distributed...")
        
        # 检测可用的后端
        available_backends = []
        if torch.distributed.is_nccl_available():
            available_backends.append('nccl')
        if torch.distributed.is_gloo_available():
            available_backends.append('gloo')
        if torch.distributed.is_mpi_available():
            available_backends.append('mpi')
            
        print(f"🔍 可用的分布式后端: {available_backends}")
        
        # 选择合适的后端
        backend = 'gloo'  # 默认使用gloo，兼容性更好
        if torch_gcu_available and 'nccl' in available_backends:
            # 如果有GCU且NCCL可用，尝试使用nccl
            backend = 'nccl'
            print(f"🔥 检测到GCU环境，尝试使用 {backend} 后端")
        else:
            print(f"🔧 使用 {backend} 后端进行初始化")
        
        # 初始化分布式进程组
        torch.distributed.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=1,
            rank=0,
            timeout=torch.distributed.default_pg_timeout
        )
        
        print("✅ torch.distributed 初始化成功")
        print(f"   Backend: {torch.distributed.get_backend()}")
        print(f"   World Size: {torch.distributed.get_world_size()}")
        print(f"   Rank: {torch.distributed.get_rank()}")
        
        return True
        
    except Exception as e:
        print(f"❌ torch.distributed 初始化失败: {e}")
        return False

def test_deepspeed_import_and_config():
    """测试 DeepSpeed 导入和配置"""
    print("\n=== 测试2: DeepSpeed 导入和配置 ===")
    
    try:
        import deepspeed
        print(f"✅ DeepSpeed 导入成功，版本: {deepspeed.__version__}")
        
        # 测试 DeepSpeed 配置
        ds_config = {
            "train_batch_size": 16,
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
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
                    "warmup_num_steps": 1000
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            }
        }
        
        print("✅ DeepSpeed 配置创建成功")
        return ds_config
        
    except Exception as e:
        print(f"❌ DeepSpeed 导入或配置失败: {e}")
        return None

def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试3: 模型创建 ===")
    
    try:
        # 加载配置
        config_path = '/workspace/code/DINOv3_T20_Training/configs/train_dinov3_mmrs1m_t20_gcu_8card.py'
        cfg = Config.fromfile(config_path)
        
        # 创建模型
        from mmengine.registry import MODELS
        model = MODELS.build(cfg.model)
        print("✅ 模型创建成功")
        
        # 设置设备
        device = torch.device('gcu:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✅ 模型移动到设备: {device}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None, None

def test_deepspeed_initialization():
    """测试 DeepSpeed 初始化"""
    print("\n=== 测试4: DeepSpeed 初始化 ===")
    
    try:
        import deepspeed
        
        # 获取模型和配置
        model, device = test_model_creation()
        if model is None:
            return False
            
        ds_config = test_deepspeed_import_and_config()
        if ds_config is None:
            return False
        
        print("🚀 开始 DeepSpeed 初始化...")
        print("⚠️ 这里可能会出现死循环，等待30秒...")
        
        # 设置超时
        start_time = time.time()
        timeout = 30  # 30秒超时
        
        # 在子进程中尝试初始化
        try:
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                config=ds_config,
                model_parameters=model.parameters()
            )
            
            elapsed = time.time() - start_time
            print(f"✅ DeepSpeed 初始化成功，耗时: {elapsed:.2f}秒")
            return True
            
        except Exception as init_error:
            elapsed = time.time() - start_time
            print(f"❌ DeepSpeed 初始化失败，耗时: {elapsed:.2f}秒")
            print(f"   错误: {init_error}")
            return False
            
    except Exception as e:
        print(f"❌ DeepSpeed 初始化测试失败: {e}")
        return False

def test_launcher_mode():
    """测试不同的启动模式"""
    print("\n=== 测试5: 启动模式检查 ===")
    
    # 检查环境变量
    env_vars = [
        'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT',
        'CUDA_VISIBLE_DEVICES', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK'
    ]
    
    print("🔍 环境变量检查:")
    for var in env_vars:
        value = os.environ.get(var, 'Not Set')
        print(f"   {var}: {value}")
    
    # 检查启动方式
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("🚀 检测到分布式训练环境")
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"   Rank: {rank}")
        print(f"   World Size: {world_size}")
        print(f"   Local Rank: {local_rank}")
        
        if world_size > 1:
            print("⚠️ 多进程分布式训练可能导致死循环")
            return False
    else:
        print("🔧 单进程模式")
        return True

def main():
    print("🔍 DeepSpeed 初始化死循环调试测试")
    print("=" * 50)
    
    # 测试启动模式
    if not test_launcher_mode():
        print("\n❌ 检测到可能导致死循环的分布式环境")
        return
    
    # 测试基础分布式初始化
    if not test_basic_torch_distributed():
        print("\n❌ 基础分布式初始化失败，跳过后续测试")
        return
    
    # 测试 DeepSpeed 初始化
    success = test_deepspeed_initialization()
    
    if success:
        print("\n🎉 DeepSpeed 初始化测试通过！")
        print("💡 问题可能在于多进程启动器或其他环境因素")
    else:
        print("\n❌ DeepSpeed 初始化出现问题")
        print("💡 建议检查:")
        print("   1. GCU 驱动兼容性")
        print("   2. DeepSpeed 版本兼容性")
        print("   3. 分布式通信配置")
        print("   4. 启动器参数设置")

if __name__ == "__main__":
    main()