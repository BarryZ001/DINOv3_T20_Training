#!/usr/bin/env python3
"""
DeepSpeed 配置测试脚本 - 用于 T20 GCU 环境
测试 FusedAdam 禁用配置是否正确，避免 IndexError: list index out of range
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Any

# 设置项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 🔧 关键：设置环境变量禁用 FusedAdam
os.environ['DEEPSPEED_DISABLE_FUSED_ADAM'] = '1'
os.environ['DS_BUILD_FUSED_ADAM'] = '0'
os.environ['DS_BUILD_CPU_ADAM'] = '1'
os.environ['DS_BUILD_UTILS'] = '1'

# 类型注解，避免 linter 错误
deepspeed: Optional[Any] = None
Config: Optional[Any] = None

def test_deepspeed_config():
    """测试 DeepSpeed 配置是否正确"""
    
    print("🔧 测试 DeepSpeed 配置...")
    print(f"DEEPSPEED_DISABLE_FUSED_ADAM: {os.environ.get('DEEPSPEED_DISABLE_FUSED_ADAM')}")
    print(f"DS_BUILD_FUSED_ADAM: {os.environ.get('DS_BUILD_FUSED_ADAM')}")
    print(f"DS_BUILD_CPU_ADAM: {os.environ.get('DS_BUILD_CPU_ADAM')}")
    
    try:
        # 尝试导入 DeepSpeed
        try:
            import deepspeed  # type: ignore
            print("✅ DeepSpeed 导入成功")
        except ImportError:
            print("❌ DeepSpeed 未安装，请在 T20 环境中运行")
            return
        
        # 测试配置文件加载
        config_path = project_root / "configs" / "train_dinov3_mmrs1m_t20_gcu_8card.py"
        if config_path.exists():
            print(f"✅ 配置文件存在: {config_path}")
            
            # 尝试加载配置
            try:
                from mmengine.config import Config  # type: ignore
                cfg = Config.fromfile(str(config_path))
            except ImportError:
                print("❌ MMEngine 未安装，请在 T20 环境中运行")
                return
            
            # 检查 deepspeed_config
            if hasattr(cfg, 'deepspeed_config'):
                ds_config = cfg.deepspeed_config
                print("✅ DeepSpeed 配置加载成功")
                
                # 检查关键配置项
                if ds_config.get('disable_fused_adam', False):
                    print("✅ disable_fused_adam = True")
                else:
                    print("❌ disable_fused_adam 未设置或为 False")
                
                if 'optimizer' in ds_config:
                    opt_type = ds_config['optimizer'].get('type', 'Unknown')
                    print(f"✅ 优化器类型: {opt_type}")
                    if opt_type == 'AdamW':
                        print("✅ 使用 AdamW 优化器，兼容 GCU")
                    else:
                        print(f"⚠️ 优化器类型: {opt_type}，请确认兼容性")
                else:
                    print("❌ 未找到优化器配置")
                    
            else:
                print("❌ 未找到 deepspeed_config")
        else:
            print(f"❌ 配置文件不存在: {config_path}")
            
    except ImportError as e:
        print(f"❌ DeepSpeed 导入失败: {e}")
        print("请确保在 T20 环境中运行此脚本")
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")

def create_test_deepspeed_json():
    """创建测试用的 DeepSpeed JSON 配置"""
    
    test_config = {
        "train_batch_size": 64,
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 1,
        "disable_fused_adam": True,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.05
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 200000000,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 200000000,
            "contiguous_gradients": True
        },
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
        "steps_per_print": 100
    }
    
    # 保存测试配置
    test_config_path = project_root / "test_deepspeed_config.json"
    with open(test_config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"✅ 测试配置已保存: {test_config_path}")
    return test_config_path

if __name__ == "__main__":
    print("🚀 DeepSpeed 配置测试开始...")
    print("=" * 50)
    
    # 测试配置
    test_deepspeed_config()
    
    print("\n" + "=" * 50)
    
    # 创建测试 JSON 配置
    test_config_path = create_test_deepspeed_json()
    
    print("\n🔧 使用说明:")
    print("1. 将此脚本上传到 T20 服务器")
    print("2. 在 T20 环境中运行: python test_deepspeed_config.py")
    print("3. 检查输出确认配置正确")
    print("4. 如果测试通过，可以运行完整训练脚本")
    
    print("\n✅ 配置测试脚本准备完成")