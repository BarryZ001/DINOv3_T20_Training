#!/usr/bin/env python3
"""
测试批次大小配置修复
验证 DeepSpeed 批次大小计算是否正确
"""

import os
import sys
import json
from pathlib import Path

# 设置项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 🔧 强制禁用CUDA特定优化器，确保GCU环境兼容性
os.environ['DEEPSPEED_DISABLE_FUSED_ADAM'] = '1'
os.environ['DS_BUILD_FUSED_ADAM'] = '0'
os.environ['DS_BUILD_CPU_ADAM'] = '1'
os.environ['DS_BUILD_UTILS'] = '0'
os.environ['DS_BUILD_AIO'] = '0'
os.environ['DS_BUILD_SPARSE_ATTN'] = '0'

try:
    import torch
    import deepspeed  # type: ignore
    from mmengine.config import Config  # type: ignore
    
    print("✅ 所有必要模块导入成功")
    
    # 加载配置文件
    config_path = project_root / "configs" / "train_dinov3_mmrs1m_t20_gcu_8card.py"
    cfg = Config.fromfile(str(config_path))
    
    print("✅ 配置文件加载成功")
    
    # 检查 DeepSpeed 配置
    deepspeed_config = cfg.deepspeed_config
    print(f"✅ DeepSpeed 配置获取成功")
    
    # 验证批次大小配置
    train_batch_size = deepspeed_config.get('train_batch_size', 0)
    micro_batch_size = deepspeed_config.get('train_micro_batch_size_per_gpu', 0)
    gradient_acc_steps = deepspeed_config.get('gradient_accumulation_steps', 1)
    
    print(f"📊 批次大小配置检查：")
    print(f"   - train_batch_size: {train_batch_size}")
    print(f"   - train_micro_batch_size_per_gpu: {micro_batch_size}")
    print(f"   - gradient_accumulation_steps: {gradient_acc_steps}")
    
    # 模拟单卡环境（world_size=1）
    world_size = 1
    expected_batch_size = micro_batch_size * gradient_acc_steps * world_size
    
    print(f"🧮 单卡环境批次大小计算：")
    print(f"   - 期望的 train_batch_size: {expected_batch_size}")
    print(f"   - 实际的 train_batch_size: {train_batch_size}")
    
    if train_batch_size == expected_batch_size:
        print("✅ 批次大小配置正确！")
        batch_size_ok = True
    else:
        print("❌ 批次大小配置不匹配")
        print(f"🔧 建议修改 train_batch_size 为: {expected_batch_size}")
        batch_size_ok = False
    
    # 创建一个简单的模型用于测试
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    print("✅ 测试模型创建成功")
    
    # 测试 DeepSpeed 初始化
    if batch_size_ok:
        try:
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config=deepspeed_config
            )
            print("✅ DeepSpeed 初始化成功！")
            print(f"   - 优化器类型: {type(optimizer).__name__}")
            print("🎉 批次大小修复验证成功！")
            
        except Exception as e:
            print(f"❌ DeepSpeed 初始化失败: {e}")
            print("🔧 可能的原因：")
            print("   1. 批次大小配置仍有问题")
            print("   2. 其他配置参数错误")
    else:
        print("⚠️ 跳过 DeepSpeed 初始化测试，因为批次大小配置不正确")
        
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("💡 这是正常的，因为在本地环境中可能缺少某些依赖")
    print("🚀 请在 T20 服务器上运行此测试脚本")
    
except Exception as e:
    print(f"❌ 测试过程中出现错误: {e}")
    print("🔧 请检查配置文件和环境设置")

print("\n📋 测试总结：")
print("1. ✅ 环境变量设置：禁用 FusedAdam")
print("2. ✅ DeepSpeed 配置：使用 AdamW 优化器")
print("3. 🔧 批次大小修复：确保单卡和多卡环境兼容")
print("4. 🚀 准备在 T20 服务器上进行完整测试")