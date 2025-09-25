#!/usr/bin/env python3
"""
GCU内存调试脚本
用于诊断PyTorch GCU环境下的内存分配问题和free(): invalid pointer错误
"""

import os
import sys
import torch

# 设置最严格的内存管理环境变量
print("🔧 设置GCU内存管理环境变量...")

# 禁用所有混合精度
os.environ['GCU_DISABLE_AMP'] = '1'
os.environ['GCU_FORCE_FP32'] = '1'
os.environ['TORCH_GCU_DISABLE_AMP'] = '1'
os.environ['TORCH_DISABLE_AMP'] = '1'
os.environ['DEEPSPEED_DISABLE_FP16'] = '1'

# 最保守的内存分配策略
os.environ['PYTORCH_GCU_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.5,expandable_segments:False'
os.environ['GCU_MEMORY_FRACTION'] = '0.6'
os.environ['GCU_ENABLE_LAZY_INIT'] = '0'
os.environ['GCU_SYNC_ALLOC'] = '1'
os.environ['GCU_DISABLE_CACHING'] = '1'

# 额外的调试环境变量
os.environ['GCU_DEBUG'] = '1'
os.environ['TORCH_GCU_DEBUG'] = '1'

print("✅ 环境变量设置完成")

# 强制设置默认数据类型
torch.set_default_dtype(torch.float32)
print(f"✅ 默认数据类型设置为: {torch.get_default_dtype()}")

try:
    import torch_gcu  # type: ignore
    print("✅ torch_gcu 导入成功")
    
    # 检查GCU设备
    if torch_gcu.is_available():
        device_count = torch_gcu.device_count()
        print(f"✅ 检测到 {device_count} 个GCU设备")
        
        for i in range(device_count):
            device = f'gcu:{i}'
            print(f"📊 GCU设备 {i}:")
            
            # 测试基本张量操作
            try:
                # 创建小张量测试
                x = torch.randn(10, 10, dtype=torch.float32, device=device)
                y = torch.randn(10, 10, dtype=torch.float32, device=device)
                z = x + y
                print(f"  ✅ 基本张量操作成功")
                
                # 测试内存分配和释放
                large_tensor = torch.randn(1000, 1000, dtype=torch.float32, device=device)
                del large_tensor
                torch_gcu.empty_cache()
                print(f"  ✅ 大张量分配和释放成功")
                
                # 测试矩阵乘法
                a = torch.randn(100, 100, dtype=torch.float32, device=device)
                b = torch.randn(100, 100, dtype=torch.float32, device=device)
                c = torch.matmul(a, b)
                print(f"  ✅ 矩阵乘法操作成功")
                
                # 清理
                del x, y, z, a, b, c
                torch_gcu.empty_cache()
                
            except Exception as e:
                print(f"  ❌ GCU设备 {i} 测试失败: {e}")
                
    else:
        print("❌ GCU设备不可用")
        
except ImportError as e:
    print(f"❌ torch_gcu 导入失败: {e}")
    
except Exception as e:
    print(f"❌ GCU测试过程中出现错误: {e}")

print("\n🔍 内存调试完成")
print("如果看到任何错误，请检查:")
print("1. GCU驱动是否正确安装")
print("2. PyTorch GCU版本是否兼容")
print("3. 内存分配器配置是否正确")