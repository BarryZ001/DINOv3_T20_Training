#!/usr/bin/env python3
"""
强制禁用GCU混合精度训练的脚本
解决 free(): invalid pointer 错误，确保所有操作都使用 float32 精度
"""

import os
import torch

def force_float32_gcu():
    """强制GCU使用float32精度，禁用所有混合精度优化"""
    
    # 1. 设置环境变量禁用GCU的自动混合精度
    os.environ['GCU_DISABLE_AMP'] = '1'
    os.environ['GCU_FORCE_FP32'] = '1'
    os.environ['TORCH_GCU_DISABLE_AMP'] = '1'
    
    # 2. 禁用PyTorch的自动混合精度
    os.environ['TORCH_DISABLE_AMP'] = '1'
    
    # 3. 设置默认数据类型为float32
    torch.set_default_dtype(torch.float32)
    
    # 4. 禁用CUDA的混合精度（如果存在）
    os.environ['CUDA_DISABLE_AMP'] = '1'
    
    # 5. 强制DeepSpeed使用float32
    os.environ['DEEPSPEED_DISABLE_FP16'] = '1'
    
    print("🔧 已强制禁用所有混合精度设置，确保使用 float32 精度")
    print("📋 设置的环境变量:")
    for key in ['GCU_DISABLE_AMP', 'GCU_FORCE_FP32', 'TORCH_GCU_DISABLE_AMP', 
                'TORCH_DISABLE_AMP', 'CUDA_DISABLE_AMP', 'DEEPSPEED_DISABLE_FP16']:
        print(f"   {key} = {os.environ.get(key, 'Not set')}")
    
    return True

if __name__ == '__main__':
    force_float32_gcu()