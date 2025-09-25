#!/usr/bin/env python3
"""简化调试脚本：直接检查关键文件中的tensor处理逻辑"""

import torch
import numpy as np

def test_custom_image_to_tensor():
    """测试CustomImageToTensor的行为"""
    print("=== 测试 CustomImageToTensor ===")
    
    # 模拟HWC格式的图像 (512, 512, 3)
    img_hwc = np.random.rand(512, 512, 3).astype(np.float32)
    print(f"输入图像形状 (HWC): {img_hwc.shape}")
    
    # 模拟CustomImageToTensor的转换逻辑
    # 从 mmseg_custom/transforms/standard_transforms.py 中的实现
    img_chw = img_hwc.transpose(2, 0, 1)  # HWC -> CHW
    img_tensor = torch.from_numpy(img_chw)
    
    print(f"CustomImageToTensor输出形状 (CHW): {img_tensor.shape}")
    print(f"输出tensor维度: {img_tensor.dim()}")
    
    return img_tensor

def test_batch_processing():
    """测试批处理逻辑"""
    print("\n=== 测试批处理 ===")
    
    # 获取单个样本
    single_tensor = test_custom_image_to_tensor()
    
    # 模拟多个样本
    batch_tensors = [single_tensor, single_tensor]
    print(f"批次中的样本数: {len(batch_tensors)}")
    
    # 模拟pseudo_collate的行为
    print("\n--- 模拟 pseudo_collate ---")
    
    # 检查每个样本的形状
    for i, tensor in enumerate(batch_tensors):
        print(f"样本 {i} 形状: {tensor.shape}, 维度: {tensor.dim()}")
    
    # 尝试堆叠成批次
    try:
        batched = torch.stack(batch_tensors, dim=0)
        print(f"堆叠后的批次形状: {batched.shape}")
        print(f"堆叠后的维度: {batched.dim()}")
        
        if batched.dim() == 4:
            print("✅ 成功创建4D tensor (B, C, H, W)")
        else:
            print(f"❌ 意外的维度: {batched.dim()}")
            
    except Exception as e:
        print(f"❌ 堆叠失败: {e}")
    
    # 测试如果直接传递3D tensor会发生什么
    print("\n--- 测试直接传递3D tensor ---")
    print(f"单个3D tensor形状: {single_tensor.shape}")
    print("这就是导致 'Expected 4D tensor, got 3D tensor' 错误的原因！")

def analyze_problem():
    """分析问题"""
    print("\n=== 问题分析 ===")
    print("1. CustomImageToTensor 正确地将 HWC -> CHW，产生3D tensor")
    print("2. 但是模型期望4D tensor (B, C, H, W)")
    print("3. 问题可能在于：")
    print("   a) DataLoader的collate_fn没有正确添加batch维度")
    print("   b) 或者数据在传递给模型之前丢失了batch维度")
    print("   c) 或者模型接收到的不是collated的批次数据")

def suggest_fixes():
    """建议修复方案"""
    print("\n=== 修复建议 ===")
    print("1. 检查训练脚本中的数据传递：")
    print("   - 确保使用DataLoader返回的批次数据")
    print("   - 检查是否意外地取了batch[0]或类似操作")
    
    print("\n2. 在模型forward之前添加调试：")
    print("   - 打印输入tensor的形状")
    print("   - 如果是3D，手动添加batch维度")
    
    print("\n3. 检查seg_data_preprocessor.py：")
    print("   - 确保它正确处理批次数据")
    print("   - 验证inputs的格式")

if __name__ == '__main__':
    test_custom_image_to_tensor()
    test_batch_processing()
    analyze_problem()
    suggest_fixes()