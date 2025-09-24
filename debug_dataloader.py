#!/usr/bin/env python3
"""
数据加载调试脚本 - 本地版本
用于诊断 RuntimeError: stack expects a non-empty TensorList 问题
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, DATASETS
from torch.utils.data import DataLoader

# 导入自定义模块
import mmseg_custom.datasets
import mmseg_custom.transforms
import mmseg_custom.models

def debug_dataloader(config_path):
    """调试数据加载器"""
    print("🔍 === 开始数据加载调试 ===")
    
    # 1. 加载配置
    cfg = Config.fromfile(config_path)
    print(f"✅ 配置文件加载成功: {config_path}")
    
    # 2. 修改数据路径为本地路径
    if hasattr(cfg, 'local_data_root'):
        cfg.train_dataloader.dataset.data_root = cfg.local_data_root
        print(f"📁 使用本地数据路径: {cfg.local_data_root}")
    
    # 3. 构建数据集
    try:
        train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
        print(f"✅ 数据集构建成功，样本数量: {len(train_dataset)}")
    except Exception as e:
        print(f"❌ 数据集构建失败: {e}")
        return
    
    # 4. 创建DataLoader
    from mmengine.dataset import pseudo_collate as collate
    
    debug_dataloader = DataLoader(
        train_dataset,
        batch_size=2,  # 小批次用于调试
        shuffle=True,
        num_workers=0,  # 单进程调试
        collate_fn=collate
    )
    
    print(f"✅ DataLoader创建成功")
    
    # 5. 迭代检查批次
    for i, batch in enumerate(debug_dataloader):
        print(f"\n--- 正在检查 Batch #{i} ---")
        print(f"Batch keys: {list(batch.keys())}")
        
        if 'data_samples' not in batch:
            print("❌ 错误: 批处理数据中没有 'data_samples' 键！")
            print(f"实际的键: {list(batch.keys())}")
            continue
        
        print(f"Batch size: {len(batch['data_samples'])}")
        has_labels_count = 0
        
        for j, sample in enumerate(batch['data_samples']):
            print(f"  样本 #{j}:")
            print(f"    类型: {type(sample)}")
            
            # 检查所有可能的标签字段
            label_fields = ['gt_sem_seg', 'gt_semantic_seg', 'gt_seg_map']
            found_label = False
            
            for field in label_fields:
                if hasattr(sample, field):
                    label_data = getattr(sample, field)
                    if label_data is not None:
                        has_labels_count += 1
                        found_label = True
                        if hasattr(label_data, 'data'):
                            print(f"    ✅ 找到标签字段 '{field}', 形状: {label_data.data.shape}")
                        else:
                            print(f"    ✅ 找到标签字段 '{field}', 类型: {type(label_data)}")
                        break
                    else:
                        print(f"    ⚠️ 字段 '{field}' 存在但为 None")
            
            if not found_label:
                print(f"    ❌ 未找到任何标签字段")
                # 打印样本的所有属性
                if hasattr(sample, '__dict__'):
                    print(f"    所有属性: {list(sample.__dict__.keys())}")
                elif hasattr(sample, '_fields'):
                    print(f"    所有字段: {sample._fields}")
        
        print(f"批次 #{i} 中有效标签数量: {has_labels_count}/{len(batch['data_samples'])}")
        
        if has_labels_count == 0:
            print(f"❌❌❌ 致命错误: Batch #{i} 中所有样本都缺少有效标签！")
            print("这就是导致 torch.stack 失败的原因。")
            
            # 详细检查第一个样本
            if len(batch['data_samples']) > 0:
                sample = batch['data_samples'][0]
                print(f"\n详细检查第一个样本:")
                print(f"样本类型: {type(sample)}")
                if hasattr(sample, '__dict__'):
                    for key, value in sample.__dict__.items():
                        print(f"  {key}: {type(value)} = {value}")
        else:
            print(f"✅ 批次 #{i} 包含有效标签")
        
        if i >= 3:  # 只检查前几个批次
            break
    
    print("\n🔍 === 数据加载调试结束 ===")

if __name__ == "__main__":
    config_path = "configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
    debug_dataloader(config_path)