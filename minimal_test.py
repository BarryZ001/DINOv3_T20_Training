#!/usr/bin/env python3
"""
最小化测试脚本 - 用于隔离死循环问题
逐步测试各个组件，找出问题根源
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🔍 开始最小化测试...")

def test_basic_imports():
    """测试基础导入"""
    print("\n=== 测试1: 基础导入 ===")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        try:
            import torch_gcu
            print(f"✅ torch_gcu可用: {torch_gcu.is_available()}")
        except ImportError:
            print("⚠️ torch_gcu不可用（在Mac环境下正常）")
        
        try:
            import deepspeed
            print(f"✅ DeepSpeed: {deepspeed.__version__}")
        except ImportError:
            print("⚠️ DeepSpeed不可用（在Mac环境下正常）")
        
        try:
            from mmengine.config import Config
            print("✅ MMEngine导入成功")
        except ImportError:
            print("⚠️ MMEngine不可用（在Mac环境下正常）")
        
        try:
            from mmengine.dataset import pseudo_collate
            print("✅ pseudo_collate导入成功")
        except ImportError:
            print("⚠️ pseudo_collate不可用（在Mac环境下正常）")
        
        return True
    except Exception as e:
        print(f"❌ 基础导入失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n=== 测试2: 配置文件加载 ===")
    try:
        from mmengine.config import Config
        config_path = "configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
        cfg = Config.fromfile(config_path)
        print("✅ 配置文件加载成功")
        print(f"✅ 数据集类型: {cfg.train_dataloader.dataset.type}")
        return True, cfg
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False, None

def test_dataset_creation():
    """测试数据集创建"""
    print("\n=== 测试3: 数据集创建 ===")
    try:
        # 导入自定义模块
        import mmseg_custom.datasets
        import mmseg_custom.transforms
        
        from mmengine.config import Config
        from mmengine.registry import DATASETS
        
        config_path = "configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
        cfg = Config.fromfile(config_path)
        
        # 创建数据集
        train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
        print(f"✅ 数据集创建成功，长度: {len(train_dataset)}")
        
        # 测试获取一个样本
        sample = train_dataset[0]
        print(f"✅ 样本获取成功，类型: {type(sample)}")
        
        return True, train_dataset
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_collate_function():
    """测试collate函数"""
    print("\n=== 测试4: collate函数测试 ===")
    try:
        from mmengine.dataset import pseudo_collate
        
        # 创建测试数据
        test_data = [
            {"inputs": "test1", "data_samples": "sample1"},
            {"inputs": "test2", "data_samples": "sample2"}
        ]
        
        # 测试collate
        batched = pseudo_collate(test_data)
        print(f"✅ collate函数测试成功，结果类型: {type(batched)}")
        
        return True
    except Exception as e:
        print(f"❌ collate函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader_creation():
    """测试DataLoader创建"""
    print("\n=== 测试5: DataLoader创建 ===")
    try:
        import torch
        from torch.utils.data import DataLoader
        from mmengine.dataset import pseudo_collate
        from mmengine.config import Config
        from mmengine.registry import DATASETS
        
        # 导入自定义模块
        import mmseg_custom.datasets
        import mmseg_custom.transforms
        
        config_path = "configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
        cfg = Config.fromfile(config_path)
        
        train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
        
        # 创建DataLoader - 使用最小参数
        dataloader = DataLoader(
            train_dataset,
            batch_size=1,  # 最小batch size
            shuffle=False,  # 不shuffle
            num_workers=0,  # 不使用多进程
            collate_fn=pseudo_collate
        )
        
        print("✅ DataLoader创建成功")
        
        # 测试迭代一个batch
        print("🔍 测试迭代第一个batch...")
        for i, batch in enumerate(dataloader):
            print(f"✅ 成功获取batch {i}，类型: {type(batch)}")
            if i >= 2:  # 只测试前3个batch
                break
        
        return True
    except Exception as e:
        print(f"❌ DataLoader测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试6: 模型创建 ===")
    try:
        import torch
        from mmengine.config import Config
        from mmengine.registry import MODELS
        
        # 导入自定义模块
        import mmseg_custom.models
        
        config_path = "configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
        cfg = Config.fromfile(config_path)
        
        # 创建模型
        model = MODELS.build(cfg.model)
        print("✅ 模型创建成功")
        
        # 移动到设备
        device = torch.device('gcu:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✅ 模型移动到设备: {device}")
        
        return True, model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """主测试函数"""
    print("🚀 开始逐步测试各个组件...")
    
    # 测试1: 基础导入
    if not test_basic_imports():
        print("💥 基础导入失败，停止测试")
        return
    
    # 测试2: 配置文件
    success, cfg = test_config_loading()
    if not success:
        print("💥 配置文件加载失败，停止测试")
        return
    
    # 测试3: 数据集创建
    success, dataset = test_dataset_creation()
    if not success:
        print("💥 数据集创建失败，停止测试")
        return
    
    # 测试4: collate函数
    if not test_collate_function():
        print("💥 collate函数测试失败，停止测试")
        return
    
    # 测试5: DataLoader
    if not test_dataloader_creation():
        print("💥 DataLoader测试失败，这可能是死循环的原因！")
        return
    
    # 测试6: 模型创建
    success, model = test_model_creation()
    if not success:
        print("💥 模型创建失败，停止测试")
        return
    
    print("\n🎉 所有基础组件测试通过！")
    print("如果到这里都没问题，那么死循环可能出现在DeepSpeed初始化阶段")

if __name__ == '__main__':
    main()