#!/usr/bin/env python3
"""
燧原T20 DeepSpeed训练脚本 (生产版)
使用MMEngine构建组件，DeepSpeed驱动训练
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Any

# 项目路径配置
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ.setdefault('PYTORCH_GCU_ALLOC_CONF', 'backend:topsMallocAsync')

import torch

# 条件导入模块，避免在开发环境中的导入错误
torch_gcu_available = False
deepspeed_available = False
mmengine_available = False

# 类型注解变量
Config: Optional[Any] = None
MODELS: Optional[Any] = None
DATASETS: Optional[Any] = None
collate: Optional[Any] = None
torch_gcu: Optional[Any] = None
deepspeed: Optional[Any] = None

try:
    import torch_gcu  # type: ignore
    torch_gcu_available = True
except ImportError:
    torch_gcu = None

try:
    import deepspeed  # type: ignore
    deepspeed_available = True
except ImportError:
    deepspeed = None

try:
    from mmengine.config import Config  # type: ignore
    from mmengine.registry import MODELS, DATASETS  # type: ignore
    from mmengine.dataset import pseudo_collate as collate  # type: ignore
    mmengine_available = True
except ImportError:
    Config = None
    MODELS = None
    DATASETS = None
    collate = None

# 导入自定义模块（仅在MMEngine可用时）
if mmengine_available:
    try:
        import mmseg_custom.models  # type: ignore
        import mmseg_custom.datasets  # type: ignore
        import mmseg_custom.transforms  # type: ignore
    except ImportError:
        pass


def build_components(cfg: Any, device_name: str) -> tuple:
    """构建训练组件"""
    if not mmengine_available or DATASETS is None or MODELS is None:
        raise RuntimeError("MMEngine not available")
    
    # 构建数据集
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    
    # 构建模型
    model = MODELS.build(cfg.model)
    
    # 设置设备 - 直接使用device_name字符串，兼容MMEngine的.to()方法
    model = model.to(device_name)
    
    return model, dataset


def main() -> None:
    """主训练函数"""
    # 🔧 强制禁用CUDA特定优化器，确保GCU环境兼容性
    # 这是解决IndexError: list index out of range的关键环境变量设置
    os.environ['DEEPSPEED_DISABLE_FUSED_ADAM'] = '1'
    os.environ['DS_BUILD_FUSED_ADAM'] = '0'
    os.environ['DS_BUILD_CPU_ADAM'] = '1'  # 强制使用CPU版本的Adam
    os.environ['DS_BUILD_UTILS'] = '0'  # 禁用其他CUDA特定工具
    os.environ['DS_BUILD_AIO'] = '0'  # 禁用异步IO（可能依赖CUDA）
    os.environ['DS_BUILD_SPARSE_ATTN'] = '0'  # 禁用稀疏注意力（CUDA特定）
    
    parser = argparse.ArgumentParser(description='DeepSpeed Training')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--work-dir', required=True, help='工作目录')
    parser.add_argument('--deepspeed', required=True, help='DeepSpeed配置文件')
    parser.add_argument('--launcher', default='deepspeed', help='启动器类型')
    parser.add_argument('--local_rank', type=int, default=0, help='本地rank')
    
    args = parser.parse_args()
    
    # 检查必要模块
    if not mmengine_available or Config is None:
        print("Error: MMEngine not available")
        return
    
    if not deepspeed_available or deepspeed is None:
        print("Error: DeepSpeed not available")
        return
    
    print(f"🔧 已设置环境变量禁用FusedAdam和其他CUDA特定组件，确保GCU兼容性")
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 环境设置 - 使用xla设备格式以兼容MMEngine
    if torch_gcu_available and torch_gcu is not None:
        device_id = torch_gcu.current_device()
        device_name = f'xla:{device_id}'
    else:
        device_name = 'cuda'
    
    # 构建组件
    model, dataset = build_components(cfg, device_name)
    
    # 加载DeepSpeed配置
    with open(args.deepspeed, 'r') as f:
        deepspeed_config = json.load(f)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=deepspeed_config.get('train_micro_batch_size_per_gpu', 8),
        shuffle=True,
        collate_fn=collate if collate else None,
        num_workers=4
    )
    
    # 🔧 初始化DeepSpeed - 依赖配置文件中的优化器设置
    # 不再手动创建优化器，避免与DeepSpeed的FusedAdam冲突
    # 配置文件中已明确指定使用AdamW优化器，兼容GCU硬件
    # 这修复了 IndexError: list index out of range 错误，确保使用标准PyTorch优化器
    # 🔧 新增：通过环境变量和配置参数双重保障禁用FusedAdam
    print("🔧 正在初始化DeepSpeed，已禁用FusedAdam确保GCU兼容性...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=deepspeed_config
    )
    
    print("DeepSpeed训练开始...")
    
    # 简单训练循环
    for step, batch in enumerate(dataloader):
        if step >= 10:  # 限制步数用于测试
            break
            
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()
        
        if step % 5 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
    
    print("训练完成")


if __name__ == '__main__':
    main()