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
    # 🔧 强化版GCU兼容性设置 - 彻底禁用所有CUDA特定功能
    # 这是解决IndexError: list index out of range的关键环境变量设置
    
    # DeepSpeed CUDA特定组件禁用
    os.environ['DEEPSPEED_DISABLE_FUSED_ADAM'] = '1'
    os.environ['DS_BUILD_FUSED_ADAM'] = '0'
    os.environ['DS_BUILD_CPU_ADAM'] = '1'  # 强制使用CPU版本的Adam
    os.environ['DS_BUILD_UTILS'] = '0'  # 禁用其他CUDA特定工具
    os.environ['DS_BUILD_AIO'] = '0'  # 禁用异步IO（可能依赖CUDA）
    os.environ['DS_BUILD_SPARSE_ATTN'] = '0'  # 禁用稀疏注意力（CUDA特定）
    
    # 额外的CUDA特定功能禁用
    os.environ['DS_BUILD_FUSED_LAMB'] = '0'  # 禁用FusedLamb优化器
    os.environ['DS_BUILD_TRANSFORMER'] = '0'  # 禁用CUDA Transformer内核
    os.environ['DS_BUILD_STOCHASTIC_TRANSFORMER'] = '0'  # 禁用随机Transformer
    os.environ['DS_BUILD_TRANSFORMER_INFERENCE'] = '0'  # 禁用Transformer推理内核
    os.environ['DS_BUILD_QUANTIZER'] = '0'  # 禁用量化器（可能依赖CUDA）
    os.environ['DS_BUILD_RANDOM_LTD'] = '0'  # 禁用随机LTD
    
    # PyTorch CUDA相关设置
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 隐藏CUDA设备
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''  # 清空CUDA架构列表
    
    # 强制使用CPU后端进行某些操作
    os.environ['OMP_NUM_THREADS'] = '4'  # 限制OpenMP线程数
    
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
    
    # 🔧 初始化DeepSpeed - 手动创建优化器避免FusedAdam编译问题
    # 这是解决 IndexError: list index out of range 的最终方案
    # 通过手动创建标准PyTorch优化器，绕过DeepSpeed内部的CUDA特定代码路径
    print("🔧 正在手动创建优化器，确保GCU兼容性...")
    
    # 🔧 关键修正 (1/2): 从配置中获取优化器参数并手动创建
    optimizer_params = deepspeed_config.get('optimizer', {}).get('params', {})
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
    print(f"✅ 手动创建优化器成功: {type(optimizer).__name__}")
    
    print("🔧 正在初始化DeepSpeed，使用手动创建的优化器...")
    
    # 🔧 关键修正 (2/2): 将手动创建的optimizer实例传递给initialize函数
    # 这避免了DeepSpeed内部尝试编译FusedAdam的问题
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),  # 关键：提供模型参数给DeepSpeed
        optimizer=optimizer,  # 🔧 关键：传入手动创建的优化器
        config=deepspeed_config
    )
    
    print("DeepSpeed训练开始...")
    
    # 🔥 修复的训练循环 - 正确处理批次数据格式
    for step, batch in enumerate(dataloader):
        if step >= 10:  # 限制步数用于测试
            break
        
        # 🔧 关键修复：正确提取和处理批次数据
        if step == 0:
            print(f"🔍 调试信息 - Batch 结构: {type(batch)}")
            if isinstance(batch, dict):
                print(f"🔍 Batch keys: {list(batch.keys())}")
                if 'inputs' in batch:
                    print(f"🔍 inputs 形状: {batch['inputs'].shape if hasattr(batch['inputs'], 'shape') else type(batch['inputs'])}")
                if 'data_samples' in batch:
                    print(f"🔍 data_samples 类型: {type(batch['data_samples'])}")
        
        # 根据 MMEngine 标准，模型期望接收 inputs 和 data_samples
        if isinstance(batch, dict) and 'inputs' in batch:
            # 标准 MMEngine 格式
            inputs = batch['inputs']
            data_samples = batch.get('data_samples', None)
            
            # 🔧 关键修复：确保 inputs 是正确的 4D tensor (B, C, H, W)
            if isinstance(inputs, list):
                print(f"[DEBUG] inputs is list, stacking {len(inputs)} tensors...")
                inputs = torch.stack(inputs, dim=0)
                print(f"[DEBUG] after stacking: {inputs.shape}")
            elif isinstance(inputs, torch.Tensor) and inputs.dim() == 3:
                print("[DEBUG] single image tensor, unsqueezing batch dim...")
                inputs = inputs.unsqueeze(0)
                print(f"[DEBUG] after unsqueeze: {inputs.shape}")
            
            print(f"[DEBUG] final inputs shape: {inputs.shape}")
            
            # 调用模型的 forward 方法
            loss_dict = model_engine(inputs, data_samples, mode='loss')
            
            # 处理返回的 loss
            if isinstance(loss_dict, dict):
                loss = loss_dict.get('loss', loss_dict.get('decode.loss_ce', list(loss_dict.values())[0]))
            else:
                loss = loss_dict
                
        else:
            # 兜底处理：直接传递整个 batch
            print(f"⚠️ 警告：使用兜底处理，直接传递 batch")
            loss = model_engine(batch)
        
        model_engine.backward(loss)
        model_engine.step()
        
        if step % 5 == 0:
            loss_value = loss.item() if hasattr(loss, 'item') else loss
            print(f"Step {step}, Loss: {loss_value}")
    
    print("训练完成")


if __name__ == '__main__':
    main()