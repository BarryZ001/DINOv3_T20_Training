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
import numpy as np
from torch.utils.data.dataloader import default_collate

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


def mmseg_collate_fn(batch, pad_value=0):
    """
    mmsegmentation-style collate_fn:
    - 自动把 numpy 转 torch.Tensor
    - 自动 pad 保证 batch 内图像尺寸一致
    - 保持 dict 结构 (inputs / gt_semantic_seg)
    """
    # batch: list[dict]
    elem = batch[0]
    if isinstance(elem, dict):
        collated = {}
        for key in elem:
            values = [d[key] for d in batch]

            # === 自动把 numpy 转 tensor ===
            if isinstance(values[0], np.ndarray):
                values = [torch.from_numpy(v) for v in values]
            elif isinstance(values[0], list) and len(values[0]) > 0 and isinstance(values[0][0], np.ndarray):
                # 处理 list of numpy arrays
                values = [[torch.from_numpy(arr) for arr in v] for v in values]

            # === 图像数据 (inputs) ===
            if key == "inputs":
                # 如果是 list of tensors，先处理成统一格式
                if isinstance(values[0], list):
                    # 展平成单个 tensor 列表
                    flat_tensors = []
                    for v in values:
                        flat_tensors.extend(v)
                    values = flat_tensors
                
                # 找出最大高宽
                max_h = max(v.shape[-2] for v in values if hasattr(v, 'shape'))
                max_w = max(v.shape[-1] for v in values if hasattr(v, 'shape'))
                padded = []
                for v in values:
                    if hasattr(v, 'dim') and v.dim() == 3:  # [C, H, W]
                        c, h, w = v.shape
                        pad = torch.full((c, max_h, max_w), pad_value, dtype=v.dtype)
                        pad[:, :h, :w] = v
                        padded.append(pad)
                    elif hasattr(v, 'dim') and v.dim() == 4:  # [B, C, H, W] - 已经是批次
                        padded.append(v)
                
                collated[key] = torch.stack(padded, dim=0)  # [B,C,H,W]

            # === 标签数据 (gt_semantic_seg) ===
            elif key == "gt_semantic_seg":
                # 如果是 list of tensors，先处理成统一格式
                if isinstance(values[0], list):
                    # 展平成单个 tensor 列表
                    flat_tensors = []
                    for v in values:
                        flat_tensors.extend(v)
                    values = flat_tensors
                
                max_h = max(v.shape[-2] for v in values if hasattr(v, 'shape'))
                max_w = max(v.shape[-1] for v in values if hasattr(v, 'shape'))
                padded = []
                for v in values:
                    if hasattr(v, 'dim') and v.dim() == 2:  # [H, W]
                        h, w = v.shape
                        pad = torch.full((max_h, max_w), pad_value, dtype=torch.long)
                        pad[:h, :w] = v.long()
                        padded.append(pad)
                    elif hasattr(v, 'dim') and v.dim() == 3:  # [1, H, W] or [C, H, W]
                        if v.shape[0] == 1:  # [1, H, W]
                            h, w = v.shape[-2:]
                            pad = torch.full((1, max_h, max_w), pad_value, dtype=torch.long)
                            pad[:, :h, :w] = v.long()
                            padded.append(pad)
                        else:  # [C, H, W] - 取第一个通道作为标签
                            h, w = v.shape[-2:]
                            pad = torch.full((max_h, max_w), pad_value, dtype=torch.long)
                            pad[:h, :w] = v[0].long()  # 取第一个通道
                            padded.append(pad)
                
                collated[key] = torch.stack(padded, dim=0)  # [B,H,W] or [B,1,H,W]

            else:
                # 其他 key 用默认方式
                try:
                    collated[key] = default_collate(values)
                except:
                    # 如果默认 collate 失败，保持原样
                    collated[key] = values

        return collated

    else:
        return default_collate(batch)


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
        collate_fn=collate if collate else None,  # 🔧 暂时使用原来的 collate，在训练循环中手动处理
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
            # 检查是否有不同尺寸的图像
            shapes = [inp.shape for inp in inputs]
            print(f"[DEBUG] input shapes: {shapes}")
            
            # 如果尺寸不一致，需要 resize 到统一大小
            if len(set(shapes)) > 1:
                print("[DEBUG] Found different input sizes, resizing to common size...")
                # 找到最大尺寸
                max_h = max(inp.shape[-2] for inp in inputs)
                max_w = max(inp.shape[-1] for inp in inputs)
                print(f"[DEBUG] Target size: {max_h}x{max_w}")
                
                # 🔧 修复：使用 F.pad 确保内存连续性和正确的尺寸计算
                import torch.nn.functional as F
                padded_inputs = []
                for inp in inputs:
                    c, h, w = inp.shape
                    if h != max_h or w != max_w:
                        # 计算需要的 padding
                        pad_h = max_h - h
                        pad_w = max_w - w
                        # F.pad 格式: (left, right, top, bottom)
                        padded = F.pad(inp, (0, pad_w, 0, pad_h), mode='constant', value=0)
                        # 确保内存连续性
                        padded = padded.contiguous()
                        padded_inputs.append(padded)
                    else:
                        # 确保原始张量也是连续的
                        padded_inputs.append(inp.contiguous())
                inputs = padded_inputs
            
            inputs = torch.stack(inputs, dim=0)
            print(f"[DEBUG] after stacking: {inputs.shape}")
        elif isinstance(inputs, torch.Tensor) and inputs.dim() == 3:
            print("[DEBUG] single image tensor, unsqueezing batch dim...")
            inputs = inputs.unsqueeze(0)
            print(f"[DEBUG] after unsqueeze: {inputs.shape}")
        
        # 🔧 混合精度修复：使用模型参数的真实 device 和 dtype
        # 获取 DeepSpeed 包裹后的真实模型参数信息
        device = next(model_engine.parameters()).device
        dtype = next(model_engine.parameters()).dtype
        
        # 🔧 T20 内存安全修复：分步骤进行设备转换，避免内存指针错误
        print(f"[DEBUG] Converting inputs to device: {device}, dtype: {dtype}")
        print(f"[DEBUG] Original inputs device: {inputs.device}, dtype: {inputs.dtype}")
        
        # 先确保张量在 CPU 上且内存连续
        if inputs.device != torch.device('cpu'):
            inputs = inputs.cpu().contiguous()
        
        # 分步转换：先转换数据类型，再转换设备
        if inputs.dtype != dtype:
            inputs = inputs.to(dtype=dtype)
            print(f"[DEBUG] Converted dtype to: {inputs.dtype}")
        
        # 最后转换设备，使用 non_blocking=False 确保同步转换
        if inputs.device != device:
            inputs = inputs.to(device=device, non_blocking=False)
            print(f"[DEBUG] Converted device to: {inputs.device}")
        
        print(f"[DEBUG] final inputs shape: {inputs.shape}, device: {inputs.device}, dtype: {inputs.dtype}")
        
        # 🔧 处理 batch 中的 gt_semantic_seg（如果存在）
        if 'gt_semantic_seg' in batch:
            gt_seg = batch['gt_semantic_seg']
            print(f"[DEBUG] gt_semantic_seg type: {type(gt_seg)}")
            
            if isinstance(gt_seg, list):
                print(f"[DEBUG] gt_semantic_seg is list, len={len(gt_seg)}")
                print(f"[DEBUG] gt_semantic_seg element types: {[type(x) for x in gt_seg]}")
                
                # 先将 numpy 转换为 tensor
                gt_tensors = []
                for i, seg in enumerate(gt_seg):
                    if isinstance(seg, np.ndarray):
                        gt_tensors.append(torch.from_numpy(seg))
                    elif isinstance(seg, torch.Tensor):
                        gt_tensors.append(seg)
                    else:
                        raise TypeError(f"Unexpected gt_semantic_seg[{i}] type: {type(seg)}")
                
                # 检查尺寸是否一致
                shapes = [t.shape for t in gt_tensors]
                print(f"[DEBUG] gt_semantic_seg shapes: {shapes}")
                
                if len(set(shapes)) > 1:
                    print("[DEBUG] Found different gt_semantic_seg sizes, padding to common size...")
                    # 找到最大尺寸
                    max_h = max(t.shape[-2] for t in gt_tensors)
                    max_w = max(t.shape[-1] for t in gt_tensors)
                    print(f"[DEBUG] Target gt_semantic_seg size: {max_h}x{max_w}")
                    
                    # 🔧 修复：使用 F.pad 确保内存连续性
                    import torch.nn.functional as F
                    padded_gts = []
                    for t in gt_tensors:
                        if t.dim() == 2:  # [H, W]
                            h, w = t.shape
                            if h != max_h or w != max_w:
                                # 计算需要的 padding
                                pad_h = max_h - h
                                pad_w = max_w - w
                                # F.pad 格式: (left, right, top, bottom)
                                padded = F.pad(t.long(), (0, pad_w, 0, pad_h), mode='constant', value=0)
                                # 确保内存连续性
                                padded = padded.contiguous()
                                padded_gts.append(padded)
                            else:
                                # 确保原始张量也是连续的
                                padded_gts.append(t.long().contiguous())
                        else:
                            padded_gts.append(t.long())
                    gt_tensors = padded_gts
                
                # 堆叠并移动到设备 - 🔧 T20 内存安全修复
                print("[DEBUG] Stacking gt_semantic_seg tensors...")
                stacked_gt = torch.stack(gt_tensors)
                print(f"[DEBUG] Stacked gt_semantic_seg shape: {stacked_gt.shape}")
                
                # 分步转换到目标设备，避免内存指针错误
                if stacked_gt.device != torch.device('cpu'):
                    stacked_gt = stacked_gt.cpu().contiguous()
                
                # 先转换数据类型
                if stacked_gt.dtype != torch.long:
                    stacked_gt = stacked_gt.to(dtype=torch.long)
                    print(f"[DEBUG] Converted gt_semantic_seg dtype to: {stacked_gt.dtype}")
                
                # 最后转换设备
                if stacked_gt.device != device:
                    stacked_gt = stacked_gt.to(device=device, non_blocking=False)
                    print(f"[DEBUG] Converted gt_semantic_seg device to: {stacked_gt.device}")
                
                batch['gt_semantic_seg'] = stacked_gt
                print(f"[DEBUG] Final gt_semantic_seg: shape={batch['gt_semantic_seg'].shape}, device={batch['gt_semantic_seg'].device}, dtype={batch['gt_semantic_seg'].dtype}")
            else:
                # 单个张量或 numpy 数组 - 🔧 T20 内存安全修复
                print("[DEBUG] Processing single gt_semantic_seg...")
                if isinstance(gt_seg, np.ndarray):
                    gt_seg = torch.from_numpy(gt_seg)
                    print(f"[DEBUG] Converted numpy to tensor: {gt_seg.shape}")
                
                # 分步转换到目标设备，避免内存指针错误
                if gt_seg.device != torch.device('cpu'):
                    gt_seg = gt_seg.cpu().contiguous()
                
                # 先转换数据类型
                if gt_seg.dtype != torch.long:
                    gt_seg = gt_seg.to(dtype=torch.long)
                    print(f"[DEBUG] Converted single gt_semantic_seg dtype to: {gt_seg.dtype}")
                
                # 最后转换设备
                if gt_seg.device != device:
                    gt_seg = gt_seg.to(device=device, non_blocking=False)
                    print(f"[DEBUG] Converted single gt_semantic_seg device to: {gt_seg.device}")
                
                batch['gt_semantic_seg'] = gt_seg
                print(f"[DEBUG] Final single gt_semantic_seg: shape={batch['gt_semantic_seg'].shape}, device={batch['gt_semantic_seg'].device}, dtype={batch['gt_semantic_seg'].dtype}")
            
            # 🔧 将监督信号也转移到相同设备
            if data_samples is not None:
                for i, sample in enumerate(data_samples):
                    if hasattr(sample, 'gt_sem_seg') and sample.gt_sem_seg is not None:
                        if hasattr(sample.gt_sem_seg, 'data'):
                            sample.gt_sem_seg.data = sample.gt_sem_seg.data.to(device=device)
                            print(f"[DEBUG] gt_sem_seg[{i}] moved to device: {device}")
            
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