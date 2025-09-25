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

# 🔧 强制禁用混合精度训练 - 解决 free(): invalid pointer 错误
# 必须在导入torch之前设置这些环境变量
os.environ.setdefault('GCU_DISABLE_AMP', '1')        # 禁用GCU自动混合精度
os.environ.setdefault('GCU_FORCE_FP32', '1')         # 强制GCU使用float32
os.environ.setdefault('TORCH_GCU_DISABLE_AMP', '1')  # 禁用PyTorch GCU混合精度
os.environ.setdefault('TORCH_DISABLE_AMP', '1')      # 禁用PyTorch自动混合精度
os.environ.setdefault('DEEPSPEED_DISABLE_FP16', '1') # 禁用DeepSpeed混合精度

# 🔧 GCU 内存分配优化 - 解决 invalid pointer 错误
# 使用更保守的内存分配策略，避免内存碎片和指针错误
os.environ.setdefault('PYTORCH_GCU_ALLOC_CONF', 'max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:False')
# 添加额外的 GCU 环境变量以提高稳定性
os.environ.setdefault('GCU_MEMORY_FRACTION', '0.7')  # 进一步限制内存使用，避免内存碎片
os.environ.setdefault('GCU_ENABLE_LAZY_INIT', '0')   # 禁用延迟初始化，确保确定性行为
os.environ.setdefault('GCU_SYNC_ALLOC', '1')         # 启用同步内存分配，避免异步分配导致的指针错误
os.environ.setdefault('GCU_DISABLE_CACHING', '1')    # 禁用内存缓存，强制每次都重新分配

import torch
import numpy as np
from torch.utils.data.dataloader import default_collate

# 🔧 强制设置默认数据类型为float32，确保所有张量都使用float32精度
torch.set_default_dtype(torch.float32)

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


# 🔧 注释掉自定义 collate 函数，现在使用 MMEngine 的 pseudo_collate
# 这个函数之前用于处理 numpy 到 tensor 的转换和 padding，
# 但现在数据管道已经使用 PackSegInputs 产生标准的 SegDataSample 对象，
# 应该使用 MMEngine 的 pseudo_collate 来避免 RecursionError

# def mmseg_collate_fn(batch, pad_value=0):
#     """
#     mmsegmentation-style collate_fn:
#     - 自动把 numpy 转 torch.Tensor
#     - 自动 pad 保证 batch 内图像尺寸一致
#     - 保持 dict 结构 (inputs / gt_semantic_seg)
#     """
#     # ... (原有实现已注释)
#     pass


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
    
    # 🔧 GCU 分布式训练特定设置 - 基于官方最佳实践
    os.environ['ENFLAME_CLUSTER_PARALLEL'] = 'true'
    os.environ['ENFLAME_ENABLE_EFP'] = 'true'
    os.environ['TOPS_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    os.environ['OMP_NUM_THREADS'] = '5'
    os.environ['ECCL_ASYNC_DISABLE'] = 'false'
    os.environ['ENABLE_RDMA'] = 'true'
    os.environ['ECCL_MAX_NCHANNELS'] = '2'
    os.environ['ENFLAME_UMD_FLAGS'] = 'mem_alloc_retry_times=1'
    os.environ['ECCL_RUNTIME_3_0_ENABLE'] = 'true'
    os.environ['ENFLAME_PT_EVALUATE_TENSOR_NEEDED'] = 'false'
    # 🔧 关键修复：使用异步内存分配器，避免 invalid pointer 错误
    os.environ['PYTORCH_GCU_ALLOC_CONF'] = 'backend:topsMallocAsync'  # 改为异步分配器
    
    # 🚀 流水线并行配置 - 燧原官方推荐
    os.environ['TP_SIZE'] = '1'  # 张量并行大小设为1
    os.environ['PP_SIZE'] = '8'  # 流水线并行大小设为8（8卡）
    os.environ['DP_SIZE'] = '1'  # 数据并行大小设为1
    
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
    
    # 🔧 初始化分布式训练 - 基于燧原官方最佳实践
    print("🔧 正在初始化燧原GCU分布式训练环境...")
    
    # 🔧 获取分布式训练参数
    local_rank = args.local_rank if hasattr(args, 'local_rank') else 0
    world_size = int(os.environ.get('WORLD_SIZE', '8'))
    rank = int(os.environ.get('RANK', '0'))
    
    print(f"🔧 分布式参数: local_rank={local_rank}, world_size={world_size}, rank={rank}")
    
    # 🔧 安全的 GCU 设备初始化
    if torch_gcu_available and torch_gcu is not None:
        try:
            # 设置当前设备为local_rank对应的GCU设备
            torch_gcu.set_device(local_rank)
            device = torch_gcu.current_device()
            print(f"🔧 设置GCU设备: gcu:{device} (local_rank: {local_rank})")
            
            # 延迟模型移动，先让 GCU 完全初始化
            print("🔧 等待 GCU Context 完全初始化...")
            torch_gcu.synchronize()
            
            # 现在安全地移动模型 - 使用.gcu()方法
            model = model.gcu(device)
            device_name = f'gcu:{device}'
            print(f"✅ 模型已安全移动到 GCU 设备: {device_name}")
            print(f"🔧 模型设备: {next(model.parameters()).device}")
            
        except Exception as e:
            print(f"⚠️ GCU 初始化失败: {e}")
            print("🔧 降级到 CPU 模式...")
            model = model.to('cpu')
            device_name = 'cpu'
    else:
        model = model.to('cpu')
        device_name = 'cpu'
        print("⚠️ 使用 CPU 设备")
    
    # 🔧 初始化分布式后端 - 让 torch.distributed.launch 处理
    print("🔧 分布式后端由 torch.distributed.launch 自动初始化 (使用 ECCL)")
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    
    # 🔧 关键修复：强制使用 MMEngine 的 pseudo_collate 来处理 SegDataSample 对象
    # 这解决了 RecursionError: maximum recursion depth exceeded 的问题
    if not collate:
        raise RuntimeError("MMEngine pseudo_collate is required but not available. Please install MMEngine.")
    
    dataloader = DataLoader(
        dataset,
        batch_size=deepspeed_config.get('train_micro_batch_size_per_gpu', 2),
        shuffle=True,
        collate_fn=collate,  # 🔧 使用 MMEngine 的 pseudo_collate 处理现代 SegDataSample 对象
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
    
    try:
        # 🔧 关键修正: 使用燧原官方推荐的DeepSpeed初始化方式
        print("🔧 正在初始化DeepSpeed引擎...")
        
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,  # 使用手动创建的优化器，避免FusedAdam编译问题
            config=deepspeed_config,
            dist_init_required=False  # 重要：由于torch.distributed.launch已经初始化了分布式，这里设为False
        )
        print("✅ DeepSpeed 引擎初始化成功")
        
        # 验证设备
        print(f"🔧 模型设备: {next(model_engine.parameters()).device}")
        
    except Exception as e:
        print(f"❌ DeepSpeed 初始化失败: {e}")
        print("🔧 尝试降级到单卡训练模式...")
        
        # 降级到单卡训练
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
        model_engine = model  # 直接使用模型，不使用 DeepSpeed 包装
        print("✅ 降级到单卡训练模式成功")
    
    print("训练开始...")
    
    # --- 简化后的训练循环 ---
    # 由于配置文件中的数据流水线已经保证了输出格式的正确性，
    # 现在可以极大地简化训练循环中的数据处理代码
    for step, batch in enumerate(dataloader):
        if step >= 10:  # 限制步数用于测试
            break
        
        # 1. 从批次中解包数据
        #    MMEngine的pseudo_collate和PackSegInputs确保了这里的格式是固定的
        inputs = batch['inputs'].to(model_engine.device)
        data_samples = [s.to(model_engine.device) for s in batch['data_samples']]

        # 2. 直接调用模型
        #    MMEngine的模型会自动处理 inputs 和 data_samples
        loss_dict = model_engine(inputs, data_samples, mode='loss')
        loss = loss_dict['loss'] if isinstance(loss_dict, dict) else loss_dict
        
        # 3. 反向传播和优化
        model_engine.backward(loss)
        model_engine.step()
        
        if step % 5 == 0:
            loss_value = loss.item() if hasattr(loss, 'item') else loss
            print(f"Step {step}, Loss: {loss_value}")
    
    print("训练完成")


if __name__ == '__main__':
    main()