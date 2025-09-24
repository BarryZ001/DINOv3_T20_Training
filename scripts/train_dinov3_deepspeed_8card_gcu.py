#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv3 8卡分布式训练脚本 (v2.0 - 生产版)
使用 MMEngine 构建组件，由 DeepSpeed 驱动训练
"""
import argparse
import json
import os
import sys
from pathlib import Path

# --- 环境与库导入 ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ.setdefault('PYTORCH_GCU_ALLOC_CONF', 'backend:topsMallocAsync')

import torch
import torch_gcu
import deepspeed
from mmengine.config import Config
from mmengine.registry import MODELS, DATASETS
from mmengine.dataset import pseudo_collate as collate  # 使用正确的collate函数

# 导入自定义模块
import mmseg_custom.models
import mmseg_custom.datasets
import mmseg_custom.transforms


def build_components(cfg, device_name):
    """根据配置构建模型和数据集"""
    print(f"📊 构建数据集: {cfg.train_dataloader.dataset.type}")
    train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(f"✅ 训练数据集大小: {len(train_dataset)}")
    
    print(f"🏗️ 构建模型: {cfg.model.type}")
    model = MODELS.build(cfg.model)
    model = model.to(device_name)
    print(f"✅ 模型已移动到设备: {device_name}")
    
    return model, train_dataset


def main():
    parser = argparse.ArgumentParser(description='DINOv3 8卡分布式训练 (MMEngine + DeepSpeed)')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录，覆盖配置文件中的设置')
    parser.add_argument('--local_rank', type=int, default=-1, help='DeepSpeed自动注入的参数')
    args = parser.parse_args()

    # --- 1. 环境设置 ---
    local_rank = args.local_rank
    device_name = f"xla:{local_rank}"
    torch_gcu.set_device(local_rank)
    print(f"[Rank {local_rank}] 环境设置完毕，目标设备: {device_name}")

    # --- 2. 加载MMEngine配置 ---
    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # 将DeepSpeed配置写入临时JSON文件
    ds_config_path = os.path.join(cfg.work_dir, 'ds_config.json')
    with open(ds_config_path, 'w') as f:
        json.dump(cfg.deepspeed_config, f, indent=2)
    print(f"📝 DeepSpeed 配置文件已生成: {ds_config_path}")

    # --- 3. 构建模型和数据集 ---
    model, train_dataset = build_components(cfg, device_name)

    # --- 4. 初始化DeepSpeed ---
    print("🔧 初始化DeepSpeed引擎...")
    engine, _, train_dataloader, _ = deepspeed.initialize(
        config=ds_config_path,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=collate
    )
    print("✅ DeepSpeed引擎初始化完成")

    # --- 5. 训练循环 ---
    print("🚀 开始训练...")
    max_iters = cfg.train_cfg.max_iters
    
    for step, batch in enumerate(train_dataloader):
        if step >= max_iters:
            break
        
        # MMEngine 模型期望的输入格式是 (inputs, data_samples, mode)
        inputs = batch['inputs'].to(engine.device)
        data_samples = [s.to(engine.device) for s in batch['data_samples']]

        # 前向+后向+更新
        loss = engine(inputs, data_samples, mode='loss')['loss']
        engine.backward(loss)
        engine.step()

        if local_rank == 0 and step % cfg.deepspeed_config.get("steps_per_print", 50) == 0:
            print(f"[Rank {local_rank}] Step={step}/{max_iters}  Loss={loss.item():.4f}")

    # --- 6. 保存模型 ---
    # DeepSpeed的 `save_checkpoint` 会自动处理同步，确保只在rank 0上执行IO操作
    engine.save_checkpoint(cfg.work_dir, tag=f"step_{max_iters}")
    print(f"💾 DeepSpeed Checkpoint 已保存至: {cfg.work_dir}")

    print(f"[Rank {local_rank}] 训练完成!")


if __name__ == '__main__':
    main()