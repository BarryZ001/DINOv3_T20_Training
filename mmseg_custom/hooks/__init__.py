# 创建SegVisualizationHook并注册到MMEngine的HOOKS注册表中
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from typing import Optional, Sequence, Union

@HOOKS.register_module()
class SegVisualizationHook(Hook):
    """自定义的分割可视化钩子，兼容T20服务器环境"""
    
    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.01,
                 backend_args: Optional[dict] = None):
        super().__init__()
        self.draw = draw
        self.interval = interval
        self.show = show
        self.wait_time = wait_time
        self.backend_args = backend_args
        
    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        """训练迭代后的钩子"""
        if not self.draw:
            return
            
        if self.every_n_train_iters(runner, self.interval):
            # 简单的可视化逻辑
            runner.logger.info(f"可视化钩子在第 {runner.iter} 次迭代后执行")
            
    def after_val_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        """验证迭代后的钩子"""
        if not self.draw:
            return
            
        if self.every_n_inner_iters(batch_idx, self.interval):
            # 简单的可视化逻辑
            runner.logger.info(f"验证可视化钩子在第 {batch_idx} 个batch后执行")

print("✅ 创建并注册了SegVisualizationHook到MMEngine HOOKS注册表")

__all__ = ['SegVisualizationHook']