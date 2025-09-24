# 导入MMSeg的损失函数并注册到MMEngine的MODELS注册表中
from mmengine.registry import MODELS

# 初始化可用的损失函数列表
_available_losses = []

# 导入MMSeg的损失函数
try:
    from mmseg.models.losses import CrossEntropyLoss, DiceLoss, FocalLoss, LovaszLoss
    
    # 注册损失函数到MMEngine的MODELS注册表
    MODELS.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss, force=True)
    MODELS.register_module(name='DiceLoss', module=DiceLoss, force=True)
    MODELS.register_module(name='FocalLoss', module=FocalLoss, force=True)
    MODELS.register_module(name='LovaszLoss', module=LovaszLoss, force=True)
    
    # 添加到可用损失函数列表
    _available_losses.extend(['CrossEntropyLoss', 'DiceLoss', 'FocalLoss', 'LovaszLoss'])
    
    print("✅ 成功注册MMSeg损失函数到MMEngine MODELS注册表")
    
except ImportError as e:
    print(f"⚠️ 无法导入MMSeg损失函数: {e}")
    
    # 如果无法导入MMSeg损失函数，创建简单的替代实现
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SimpleCrossEntropyLoss(nn.Module):
        """简单的CrossEntropyLoss实现"""
        def __init__(self, use_sigmoid=False, loss_weight=1.0, ignore_index=255, **kwargs):
            super().__init__()
            self.use_sigmoid = use_sigmoid
            self.loss_weight = loss_weight
            self.ignore_index = ignore_index
            
        def forward(self, pred, target):
            if self.use_sigmoid:
                loss = F.binary_cross_entropy_with_logits(pred, target.float())
            else:
                loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index)
            return loss * self.loss_weight
    
    # 注册简单实现并添加到当前模块
    MODELS.register_module(name='CrossEntropyLoss', module=SimpleCrossEntropyLoss, force=True)
    CrossEntropyLoss = SimpleCrossEntropyLoss
    _available_losses.append('CrossEntropyLoss')
    
    print("✅ 注册简单CrossEntropyLoss实现到MMEngine MODELS注册表")

# 动态设置__all__
__all__ = _available_losses