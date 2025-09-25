#!/usr/bin/env python3
"""
自定义的EncoderDecoder模型，用于解决T20服务器上的注册表问题
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from mmengine.model import BaseModel
from mmengine.registry import MODELS


@MODELS.register_module(name='EncoderDecoder')
@MODELS.register_module(name='CustomEncoderDecoder')
class EncoderDecoder(BaseModel):
    """自定义的EncoderDecoder模型，兼容T20服务器环境"""
    
    def __init__(self, 
                 backbone: Optional[Dict] = None,
                 decode_head: Optional[Dict] = None,
                 neck: Optional[Dict] = None,
                 auxiliary_head: Optional[Dict] = None,
                 train_cfg: Optional[Dict] = None,
                 test_cfg: Optional[Dict] = None,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[Dict] = None,
                 data_preprocessor: Optional[Dict] = None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        
        # 存储配置以供后续使用
        self.backbone_cfg = backbone
        self.decode_head_cfg = decode_head
        self.neck_cfg = neck
        self.auxiliary_head_cfg = auxiliary_head
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # 构建模型组件
        if backbone is not None:
            self.backbone = MODELS.build(backbone)
        else:
            self.backbone = nn.Identity()
            
        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None
            
        if decode_head is not None:
            self.decode_head = MODELS.build(decode_head)
        else:
            self.decode_head = nn.Identity()
            
        if auxiliary_head is not None:
            self.auxiliary_head = MODELS.build(auxiliary_head)
        else:
            self.auxiliary_head = None
    
    def extract_feat(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """提取特征"""
        x = self.backbone(inputs)
        if self.neck is not None:
            x = self.neck(x)
        # 确保返回List[torch.Tensor]类型
        if isinstance(x, torch.Tensor):
            return [x]
        elif isinstance(x, (list, tuple)):
            return [feat if isinstance(feat, torch.Tensor) else torch.tensor(feat) for feat in x]
        else:
            return [torch.tensor(x)]
    
    def encode_decode(self, inputs: torch.Tensor, batch_img_metas: List[Dict]) -> torch.Tensor:
        """编码-解码过程"""
        x = self.extract_feat(inputs)
        # 确保decode_head可调用
        if hasattr(self.decode_head, '__call__'):
            seg_logits = self.decode_head(x)
        else:
            # 如果decode_head是Identity或其他不可调用对象，返回输入
            seg_logits = x[0] if x else inputs
        return seg_logits
    
    def forward(self, 
                inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                data_samples: Optional[Any] = None, 
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], List[Any]]:
        """前向传播"""
        
        # 🔧 修复数据输入格式处理 - 兼容多种数据格式
        if isinstance(inputs, dict):
            # 情况1: 标准MMEngine格式 {'inputs': tensor, 'data_samples': [...]}
            if 'inputs' in inputs:
                actual_inputs = inputs['inputs']
                if data_samples is None and 'data_samples' in inputs:
                    data_samples = inputs['data_samples']
                inputs = actual_inputs
            # 情况2: 直接的batch数据格式 {'img': tensor, 'gt_semantic_seg': tensor, ...}
            elif 'img' in inputs:
                actual_inputs = inputs['img']
                # 构造data_samples用于loss计算
                if data_samples is None and 'gt_semantic_seg' in inputs:
                    gt_seg = inputs['gt_semantic_seg']
                    # 构造简单的data_samples格式
                    batch_size = gt_seg.shape[0] if hasattr(gt_seg, 'shape') else 1
                    data_samples = []
                    for i in range(batch_size):
                        # 创建简单的对象来存储分割标注
                        sample = {}
                        sample['gt_sem_seg'] = {}
                        sample['gt_sem_seg']['data'] = gt_seg[i] if batch_size > 1 else gt_seg
                        sample['metainfo'] = inputs.get('img_metas', {})
                        data_samples.append(sample)
                inputs = actual_inputs
            else:
                # 如果是其他格式的dict，尝试找到图像数据
                possible_keys = ['image', 'images', 'input', 'x']
                for key in possible_keys:
                    if key in inputs:
                        inputs = inputs[key]
                        break
                else:
                    raise KeyError(f"Cannot find image data in input dict. Available keys: {list(inputs.keys())}")
        
        # 如果inputs仍然不是tensor，尝试转换
        if not isinstance(inputs, torch.Tensor):
            if hasattr(inputs, 'data'):
                inputs = inputs.data
            elif isinstance(inputs, (list, tuple)) and len(inputs) > 0:
                inputs = inputs[0]
        
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            result = self.predict(inputs, data_samples)
            # 确保返回类型符合BaseModel要求
            return result if isinstance(result, list) else [result]
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                             'Only supports loss, predict and tensor mode')
    
    def loss(self, inputs: torch.Tensor, data_samples: Any) -> Dict[str, torch.Tensor]:
        """计算损失"""
        x = self.extract_feat(inputs)
        
        losses: Dict[str, torch.Tensor] = {}
        
        # 主解码头损失
        if hasattr(self.decode_head, 'loss_by_feat') and callable(self.decode_head.loss_by_feat):
            loss_decode = self.decode_head.loss_by_feat(x, data_samples)
        elif hasattr(self.decode_head, 'loss') and callable(self.decode_head.loss):
            # 传递train_cfg参数
            loss_decode = self.decode_head.loss(x, data_samples, self.train_cfg)
        else:
            # 简单的占位符损失
            loss_decode = {'loss_seg': torch.tensor(0.0, requires_grad=True, device=inputs.device)}
        
        if isinstance(loss_decode, dict):
            losses.update(loss_decode)
        
        # 辅助解码头损失
        if self.auxiliary_head is not None:
            if hasattr(self.auxiliary_head, 'loss_by_feat') and callable(self.auxiliary_head.loss_by_feat):
                loss_aux = self.auxiliary_head.loss_by_feat(x, data_samples)
            elif hasattr(self.auxiliary_head, 'loss') and callable(self.auxiliary_head.loss):
                # 传递train_cfg参数
                loss_aux = self.auxiliary_head.loss(x, data_samples, self.train_cfg)
            else:
                loss_aux = {'loss_aux': torch.tensor(0.0, requires_grad=True, device=inputs.device)}
            
            if isinstance(loss_aux, dict):
                losses.update(loss_aux)
        
        return losses
    
    def predict(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], data_samples: Any) -> Any:
        """预测"""
        # 处理data_preprocessor的输出格式
        if isinstance(inputs, dict):
            inputs = inputs['inputs']
            
        batch_img_metas = []
        if data_samples is not None:
            if hasattr(data_samples, '__iter__'):
                for sample in data_samples:
                    if hasattr(sample, 'metainfo'):
                        batch_img_metas.append(sample.metainfo)
                    else:
                        batch_img_metas.append({})
            else:
                batch_img_metas = [{}]
        
        seg_logits = self.encode_decode(inputs, batch_img_metas)
        
        # 简单的预测结果处理
        if data_samples is not None:
            # 将预测结果添加到data_samples中
            if hasattr(data_samples, '__iter__'):
                for i, sample in enumerate(data_samples):
                    if hasattr(sample, 'pred_sem_seg'):
                        # 假设seg_logits的形状为[B, C, H, W]
                        if i < seg_logits.shape[0]:
                            pred_mask = seg_logits[i].argmax(dim=0)
                            sample.pred_sem_seg.data = pred_mask
            return data_samples
        else:
            return seg_logits
    
    def _forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], data_samples: Optional[Any] = None) -> torch.Tensor:
        """内部前向传播（用于推理）"""
        # 处理data_preprocessor的输出格式
        if isinstance(inputs, dict):
            inputs = inputs['inputs']
            
        return self.encode_decode(inputs, [])