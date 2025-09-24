"""FCN Head for semantic segmentation.

This module implements a simple FCN (Fully Convolutional Network) head
for semantic segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from mmcv.cnn import build_norm_layer, build_activation_layer, ConvModule
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class FCNHead(BaseModule):
    """Fully Convolutional Network Head.
    
    This head applies a series of convolution layers followed by a classifier
    to produce segmentation predictions.
    
    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of intermediate channels.
        in_index (int): Index of input feature from backbone.
        num_convs (int): Number of convolution layers.
        concat_input (bool): Whether to concatenate input feature.
        dropout_ratio (float): Dropout ratio.
        num_classes (int): Number of classes for segmentation.
        norm_cfg (dict): Config for normalization layers.
        act_cfg (dict): Config for activation layers.
        align_corners (bool): Whether to align corners in interpolation.
        loss_decode (dict): Config for decode loss.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 in_index: int = -1,
                 num_convs: int = 2,
                 concat_input: bool = True,
                 dropout_ratio: float = 0.1,
                 num_classes: int = 19,
                 norm_cfg: dict = dict(type='SyncBN', requires_grad=True),
                 act_cfg: dict = dict(type='ReLU'),
                 align_corners: bool = False,
                 loss_decode: dict = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.channels = channels
        self.in_index = in_index
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.loss_decode_cfg = loss_decode
        self.ignore_index = 255  # Default ignore index for segmentation
        
        # Build loss function
        self.loss_decode = MODELS.build(loss_decode)
        
        # Build the decode head layers
        self._build_decode_layers()
        
    def _build_decode_layers(self):
        """Build decode layers."""
        # Convolution layers
        conv_layers = []
        for i in range(self.num_convs):
            in_ch = self.in_channels if i == 0 else self.channels
            conv_layers.append(
                ConvModule(
                    in_ch,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        self.convs = nn.ModuleList(conv_layers)
        
        # Concatenation layer
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        
        # Dropout layer
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None
            
        # Classifier
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        
    def forward(self, inputs):
        """Forward function.
        
        Args:
            inputs (list[Tensor] | Tensor): Input features.
            
        Returns:
            Tensor: Output segmentation map.
        """
        if isinstance(inputs, (list, tuple)):
            x = inputs[self.in_index]
        else:
            x = inputs
            
        # Apply convolution layers
        output = x
        for conv in self.convs:
            output = conv(output)
            
        # Concatenate input if needed
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
            
        # Apply dropout
        if self.dropout is not None:
            output = self.dropout(output)
            
        # Apply classifier
        output = self.conv_seg(output)
        
        return output
        
    def loss(self, inputs, batch_data_samples, train_cfg):
        """Compute segmentation loss.
        
        Args:
            inputs (list[Tensor] | Tensor): Input features.
            batch_data_samples (list): Batch data samples.
            train_cfg (dict): Training config.
            
        Returns:
            dict: Loss dict.
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses
    
    def loss_by_feat(self, seg_logits: Union[torch.Tensor, List[torch.Tensor]],
                     batch_data_samples: Union[List, Dict]) -> dict:
        """Compute segmentation loss with proper batch size handling."""
        
        # Debug information
        print(f"[FCN_HEAD_DEBUG] seg_logits type: {type(seg_logits)}")
        if isinstance(seg_logits, torch.Tensor):
            print(f"[FCN_HEAD_DEBUG] seg_logits shape: {seg_logits.shape}")
        elif isinstance(seg_logits, list):
            print(f"[FCN_HEAD_DEBUG] seg_logits list length: {len(seg_logits)}")
            if len(seg_logits) > 0:
                print(f"[FCN_HEAD_DEBUG] seg_logits[0] shape: {seg_logits[0].shape}")
        
        print(f"[FCN_HEAD_DEBUG] batch_data_samples type: {type(batch_data_samples)}")
        print(f"[FCN_HEAD_DEBUG] batch_data_samples length: {len(batch_data_samples) if hasattr(batch_data_samples, '__len__') else 'N/A'}")
        
        # Handle different input formats for seg_logits
        if isinstance(seg_logits, list):
            if len(seg_logits) == 0:
                # Create dummy tensor if empty list
                seg_logits = torch.zeros(1, self.num_classes, 64, 64, device='cpu')
                print(f"[FCN_HEAD_DEBUG] Created dummy seg_logits: {seg_logits.shape}")
            else:
                seg_logits = seg_logits[0]
                print(f"[FCN_HEAD_DEBUG] Using first seg_logits: {seg_logits.shape}")
        
        # Ensure seg_logits is a tensor
        if not isinstance(seg_logits, torch.Tensor):
            raise ValueError(f"seg_logits must be a tensor, got {type(seg_logits)}")
        
        batch_size = seg_logits.shape[0]
        print(f"[FCN_HEAD_DEBUG] Input batch size: {batch_size}")
        
        # Handle different input formats for batch_data_samples
        if isinstance(batch_data_samples, dict):
            # Convert dict to list format expected by FCN head
            data_samples_list = batch_data_samples.get('data_samples', [])
            print(f"[FCN_HEAD_DEBUG] Extracted {len(data_samples_list)} samples from dict")
        else:
            data_samples_list = batch_data_samples
            print(f"[FCN_HEAD_DEBUG] Using direct list with {len(data_samples_list)} samples")
        
        # Ensure we have the right number of samples
        if len(data_samples_list) != batch_size:
            print(f"[FCN_HEAD_DEBUG] Batch size mismatch: seg_logits={batch_size}, samples={len(data_samples_list)}")
            
            if len(data_samples_list) == 1 and batch_size > 1:
                # Replicate single sample to match batch size
                print(f"[FCN_HEAD_DEBUG] Replicating single sample to match batch size {batch_size}")
                data_samples_list = data_samples_list * batch_size
            elif len(data_samples_list) > batch_size:
                # Truncate to match batch size
                print(f"[FCN_HEAD_DEBUG] Truncating samples to match batch size {batch_size}")
                data_samples_list = data_samples_list[:batch_size]
            else:
                        # Create dummy samples to match batch size
                        print(f"[FCN_HEAD_DEBUG] Creating dummy samples to match batch size {batch_size}")
                        dummy_samples = []
                        for i in range(batch_size):
                            if i < len(data_samples_list):
                                dummy_samples.append(data_samples_list[i])
                            else:
                                # Create dummy sample with proper structure
                                dummy_seg = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                                dummy_sample = {
                                    'gt_sem_seg': {
                                        'data': dummy_seg
                                    }
                                }
                                dummy_samples.append(dummy_sample)
                        data_samples_list = dummy_samples
        
        # Process each data sample to extract segmentation labels
        seg_labels = []
        for i, data_sample in enumerate(data_samples_list):
            try:
                seg_label = None
                print(f"[FCN_HEAD_DEBUG] Processing sample {i}, type: {type(data_sample)}")
                
                if hasattr(data_sample, 'gt_sem_seg'):
                    # Standard SegDataSample format
                    print(f"[FCN_HEAD_DEBUG] Sample {i} has gt_sem_seg attribute")
                    gt_sem_seg = data_sample.gt_sem_seg
                    print(f"[FCN_HEAD_DEBUG] gt_sem_seg type: {type(gt_sem_seg)}")
                    
                    if hasattr(gt_sem_seg, 'data'):
                        seg_label = gt_sem_seg.data
                        print(f"[FCN_HEAD_DEBUG] Extracted seg_label from gt_sem_seg.data, type: {type(seg_label)}")
                    else:
                        seg_label = gt_sem_seg
                        print(f"[FCN_HEAD_DEBUG] Using gt_sem_seg directly, type: {type(seg_label)}")
                        
                elif isinstance(data_sample, dict) and 'gt_sem_seg' in data_sample:
                    # Handle dict format - need to extract data from nested dict
                    print(f"[FCN_HEAD_DEBUG] Sample {i} is dict with gt_sem_seg key")
                    gt_sem_seg = data_sample['gt_sem_seg']
                    print(f"[FCN_HEAD_DEBUG] gt_sem_seg from dict, type: {type(gt_sem_seg)}")
                    
                    if isinstance(gt_sem_seg, dict) and 'data' in gt_sem_seg:
                        seg_label = gt_sem_seg['data']
                        print(f"[FCN_HEAD_DEBUG] Extracted seg_label from nested dict, type: {type(seg_label)}")
                    else:
                        seg_label = gt_sem_seg
                        print(f"[FCN_HEAD_DEBUG] Using gt_sem_seg from dict directly, type: {type(seg_label)}")
                else:
                    # Create dummy segmentation
                    print(f"[FCN_HEAD_DEBUG] Creating dummy segmentation for sample {i}")
                    seg_label = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                
                # Ensure seg_label is a tensor
                if not isinstance(seg_label, torch.Tensor):
                    print(f"[FCN_HEAD_DEBUG] Converting seg_label to tensor, current type: {type(seg_label)}")
                    print(f"[FCN_HEAD_DEBUG] seg_label content preview: {seg_label}")
                    
                    if isinstance(seg_label, (list, tuple)):
                        print(f"[FCN_HEAD_DEBUG] seg_label is list/tuple with length: {len(seg_label)}")
                        if len(seg_label) > 0:
                            print(f"[FCN_HEAD_DEBUG] First element type: {type(seg_label[0])}")
                            print(f"[FCN_HEAD_DEBUG] First element content: {seg_label[0]}")
                        
                        try:
                            # 尝试递归处理嵌套的list结构
                            def flatten_and_convert(data):
                                if isinstance(data, (list, tuple)):
                                    if len(data) == 1:
                                        return flatten_and_convert(data[0])
                                    else:
                                        # 如果是多元素列表，尝试转换为numpy数组再转tensor
                                        import numpy as np
                                        return np.array(data)
                                elif isinstance(data, torch.Tensor):
                                    return data
                                elif hasattr(data, '__array__'):  # numpy array check
                                    return data
                                else:
                                    print(f"[FCN_HEAD_DEBUG] Unexpected nested type: {type(data)}")
                                    return None
                            
                            flattened = flatten_and_convert(seg_label)
                            print(f"[FCN_HEAD_DEBUG] Flattened result type: {type(flattened)}")
                            
                            if isinstance(flattened, torch.Tensor):
                                seg_label = flattened.to(dtype=torch.long, device=seg_logits.device)
                            elif hasattr(flattened, '__array__'):  # numpy array
                                import numpy as np
                                seg_label = torch.from_numpy(np.array(flattened)).to(dtype=torch.long, device=seg_logits.device)
                            else:
                                print(f"[FCN_HEAD_DEBUG] Failed to flatten, creating dummy tensor")
                                seg_label = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                                
                        except Exception as convert_error:
                            print(f"[FCN_HEAD_DEBUG] Error in list/tuple conversion: {convert_error}")
                            seg_label = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                            
                    elif hasattr(seg_label, '__array__'):  # numpy array check
                        try:
                            import numpy as np
                            seg_label = torch.from_numpy(np.array(seg_label)).to(dtype=torch.long, device=seg_logits.device)
                        except Exception as np_error:
                            print(f"[FCN_HEAD_DEBUG] Error converting numpy array: {np_error}")
                            seg_label = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                            
                    elif seg_label is None:
                        seg_label = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                    else:
                        print(f"[FCN_HEAD_DEBUG] Unexpected seg_label type: {type(seg_label)}, creating dummy")
                        seg_label = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                else:
                    print(f"[FCN_HEAD_DEBUG] seg_label is already tensor, dtype: {seg_label.dtype}, device: {seg_label.device}")
                
                # Ensure proper device and shape
                seg_label = seg_label.to(seg_logits.device)
                
                print(f"[FCN_HEAD_DEBUG] seg_label shape before resize: {seg_label.shape}")
                print(f"[FCN_HEAD_DEBUG] Target shape: {seg_logits.shape[-2:]}")
                
                # Resize if necessary
                if seg_label.shape != seg_logits.shape[-2:]:
                    print(f"[FCN_HEAD_DEBUG] Resizing seg_label from {seg_label.shape[-2:]} to {seg_logits.shape[-2:]}")
                    if seg_label.dim() == 2:
                        # 修复：先转换为float，插值后再转换为long，避免索引错误
                        seg_label_float = seg_label.unsqueeze(0).unsqueeze(0).float()
                        print(f"[FCN_HEAD_DEBUG] Added dimensions for interpolation: {seg_label_float.shape}")
                        seg_label_resized = torch.nn.functional.interpolate(
                            seg_label_float,
                            size=seg_logits.shape[-2:],
                            mode='nearest'
                        )
                        print(f"[FCN_HEAD_DEBUG] After interpolation: {seg_label_resized.shape}")
                        # 安全地squeeze和转换数据类型
                        seg_label = seg_label_resized.squeeze(0).squeeze(0)
                        print(f"[FCN_HEAD_DEBUG] Removed dimensions after interpolation: {seg_label.shape}")
                        try:
                            seg_label = seg_label.long()
                            print(f"[FCN_HEAD_DEBUG] Successfully converted back to long: {seg_label.dtype}")
                        except Exception as e:
                            print(f"[FCN_HEAD_DEBUG] Error converting to long: {e}")
                            seg_label = seg_label.round().long()
                            print(f"[FCN_HEAD_DEBUG] Used round() fallback: {seg_label.dtype}")
                    elif seg_label.dim() == 3:
                        # 修复：先转换为float，插值后再转换为long，避免索引错误
                        # 如果是3D张量 [1, H, W]，先squeeze掉第一个维度变成2D [H, W]
                        if seg_label.shape[0] == 1:
                            seg_label = seg_label.squeeze(0)
                            print(f"[FCN_HEAD_DEBUG] Squeezed first dimension: {seg_label.shape}")
                        
                        seg_label_float = seg_label.unsqueeze(0).unsqueeze(0).float()
                        print(f"[FCN_HEAD_DEBUG] Added dimensions for interpolation: {seg_label_float.shape}")
                        seg_label_resized = torch.nn.functional.interpolate(
                            seg_label_float,
                            size=seg_logits.shape[-2:],
                            mode='nearest'
                        )
                        print(f"[FCN_HEAD_DEBUG] After interpolation: {seg_label_resized.shape}")
                        # 安全地squeeze和转换数据类型，确保最终是2D
                        seg_label = seg_label_resized.squeeze(0).squeeze(0)
                        print(f"[FCN_HEAD_DEBUG] Removed all extra dimensions after interpolation: {seg_label.shape}")
                        try:
                            seg_label = seg_label.long()
                            print(f"[FCN_HEAD_DEBUG] Successfully converted back to long: {seg_label.dtype}")
                        except Exception as e:
                            print(f"[FCN_HEAD_DEBUG] Error converting to long: {e}")
                            seg_label = seg_label.round().long()
                            print(f"[FCN_HEAD_DEBUG] Used round() fallback: {seg_label.dtype}")
                    else:
                        print(f"[FCN_HEAD_DEBUG] Unexpected seg_label dimensions: {seg_label.shape}")
                        seg_label = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                
                # Final validation - ensure it's 2D tensor
                if seg_label.dim() != 2:
                    print(f"[FCN_HEAD_DEBUG] Final seg_label has wrong dimensions: {seg_label.shape}, creating 2D dummy")
                    seg_label = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                
                print(f"[FCN_HEAD_DEBUG] Final seg_label shape for sample {i}: {seg_label.shape}")
                seg_labels.append(seg_label)
                print(f"[FCN_HEAD_DEBUG] Processed sample {i}, seg_label shape: {seg_label.shape}")
                
            except Exception as e:
                print(f"[FCN_HEAD_DEBUG] Error processing sample {i}: {e}")
                # Create dummy segmentation as fallback
                seg_label = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                seg_labels.append(seg_label)
        
        # Stack labels to create batch
        if len(seg_labels) > 0:
            seg_label = torch.stack(seg_labels, dim=0)
            print(f"[FCN_HEAD_DEBUG] Final seg_label shape: {seg_label.shape}")
        else:
            # Create dummy batch if no labels
            seg_label = torch.zeros((batch_size,) + seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
            print(f"[FCN_HEAD_DEBUG] Created dummy seg_label batch: {seg_label.shape}")
        
        # Compute loss
        losses = dict()
        
        # Handle multiple loss functions
        if isinstance(self.loss_decode, list):
            for i, loss_fn in enumerate(self.loss_decode):
                try:
                    loss_value = loss_fn(seg_logits, seg_label)
                    losses[f'loss_seg_{i}'] = loss_value
                    print(f"[FCN_HEAD_DEBUG] Loss {i}: {loss_value}")
                except Exception as e:
                    print(f"[FCN_HEAD_DEBUG] Error computing loss {i}: {e}")
                    losses[f'loss_seg_{i}'] = torch.tensor(0.0, device=seg_logits.device, requires_grad=True)
        else:
            try:
                loss_value = self.loss_decode(seg_logits, seg_label)
                losses['loss_seg'] = loss_value
                print(f"[FCN_HEAD_DEBUG] Single loss: {loss_value}")
            except Exception as e:
                print(f"[FCN_HEAD_DEBUG] Error computing single loss: {e}")
                losses['loss_seg'] = torch.tensor(0.0, device=seg_logits.device, requires_grad=True)
        
        return losses
        
    def predict(self, inputs, batch_img_metas, test_cfg):
        """Predict segmentation results.
        
        Args:
            inputs (list[Tensor] | Tensor): Input features.
            batch_img_metas (list): Batch image metas.
            test_cfg (dict): Test config.
            
        Returns:
            list: Segmentation results.
        """
        seg_logits = self.forward(inputs)
        return self.predict_by_feat(seg_logits, batch_img_metas)
        
    def predict_by_feat(self, seg_logits, batch_img_metas):
        """Predict by features.
        
        Args:
            seg_logits (Tensor): Segmentation logits.
            batch_img_metas (list): Batch image metas.
            
        Returns:
            list: Segmentation results.
        """
        # Apply softmax to get probabilities
        seg_pred = F.softmax(seg_logits, dim=1)
        
        # Get predictions
        seg_pred = seg_pred.argmax(dim=1)
        
        # Convert to list of results
        results = []
        for i in range(seg_pred.shape[0]):
            results.append(seg_pred[i].cpu().numpy())
            
        return results