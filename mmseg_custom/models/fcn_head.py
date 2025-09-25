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
            batch_data_samples (list): Batch data samples containing ground truth.
            train_cfg (dict): Training config.
            
        Returns:
            dict: Loss dict.
        """
        seg_logits = self.forward(inputs)
        
        # Validate inputs
        if isinstance(seg_logits, list):
            if len(seg_logits) == 0:
                raise ValueError("seg_logits cannot be empty")
            seg_logits = seg_logits[0]
        
        if not isinstance(seg_logits, torch.Tensor):
            raise ValueError(f"seg_logits must be a tensor, got {type(seg_logits)}")
        
        batch_size = seg_logits.shape[0]
        
        # Handle different input formats for batch_data_samples
        if isinstance(batch_data_samples, dict):
            data_samples_list = batch_data_samples.get('data_samples', [])
        else:
            data_samples_list = batch_data_samples
        
        # Validate batch size consistency
        if len(data_samples_list) != batch_size:
            raise ValueError(f"Batch size mismatch: seg_logits has {batch_size} samples, "
                           f"but batch_data_samples has {len(data_samples_list)} samples")
        
        # Process each data sample to extract segmentation labels
        seg_labels = []
        for i, data_sample in enumerate(data_samples_list):
            seg_label = None
            
            if hasattr(data_sample, 'gt_sem_seg'):
                # Standard SegDataSample format
                gt_sem_seg = data_sample.gt_sem_seg
                if hasattr(gt_sem_seg, 'data'):
                    seg_label = gt_sem_seg.data
                else:
                    seg_label = gt_sem_seg
                    
            elif isinstance(data_sample, dict) and 'gt_sem_seg' in data_sample:
                # Handle dict format
                gt_sem_seg = data_sample['gt_sem_seg']
                if isinstance(gt_sem_seg, dict) and 'data' in gt_sem_seg:
                    seg_label = gt_sem_seg['data']
                else:
                    seg_label = gt_sem_seg
            else:
                raise ValueError(f"Sample {i} does not contain valid ground truth segmentation data")
            
            # Convert to tensor if needed
            if not isinstance(seg_label, torch.Tensor):
                if isinstance(seg_label, (list, tuple)):
                    # Handle nested list/tuple structures
                    def flatten_and_convert(data):
                        if isinstance(data, (list, tuple)):
                            if len(data) == 1:
                                return flatten_and_convert(data[0])
                            else:
                                import numpy as np
                                return np.array(data)
                        elif isinstance(data, torch.Tensor):
                            return data
                        elif hasattr(data, '__array__'):
                            return data
                        else:
                            raise ValueError(f"Cannot convert data of type {type(data)} to tensor")
                    
                    flattened = flatten_and_convert(seg_label)
                    if isinstance(flattened, torch.Tensor):
                        seg_label = flattened.to(dtype=torch.long, device=seg_logits.device)
                    elif hasattr(flattened, '__array__'):
                        import numpy as np
                        seg_label = torch.from_numpy(np.array(flattened)).to(dtype=torch.long, device=seg_logits.device)
                    else:
                        raise ValueError(f"Failed to convert seg_label to tensor")
                        
                elif hasattr(seg_label, '__array__'):
                    import numpy as np
                    seg_label = torch.from_numpy(np.array(seg_label)).to(dtype=torch.long, device=seg_logits.device)
                elif seg_label is None:
                    raise ValueError(f"Sample {i} has None as ground truth segmentation")
                else:
                    raise ValueError(f"Cannot convert seg_label of type {type(seg_label)} to tensor")
            
            # Ensure proper device and dtype
            seg_label = seg_label.to(device=seg_logits.device, dtype=torch.long)
            
            # Resize if necessary
            if seg_label.shape[-2:] != seg_logits.shape[-2:]:
                if seg_label.dim() == 2:
                    seg_label_float = seg_label.unsqueeze(0).unsqueeze(0).float()
                    seg_label_resized = torch.nn.functional.interpolate(
                        seg_label_float,
                        size=seg_logits.shape[-2:],
                        mode='nearest'
                    )
                    seg_label = seg_label_resized.squeeze(0).squeeze(0).long()
                elif seg_label.dim() == 3:
                    if seg_label.shape[0] == 1:
                        seg_label = seg_label.squeeze(0)
                    seg_label_float = seg_label.unsqueeze(0).unsqueeze(0).float()
                    seg_label_resized = torch.nn.functional.interpolate(
                        seg_label_float,
                        size=seg_logits.shape[-2:],
                        mode='nearest'
                    )
                    seg_label = seg_label_resized.squeeze(0).squeeze(0).long()
                else:
                    raise ValueError(f"Unexpected seg_label dimensions: {seg_label.shape}")
            
            # Ensure it's 2D tensor
            if seg_label.dim() != 2:
                raise ValueError(f"seg_label must be 2D, got shape {seg_label.shape}")
            
            seg_labels.append(seg_label)
        
        # Stack labels to create batch
        seg_label = torch.stack(seg_labels, dim=0)
        
        # Compute loss
        losses = dict()
        
        if isinstance(self.loss_decode, list):
            for i, loss_fn in enumerate(self.loss_decode):
                loss_value = loss_fn(seg_logits, seg_label)
                losses[f'loss_seg_{i}'] = loss_value
        else:
            loss_value = self.loss_decode(seg_logits, seg_label)
            losses['loss_seg'] = loss_value
        
        return losses
    
    def loss_by_feat(self, seg_logits: Union[torch.Tensor, List[torch.Tensor]],
                     batch_data_samples: Union[List, Dict]) -> dict:
        """Compute segmentation loss by features.
        
        Args:
            seg_logits (Tensor | List[Tensor]): Segmentation logits.
            batch_data_samples (List | Dict): Batch data samples containing ground truth.
            
        Returns:
            dict: Loss dict.
        """
        # Handle different input formats for seg_logits
        if isinstance(seg_logits, list):
            if len(seg_logits) == 0:
                raise ValueError("seg_logits cannot be empty")
            seg_logits = seg_logits[0]
        
        if not isinstance(seg_logits, torch.Tensor):
            raise ValueError(f"seg_logits must be a tensor, got {type(seg_logits)}")
        
        batch_size = seg_logits.shape[0]
        
        # Handle different input formats for batch_data_samples
        if isinstance(batch_data_samples, dict):
            data_samples_list = batch_data_samples.get('data_samples', [])
        else:
            data_samples_list = batch_data_samples
        
        # Validate batch size consistency
        if len(data_samples_list) != batch_size:
            raise ValueError(f"Batch size mismatch: seg_logits has {batch_size} samples, "
                           f"but batch_data_samples has {len(data_samples_list)} samples")
        
        # Process each data sample to extract segmentation labels
        seg_labels = []
        for i, data_sample in enumerate(data_samples_list):
            seg_label = None
            
            if hasattr(data_sample, 'gt_sem_seg'):
                # Standard SegDataSample format
                gt_sem_seg = data_sample.gt_sem_seg
                if hasattr(gt_sem_seg, 'data'):
                    seg_label = gt_sem_seg.data
                else:
                    seg_label = gt_sem_seg
                    
            elif isinstance(data_sample, dict) and 'gt_sem_seg' in data_sample:
                # Handle dict format
                gt_sem_seg = data_sample['gt_sem_seg']
                if isinstance(gt_sem_seg, dict) and 'data' in gt_sem_seg:
                    seg_label = gt_sem_seg['data']
                else:
                    seg_label = gt_sem_seg
            else:
                raise ValueError(f"Sample {i} does not contain valid ground truth segmentation data")
            
            # Convert to tensor if needed
            if not isinstance(seg_label, torch.Tensor):
                if isinstance(seg_label, (list, tuple)):
                    # Handle nested list/tuple structures
                    def flatten_and_convert(data):
                        if isinstance(data, (list, tuple)):
                            if len(data) == 1:
                                return flatten_and_convert(data[0])
                            else:
                                import numpy as np
                                return np.array(data)
                        elif isinstance(data, torch.Tensor):
                            return data
                        elif hasattr(data, '__array__'):
                            return data
                        else:
                            raise ValueError(f"Cannot convert data of type {type(data)} to tensor")
                    
                    flattened = flatten_and_convert(seg_label)
                    if isinstance(flattened, torch.Tensor):
                        seg_label = flattened.to(dtype=torch.long, device=seg_logits.device)
                    elif hasattr(flattened, '__array__'):
                        import numpy as np
                        seg_label = torch.from_numpy(np.array(flattened)).to(dtype=torch.long, device=seg_logits.device)
                    else:
                        raise ValueError(f"Failed to convert seg_label to tensor")
                        
                elif hasattr(seg_label, '__array__'):
                    import numpy as np
                    seg_label = torch.from_numpy(np.array(seg_label)).to(dtype=torch.long, device=seg_logits.device)
                elif seg_label is None:
                    raise ValueError(f"Sample {i} has None as ground truth segmentation")
                else:
                    raise ValueError(f"Cannot convert seg_label of type {type(seg_label)} to tensor")
            
            # Ensure proper device and dtype
            seg_label = seg_label.to(device=seg_logits.device, dtype=torch.long)
            
            # Resize if necessary
            if seg_label.shape[-2:] != seg_logits.shape[-2:]:
                if seg_label.dim() == 2:
                    seg_label_float = seg_label.unsqueeze(0).unsqueeze(0).float()
                    seg_label_resized = torch.nn.functional.interpolate(
                        seg_label_float,
                        size=seg_logits.shape[-2:],
                        mode='nearest'
                    )
                    seg_label = seg_label_resized.squeeze(0).squeeze(0).long()
                elif seg_label.dim() == 3:
                    if seg_label.shape[0] == 1:
                        seg_label = seg_label.squeeze(0)
                    seg_label_float = seg_label.unsqueeze(0).unsqueeze(0).float()
                    seg_label_resized = torch.nn.functional.interpolate(
                        seg_label_float,
                        size=seg_logits.shape[-2:],
                        mode='nearest'
                    )
                    seg_label = seg_label_resized.squeeze(0).squeeze(0).long()
                else:
                    raise ValueError(f"Unexpected seg_label dimensions: {seg_label.shape}")
            
            # Ensure it's 2D tensor
            if seg_label.dim() != 2:
                raise ValueError(f"seg_label must be 2D, got shape {seg_label.shape}")
            
            seg_labels.append(seg_label)
        
        # Stack labels to create batch
        seg_label = torch.stack(seg_labels, dim=0)
        
        # Compute loss
        losses = dict()
        
        if isinstance(self.loss_decode, list):
            for i, loss_fn in enumerate(self.loss_decode):
                loss_value = loss_fn(seg_logits, seg_label)
                losses[f'loss_seg_{i}'] = loss_value
        else:
            loss_value = self.loss_decode(seg_logits, seg_label)
            losses['loss_seg'] = loss_value
        
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