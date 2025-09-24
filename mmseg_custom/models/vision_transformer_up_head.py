"""Vision Transformer Upsampling Head for semantic segmentation.

This module implements a decode head specifically designed for Vision Transformer
backbones, with upsampling layers to restore spatial resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from mmcv.cnn import build_norm_layer, build_activation_layer
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class VisionTransformerUpHead(BaseModule):
    """Vision Transformer Upsampling Head.
    
    This head is designed for Vision Transformer backbones that output
    low-resolution feature maps. It uses upsampling layers to restore
    the spatial resolution for dense prediction tasks.
    
    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of intermediate channels.
        in_index (int): Index of input feature from backbone.
        img_size (int | tuple): Input image size.
        embed_dims (int): Embedding dimensions of the ViT backbone.
        num_classes (int): Number of classes for segmentation.
        norm_cfg (dict): Config for normalization layers.
        num_conv (int): Number of convolution layers in the head.
        upsampling_method (str): Method for upsampling ('bilinear', 'nearest', 'deconv').
        num_upsample_layer (int): Number of upsampling layers.
        align_corners (bool): Whether to align corners in interpolation.
        loss_decode (dict): Config for decode loss.
        init_cfg (dict, optional): Initialization config.
    """
    
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 in_index: int = -1,
                 img_size: Union[int, Tuple[int, int]] = 224,
                 embed_dims: int = 768,
                 num_classes: int = 19,
                 norm_cfg: dict = dict(type='SyncBN', requires_grad=True),
                 num_conv: int = 2,
                 upsampling_method: str = 'bilinear',
                 num_upsample_layer: int = 2,
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
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.num_conv = num_conv
        self.upsampling_method = upsampling_method
        self.num_upsample_layer = num_upsample_layer
        self.align_corners = align_corners
        self.loss_decode_cfg = loss_decode
        
        # Build loss function
        self.loss_decode = MODELS.build(loss_decode)
        
        # Build the decode head layers
        self._build_decode_layers()
        
    def _build_decode_layers(self):
        """Build decode layers."""
        # Input projection layer
        self.input_proj = nn.Conv2d(
            self.in_channels, 
            self.channels, 
            kernel_size=1
        )
        
        # Intermediate convolution layers
        conv_layers = []
        for i in range(self.num_conv):
            conv_layers.extend([
                nn.Conv2d(
                    self.channels, 
                    self.channels, 
                    kernel_size=3, 
                    padding=1
                ),
                build_norm_layer(self.norm_cfg, self.channels)[1],
                nn.ReLU(inplace=True)
            ])
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Upsampling layers
        if self.upsampling_method == 'deconv':
            # Use transposed convolution for upsampling
            upsample_layers = []
            current_channels = self.channels
            
            for i in range(self.num_upsample_layer):
                upsample_layers.extend([
                    nn.ConvTranspose2d(
                        current_channels,
                        current_channels // 2 if i < self.num_upsample_layer - 1 else current_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    build_norm_layer(self.norm_cfg, current_channels // 2 if i < self.num_upsample_layer - 1 else current_channels)[1],
                    nn.ReLU(inplace=True)
                ])
                if i < self.num_upsample_layer - 1:
                    current_channels = current_channels // 2
                    
            self.upsample_layers = nn.Sequential(*upsample_layers)
        else:
            # Use interpolation for upsampling
            self.upsample_layers = None
            
        # Final classification layer
        self.cls_seg = nn.Conv2d(
            self.channels, 
            self.num_classes, 
            kernel_size=1
        )
        
        # Dropout layer
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, inputs):
        """Forward function.
        
        Args:
            inputs (list[Tensor] | Tensor): Input features from backbone.
            
        Returns:
            Tensor: Output segmentation logits.
        """
        if isinstance(inputs, (list, tuple)):
            x = inputs[self.in_index]
        else:
            x = inputs
            
        # Handle ViT output format
        if x.dim() == 3:
            # [B, N, C] -> [B, C, H, W]
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Input projection
        x = self.input_proj(x)
        
        # Intermediate convolutions
        x = self.conv_layers(x)
        
        # Upsampling
        if self.upsample_layers is not None:
            # Use transposed convolution
            x = self.upsample_layers(x)
        else:
            # Use interpolation
            target_size = (self.img_size[0] // 4, self.img_size[1] // 4)  # 4x upsampling
            for _ in range(self.num_upsample_layer):
                target_size = (target_size[0] * 2, target_size[1] * 2)
                x = F.interpolate(
                    x, 
                    size=target_size, 
                    mode=self.upsampling_method, 
                    align_corners=self.align_corners
                )
        
        # Apply dropout
        x = self.dropout(x)
        
        # Final classification
        x = self.cls_seg(x)
        
        # Ensure output matches input image size
        if x.shape[2:] != self.img_size:
            x = F.interpolate(
                x, 
                size=self.img_size, 
                mode='bilinear', 
                align_corners=self.align_corners
            )
        
        return x
    
    def loss(self, inputs, batch_data_samples, train_cfg):
        """Compute segmentation loss.
        
        Args:
            inputs: Input features from backbone.
            batch_data_samples: Batch of data samples with ground truth.
            train_cfg: Training config.
            
        Returns:
            dict: Loss dictionary.
        """
        seg_logits = self.forward(inputs)
        
        # Extract ground truth labels
        seg_labels = []
        for data_sample in batch_data_samples:
            if hasattr(data_sample, 'gt_sem_seg'):
                if hasattr(data_sample.gt_sem_seg, 'data'):
                    seg_labels.append(data_sample.gt_sem_seg.data)
                else:
                    seg_labels.append(data_sample.gt_sem_seg)
            else:
                # Fallback: create dummy labels
                seg_labels.append(torch.zeros(
                    seg_logits.shape[2:], 
                    dtype=torch.long, 
                    device=seg_logits.device
                ))
        
        seg_labels = torch.stack(seg_labels, dim=0)
        
        # Compute loss
        loss = self.loss_decode(seg_logits, seg_labels)
        
        return {'loss_seg': loss}
    
    def predict(self, inputs, batch_img_metas, test_cfg):
        """Predict segmentation results.
        
        Args:
            inputs: Input features from backbone.
            batch_img_metas: Batch of image meta information.
            test_cfg: Test config.
            
        Returns:
            Tensor: Segmentation predictions.
        """
        seg_logits = self.forward(inputs)
        return seg_logits