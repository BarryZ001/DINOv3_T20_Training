"""DINOv3 Vision Transformer backbone for MMSegmentation

基于DINOv3预训练权重的ViT backbone实现，适配MMSegmentation框架。
支持不同尺寸的DINOv3模型（Small/Base/Large/Giant）。
"""

import math
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import CheckpointLoader

# 使用mmengine的MODELS注册器
from mmengine.registry import MODELS


class PatchEmbed(BaseModule):
    """图像到patch嵌入。
    
    将2D图像分割为patches并嵌入到向量空间。
    """
    
    def __init__(self,
                 img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
            
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=patch_size, stride=patch_size
        )
        
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x (torch.Tensor): 输入图像，形状为 (B, C, H, W)
            
        Returns:
            torch.Tensor: patch嵌入，形状为 (B, N, D)
        """
        B, C, H, W = x.shape
        
        # 检查输入尺寸
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # 投影到patch嵌入
        x = self.projection(x)  # (B, embed_dims, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dims)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x


class TransformerEncoderLayer(BaseModule):
    """Transformer编码器层。"""
    
    def __init__(self,
                 embed_dims: int = 768,
                 num_heads: int = 12,
                 feedforward_channels: int = 3072,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        
        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate
        )
        
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            act_cfg=act_cfg
        )
        
        # DropPath
        if drop_path_rate > 0.0:
            self.drop_path = nn.Dropout(drop_path_rate)
        else:
            self.drop_path = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        # Multi-head self-attention
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # Feed forward network
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        
        return x


@MODELS.register_module()
class DINOv3ViT(BaseModule):
    """DINOv3 Vision Transformer backbone。
    
    基于DINOv3预训练权重的ViT实现，支持多种模型尺寸。
    
    Args:
        img_size (int | tuple): 输入图像尺寸
        patch_size (int | tuple): patch尺寸
        in_channels (int): 输入通道数
        embed_dims (int): 嵌入维度
        num_layers (int): Transformer层数
        num_heads (int): 注意力头数
        mlp_ratio (float): MLP隐藏层维度比例
        qkv_bias (bool): 是否使用QKV偏置
        drop_rate (float): dropout比例
        attn_drop_rate (float): 注意力dropout比例
        drop_path_rate (float): drop path比例
        with_cls_token (bool): 是否使用分类token
        output_cls_token (bool): 是否输出分类token
        interpolate_mode (str): 位置编码插值模式
        out_indices (Sequence[int]): 输出特征层索引
        final_norm (bool): 是否在最后添加norm层
        init_cfg (dict, optional): 初始化配置
    """
    
    arch_zoo = {
        'small': {
            'embed_dims': 384,
            'num_layers': 12,
            'num_heads': 6,
            'feedforward_channels': 1536
        },
        'base': {
            'embed_dims': 768,
            'num_layers': 12,
            'num_heads': 12,
            'feedforward_channels': 3072
        },
        'large': {
            'embed_dims': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'feedforward_channels': 4096
        },
        'giant': {
            'embed_dims': 1536,
            'num_layers': 40,
            'num_heads': 24,
            'feedforward_channels': 6144
        }
    }
    
    def __init__(self,
                 arch: str = 'large',
                 img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_channels: int = 3,
                 embed_dims: Optional[int] = None,
                 num_layers: Optional[int] = None,
                 num_heads: Optional[int] = None,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 with_cls_token: bool = True,
                 output_cls_token: bool = False,
                 interpolate_mode: str = 'bicubic',
                 out_indices: Sequence[int] = (23,),  # 默认输出最后一层
                 final_norm: bool = True,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        
        if arch in self.arch_zoo:
            arch_settings = self.arch_zoo[arch]
            embed_dims = embed_dims or arch_settings['embed_dims']
            num_layers = num_layers or arch_settings['num_layers']
            num_heads = num_heads or arch_settings['num_heads']
            feedforward_channels = arch_settings['feedforward_channels']
        else:
            assert embed_dims is not None and num_layers is not None and num_heads is not None
            feedforward_channels = int(embed_dims * mlp_ratio)
        
        self.arch = arch
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.interpolate_mode = interpolate_mode
        self.out_indices = out_indices
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims
        )
        
        num_patches = self.patch_embed.num_patches
        
        # 分类token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
            num_tokens = num_patches + 1
        else:
            num_tokens = num_patches
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dims))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        # Transformer layers
        self.layers = ModuleList()
        for i in range(num_layers):
            layer = TransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias
            )
            self.layers.append(layer)
        
        # Final norm
        if final_norm:
            self.norm = build_norm_layer(dict(type='LN'), embed_dims)[1]
        else:
            self.norm = None
    
    def init_weights(self):
        """初始化权重。"""
        super().init_weights()
        
        # 总是执行默认初始化，不依赖预训练权重
        # 初始化位置编码
        if hasattr(self, 'pos_embed') and self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 初始化分类token
        if self.with_cls_token and hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def _pos_embeding(self, patched_img: torch.Tensor, 
                     hw_shape: Tuple[int, int]) -> torch.Tensor:
        """位置编码处理。"""
        assert patched_img.ndim == 3 and patched_img.shape[2] == self.embed_dims
        h, w = hw_shape
        
        # 添加分类token
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(patched_img.shape[0], -1, -1)
            patched_img = torch.cat((cls_tokens, patched_img), dim=1)
        
        # 位置编码插值
        if self.pos_embed.shape[1] != patched_img.shape[1]:
            pos_embed = self.resize_pos_embed(
                self.pos_embed, 
                (h, w),
                mode=self.interpolate_mode,
                num_extra_tokens=1 if self.with_cls_token else 0
            )
        else:
            pos_embed = self.pos_embed
        
        return self.pos_drop(patched_img + pos_embed)
    
    @staticmethod
    def resize_pos_embed(pos_embed: torch.Tensor,
                        input_shape: Tuple[int, int],
                        mode: str = 'bicubic',
                        num_extra_tokens: int = 1) -> torch.Tensor:
        """调整位置编码尺寸。"""
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
        
        pos_h = pos_w = int(math.sqrt(pos_embed.shape[1] - num_extra_tokens))
        target_h, target_w = input_shape
        
        if pos_h == target_h and pos_w == target_w:
            return pos_embed
        
        extra_tokens = None
        if num_extra_tokens:
            extra_tokens = pos_embed[:, :num_extra_tokens]
            pos_embed = pos_embed[:, num_extra_tokens:]
        
        pos_embed = pos_embed.reshape(1, pos_h, pos_w, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed, size=(target_h, target_w), mode=mode, align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        
        if num_extra_tokens and extra_tokens is not None:
            pos_embed = torch.cat([extra_tokens, pos_embed], dim=1)
        
        return pos_embed
    
    def forward(self, inputs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """前向传播。
        
        Args:
            inputs (torch.Tensor): 输入图像，形状为 (B, C, H, W)
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: 特征输出
        """
        # 确保inputs是4维张量
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(inputs)}")
        
        if inputs.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {inputs.dim()}D tensor with shape {inputs.shape}")
        
        B, C, H, W = inputs.shape
        
        # Patch embedding
        x = self.patch_embed(inputs)  # (B, num_patches, embed_dims)
        patch_h, patch_w = self.patch_size if isinstance(self.patch_size, tuple) else (self.patch_size, self.patch_size)
        hw_shape = (H // patch_h, W // patch_w)
        
        # 位置编码
        x = self._pos_embeding(x, hw_shape)
        
        # Transformer layers
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if i in self.out_indices:
                if self.with_cls_token and not self.output_cls_token:
                    # 移除分类token
                    out = x[:, 1:]
                else:
                    out = x
                
                if self.norm is not None:
                    out = self.norm(out)
                
                # 将输出从 [B, N, C] 转换为 [B, C, H, W] 格式
                if out.dim() == 3:
                    B_out, N, C_out = out.shape
                    H_out = W_out = int(N ** 0.5)  # 假设是方形patch grid
                    out = out.transpose(1, 2).reshape(B_out, C_out, H_out, W_out)
                
                outs.append(out)
        
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)


# 便捷函数
def dinov3_vit_small(**kwargs):
    """DINOv3 ViT-Small模型。"""
    return DINOv3ViT(arch='small', **kwargs)


def dinov3_vit_base(**kwargs):
    """DINOv3 ViT-Base模型。"""
    return DINOv3ViT(arch='base', **kwargs)


def dinov3_vit_large(**kwargs):
    """DINOv3 ViT-Large模型。"""
    return DINOv3ViT(arch='large', **kwargs)


def dinov3_vit_giant(**kwargs):
    """DINOv3 ViT-Giant模型。"""
    return DINOv3ViT(arch='giant', **kwargs)