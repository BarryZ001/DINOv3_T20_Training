#!/usr/bin/env python3
"""
自定义的SegLocalVisualizer类，用于解决T20服务器上的注册表问题
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
import torch
from mmengine.registry import VISUALIZERS
from mmengine.visualization import Visualizer
from mmengine.structures import PixelData


@VISUALIZERS.register_module()
class SegLocalVisualizer(Visualizer):
    """自定义的分割可视化器，兼容T20服务器环境"""
    
    def __init__(self, 
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Any] = None,
                 save_dir: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            **kwargs
        )
        self.alpha = alpha
        # LoveDA数据集的调色板
        self.palette = [
            [255, 255, 255],  # 0: background - 白色
            [255, 0, 0],      # 1: building - 红色
            [255, 255, 0],    # 2: road - 黄色
            [0, 0, 255],      # 3: water - 蓝色
            [159, 129, 183],  # 4: barren - 紫色
            [0, 255, 0],      # 5: forest - 绿色
            [255, 195, 128],  # 6: agricultural - 橙色
        ]
    
    def add_datasample(self,
                      name: str,
                      image: np.ndarray,
                      data_sample: Optional[Any] = None,
                      draw_gt: bool = True,
                      draw_pred: bool = True,
                      show: bool = False,
                      wait_time: float = 0,
                      out_file: Optional[str] = None,
                      step: int = 0) -> None:
        """添加数据样本进行可视化"""
        
        # 确保图像格式正确
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # 设置当前图像
        self.set_image(image)
        
        # 如果有数据样本，绘制分割结果
        if data_sample is not None:
            # 处理真实标签
            if hasattr(data_sample, 'gt_sem_seg') and getattr(data_sample, 'gt_sem_seg', None) is not None and draw_gt:
                gt_mask = self._extract_mask(getattr(data_sample, 'gt_sem_seg'))
                if gt_mask is not None:
                    self._draw_sem_seg(gt_mask, colors=self.palette, alpha=self.alpha)
            
            # 处理预测结果
            if hasattr(data_sample, 'pred_sem_seg') and getattr(data_sample, 'pred_sem_seg', None) is not None and draw_pred:
                pred_mask = self._extract_mask(getattr(data_sample, 'pred_sem_seg'))
                if pred_mask is not None:
                    self._draw_sem_seg(pred_mask, colors=self.palette, alpha=self.alpha)
        
        # 显示或保存结果
        if show:
            self.show(win_name=name, wait_time=wait_time)
        
        if out_file is not None:
            # 使用OpenCV保存图像
            drawn_img = self.get_image()
            try:
                import cv2
                cv2.imwrite(out_file, drawn_img)
            except ImportError:
                # 如果没有cv2，使用PIL
                from PIL import Image
                if drawn_img.dtype != np.uint8:
                    drawn_img = drawn_img.astype(np.uint8)
                Image.fromarray(drawn_img).save(out_file)
    
    def _extract_mask(self, sem_seg_data: Any) -> Optional[np.ndarray]:
        """从不同格式的分割数据中提取mask"""
        if sem_seg_data is None:
            return None
        
        # 如果是PixelData对象
        if hasattr(sem_seg_data, 'data'):
            mask = sem_seg_data.data
        else:
            mask = sem_seg_data
        
        # 转换为numpy数组
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # 确保是2D数组
        if mask.ndim == 3:
            mask = mask.squeeze()
        elif mask.ndim == 4:
            mask = mask.squeeze(0).squeeze(0)
        
        return mask
    
    def _draw_sem_seg(self, 
                     sem_seg: np.ndarray,
                     colors: List[List[int]],
                     alpha: float = 0.8) -> None:
        """绘制语义分割结果"""
        
        # 创建彩色分割图
        colored_seg = np.zeros((*sem_seg.shape, 3), dtype=np.uint8)
        
        for label, color in enumerate(colors):
            if label < len(colors):
                colored_seg[sem_seg == label] = color
        
        # 叠加到当前图像上
        self.draw_binary_masks(
            colored_seg,
            colors=['red'] * len(colors),  # 这里的颜色会被colored_seg覆盖
            alphas=[alpha] * len(colors)
        )
    
    def draw_binary_masks(self,
                         binary_masks: Union[np.ndarray, torch.Tensor],
                         colors: Union[str, tuple, List[str], List[tuple]] = 'red',
                         alphas: Union[int, float, List[Union[int, float]]] = 0.8) -> None:
        """绘制二值掩码"""
        
        if isinstance(binary_masks, torch.Tensor):
            binary_masks = binary_masks.cpu().numpy()
        
        # 简单的掩码叠加实现
        if binary_masks.ndim == 3 and binary_masks.shape[2] == 3:
            # 如果是彩色掩码，直接叠加
            current_image = self.get_image()
            if isinstance(alphas, (int, float)):
                alpha = alphas
            else:
                alpha = alphas[0] if alphas else 0.8
            
            # 创建掩码（非黑色像素）
            mask = np.any(binary_masks > 0, axis=2)
            
            # 叠加图像
            blended = current_image.copy()
            blended[mask] = (alpha * binary_masks[mask] + 
                           (1 - alpha) * current_image[mask]).astype(np.uint8)
            
            self.set_image(blended)