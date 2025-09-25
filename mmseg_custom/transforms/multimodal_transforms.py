"""多模态遥感图像预处理管道

支持光学、SAR、红外等不同模态的图像预处理和数据增强。
针对MMRS-1M数据集的特点进行优化。
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import mmcv
from mmengine.registry import TRANSFORMS
from mmengine.structures import PixelData

# 基础变换类
class BaseTransform:
    """基础变换类"""
    def __call__(self, results):
        return self.transform(results)
    
    def transform(self, results):
        raise NotImplementedError


@TRANSFORMS.register_module()
class MultiModalNormalize(BaseTransform):
    """多模态图像归一化。
    
    针对不同模态（光学、SAR、红外）使用不同的归一化参数。
    """
    
    def __init__(self,
                 modality: str = 'optical',
                 mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None,
                 to_rgb: bool = True):
        """初始化多模态归一化。
        
        Args:
            modality (str): 图像模态类型
            mean (List[float], optional): 均值，如果为None则使用默认值
            std (List[float], optional): 标准差，如果为None则使用默认值
            to_rgb (bool): 是否转换为RGB格式
        """
        self.modality = modality
        self.to_rgb = to_rgb
        
        # 不同模态的默认归一化参数
        self.modality_params = {
            'optical': {
                'mean': [123.675, 116.28, 103.53],  # ImageNet统计值
                'std': [58.395, 57.12, 57.375]
            },
            'sar': {
                'mean': [127.5],  # SAR图像通常是单通道
                'std': [127.5]
            },
            'infrared': {
                'mean': [127.5, 127.5, 127.5],  # 红外图像
                'std': [127.5, 127.5, 127.5]
            }
        }
        
        # 使用提供的参数或默认参数
        if mean is not None:
            self.mean = np.array(mean, dtype=np.float32)
        else:
            self.mean = np.array(self.modality_params[modality]['mean'], dtype=np.float32)
            
        if std is not None:
            self.std = np.array(std, dtype=np.float32)
        else:
            self.std = np.array(self.modality_params[modality]['std'], dtype=np.float32)
    
    def transform(self, results: Dict) -> Dict:
        """执行归一化变换。"""
        img = results['img']
        
        # BGR转RGB
        if self.to_rgb and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化
        img = img.astype(np.float32)
        
        # 处理不同通道数的情况
        if len(self.mean) == 1 and img.shape[2] == 3:
            # SAR单通道参数应用到三通道图像
            mean = np.array([self.mean[0]] * 3)
            std = np.array([self.std[0]] * 3)
        elif len(self.mean) == 3 and img.shape[2] == 1:
            # 三通道参数应用到单通道图像
            mean = np.array([np.mean(self.mean)])
            std = np.array([np.mean(self.std)])
        else:
            mean = self.mean
            std = self.std
        
        img = (img - mean) / std
        
        results['img'] = img
        results['img_norm_cfg'] = dict(
            mean=mean.tolist(),
            std=std.tolist(),
            to_rgb=self.to_rgb
        )
        
        return results


@TRANSFORMS.register_module()
class MultiModalResize(BaseTransform):
    """多模态图像缩放。
    
    针对不同模态使用不同的插值方法。
    """
    
    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 modality: str = 'optical',
                 keep_ratio: bool = True):
        """初始化多模态缩放。
        
        Args:
            scale: 目标尺寸
            modality: 图像模态
            keep_ratio: 是否保持宽高比
        """
        self.scale = scale if isinstance(scale, tuple) else (scale, scale)
        self.modality = modality
        self.keep_ratio = keep_ratio
        
        # 不同模态的插值方法
        self.interpolation_methods = {
            'optical': cv2.INTER_LINEAR,
            'sar': cv2.INTER_NEAREST,  # SAR图像使用最近邻插值保持纹理
            'infrared': cv2.INTER_LINEAR
        }
    
    def transform(self, results: Dict) -> Dict:
        """执行缩放变换。"""
        img = results['img']
        h, w = img.shape[:2]
        
        # 计算新尺寸
        if self.keep_ratio:
            scale_factor = min(self.scale[0] / w, self.scale[1] / h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
        else:
            new_w, new_h = self.scale
        
        # 选择插值方法
        interpolation = self.interpolation_methods.get(self.modality, cv2.INTER_LINEAR)
        
        # 缩放图像
        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        
        results['img'] = img
        results['img_shape'] = img.shape
        results['scale_factor'] = (new_w / w, new_h / h)
        
        # 如果有分割标注，也需要缩放
        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            gt_seg_map = cv2.resize(gt_seg_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            results['gt_seg_map'] = gt_seg_map
        
        return results


@TRANSFORMS.register_module()
class SARSpecificAugmentation(BaseTransform):
    """SAR图像特定的数据增强。
    
    包括斑点噪声模拟、对比度增强等。
    """
    
    def __init__(self,
                 speckle_prob: float = 0.3,
                 speckle_strength: float = 0.1,
                 contrast_prob: float = 0.5,
                 contrast_range: Tuple[float, float] = (0.8, 1.2)):
        """初始化SAR增强。
        
        Args:
            speckle_prob: 斑点噪声概率
            speckle_strength: 斑点噪声强度
            contrast_prob: 对比度调整概率
            contrast_range: 对比度调整范围
        """
        self.speckle_prob = speckle_prob
        self.speckle_strength = speckle_strength
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range
    
    def transform(self, results: Dict) -> Dict:
        """执行SAR特定增强。"""
        img = results['img']
        
        # 斑点噪声
        if np.random.random() < self.speckle_prob:
            noise = np.random.gamma(1.0, self.speckle_strength, img.shape)
            img = img * noise
        
        # 对比度调整
        if np.random.random() < self.contrast_prob:
            contrast_factor = np.random.uniform(*self.contrast_range)
            img = img * contrast_factor
        
        # 确保像素值在合理范围内
        img = np.clip(img, 0, 255)
        
        results['img'] = img.astype(np.uint8)
        
        return results


@TRANSFORMS.register_module()
class InfraredSpecificAugmentation(BaseTransform):
    """红外图像特定的数据增强。
    
    包括热噪声模拟、温度范围调整等。
    """
    
    def __init__(self,
                 thermal_noise_prob: float = 0.3,
                 thermal_noise_std: float = 5.0,
                 temperature_shift_prob: float = 0.4,
                 temperature_shift_range: Tuple[float, float] = (-10, 10)):
        """初始化红外增强。
        
        Args:
            thermal_noise_prob: 热噪声概率
            thermal_noise_std: 热噪声标准差
            temperature_shift_prob: 温度偏移概率
            temperature_shift_range: 温度偏移范围
        """
        self.thermal_noise_prob = thermal_noise_prob
        self.thermal_noise_std = thermal_noise_std
        self.temperature_shift_prob = temperature_shift_prob
        self.temperature_shift_range = temperature_shift_range
    
    def transform(self, results: Dict) -> Dict:
        """执行红外特定增强。"""
        img = results['img'].astype(np.float32)
        
        # 热噪声
        if np.random.random() < self.thermal_noise_prob:
            noise = np.random.normal(0, self.thermal_noise_std, img.shape)
            img = img + noise
        
        # 温度偏移
        if np.random.random() < self.temperature_shift_prob:
            shift = np.random.uniform(*self.temperature_shift_range)
            img = img + shift
        
        # 确保像素值在合理范围内
        img = np.clip(img, 0, 255)
        
        results['img'] = img.astype(np.uint8)
        
        return results


@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):
    """光度失真变换。
    
    对图像进行亮度、对比度、饱和度和色调的随机调整。
    """
    
    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Tuple[float, float] = (0.5, 1.5),
                 saturation_range: Tuple[float, float] = (0.5, 1.5),
                 hue_delta: int = 18):
        """初始化光度失真变换。
        
        Args:
            brightness_delta: 亮度调整范围
            contrast_range: 对比度调整范围
            saturation_range: 饱和度调整范围
            hue_delta: 色调调整范围
        """
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta

    def transform(self, results: Dict) -> Dict:
        """执行光度失真变换。"""
        img = results['img'].astype(np.float32)
        
        # 随机亮度调整
        if np.random.randint(2):
            delta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
            img += delta
            
        # 随机对比度调整
        if np.random.randint(2):
            alpha = np.random.uniform(*self.contrast_range)
            img *= alpha
            
        # 随机饱和度调整（仅对RGB图像）
        if len(img.shape) == 3 and img.shape[2] == 3:
            if np.random.randint(2):
                # 转换到HSV空间进行饱和度调整
                img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                alpha = np.random.uniform(*self.saturation_range)
                img_hsv[:, :, 1] *= alpha
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
                img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            
            # 随机色调调整
            if np.random.randint(2):
                img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                delta = np.random.uniform(-self.hue_delta, self.hue_delta)
                img_hsv[:, :, 0] += delta
                img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 179)  # OpenCV中H通道范围是0-179
                img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # 确保像素值在有效范围内
        img = np.clip(img, 0, 255).astype(np.uint8)
        results['img'] = img
        
        return results


@TRANSFORMS.register_module()
class PackSegInputs:
    """Pack the inputs data for the semantic segmentation.
    
    This transform packs the image and ground truth segmentation map into
    a format that can be consumed by the model. It returns the data in the
    standard MMEngine format expected by pseudo_collate.
    """
    
    def __init__(self, meta_keys=('img_path', 'seg_map_path', 'ori_shape', 
                                  'img_shape', 'pad_shape', 'scale_factor', 
                                  'flip', 'flip_direction')):
        """Initialize PackSegInputs.
        
        Args:
            meta_keys (tuple): Keys to be packed into meta information.
        """
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Pack the inputs data.
        
        Args:
            results (dict): Result dict from loading pipeline.
            
        Returns:
            dict: Results with 'inputs' key in standard MMEngine format.
        """
        import torch
        
        # 🔥 关键修复：创建标准的 'inputs' 键
        if 'img' in results:
            img = results['img']
            
            # 确保图像是torch.Tensor格式
            if isinstance(img, np.ndarray):
                # 转换numpy数组为torch tensor
                img = torch.from_numpy(img.copy()).float()
                
            # 确保图像是CHW格式
            if len(img.shape) == 3:
                # 如果是HWC格式，转换为CHW
                if img.shape[2] == 3:  # HWC
                    img = img.permute(2, 0, 1)  # 转换为CHW
                    
            # 创建标准的 'inputs' 键供模型使用
            results['inputs'] = img
            # 保留原始的 'img' 键以兼容其他组件
            results['img'] = img
            
        # 确保分割图格式正确
        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            # 确保分割图是2D的
            if len(gt_seg_map.shape) == 3:
                gt_seg_map = gt_seg_map.squeeze()
            results['gt_semantic_seg'] = gt_seg_map
            
        # 添加必要的字段
        if 'seg_fields' not in results:
            results['seg_fields'] = []
        if 'gt_semantic_seg' in results and 'gt_semantic_seg' not in results['seg_fields']:
            results['seg_fields'].append('gt_semantic_seg')
            
        # 确保img_fields存在
        if 'img_fields' not in results:
            results['img_fields'] = ['img']
            
        return results


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """标准随机裁剪变换。
    
    兼容mmseg的RandomCrop接口。
    """
    
    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.0,
                 ignore_index: int = 255):
        """初始化随机裁剪。
        
        Args:
            crop_size: 裁剪尺寸
            cat_max_ratio: 类别最大比例
            ignore_index: 忽略索引
        """
        self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def transform(self, results: Dict) -> Dict:
        """执行随机裁剪。"""
        img = results['img']
        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size
        
        # 如果图像小于裁剪尺寸，先进行填充
        if h < crop_h or w < crop_w:
            pad_h = max(0, crop_h - h)
            pad_w = max(0, crop_w - w)
            
            # 使用ImageNet均值填充
            pad_value = [123.675, 116.28, 103.53]
            
            img = cv2.copyMakeBorder(
                img, 0, pad_h, 0, pad_w, 
                cv2.BORDER_CONSTANT, 
                value=pad_value[:img.shape[2]]
            )
            
            if 'gt_seg_map' in results:
                gt_seg_map = results['gt_seg_map']
                gt_seg_map = cv2.copyMakeBorder(
                    gt_seg_map, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT,
                    value=self.ignore_index
                )
                results['gt_seg_map'] = gt_seg_map
            
            h, w = img.shape[:2]
        
        # 随机选择裁剪位置
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        
        # 执行裁剪
        results['img'] = img[top:top + crop_h, left:left + crop_w]
        
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][top:top + crop_h, left:left + crop_w]
        
        # 更新图像形状信息
        results['img_shape'] = (crop_h, crop_w)
        
        return results


@TRANSFORMS.register_module()
class MultiModalRandomCrop(BaseTransform):
    """多模态随机裁剪。
    
    针对不同模态的特点进行优化的随机裁剪。
    """
    
    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 modality: str = 'optical',
                 cat_max_ratio: float = 1.0,
                 ignore_index: int = 255):
        """初始化多模态随机裁剪。
        
        Args:
            crop_size: 裁剪尺寸
            modality: 图像模态
            cat_max_ratio: 类别最大比例
            ignore_index: 忽略索引
        """
        self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        self.modality = modality
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
    
    def transform(self, results: Dict) -> Dict:
        """执行随机裁剪。"""
        img = results['img']
        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size
        
        # 如果图像小于裁剪尺寸，先进行填充
        if h < crop_h or w < crop_w:
            pad_h = max(0, crop_h - h)
            pad_w = max(0, crop_w - w)
            
            # 不同模态使用不同的填充值
            if self.modality == 'optical':
                pad_value = [123.675, 116.28, 103.53]  # ImageNet均值
            elif self.modality == 'sar':
                pad_value = [0]  # SAR图像用0填充
            else:  # infrared
                pad_value = [127.5, 127.5, 127.5]  # 中性灰度值
            
            img = cv2.copyMakeBorder(
                img, 0, pad_h, 0, pad_w, 
                cv2.BORDER_CONSTANT, 
                value=pad_value[:img.shape[2]]
            )
            
            if 'gt_seg_map' in results:
                gt_seg_map = results['gt_seg_map']
                gt_seg_map = cv2.copyMakeBorder(
                    gt_seg_map, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT,
                    value=self.ignore_index
                )
                results['gt_seg_map'] = gt_seg_map
            
            h, w = img.shape[:2]
        
        # 随机选择裁剪位置
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        
        # 执行裁剪
        img = img[top:top + crop_h, left:left + crop_w]
        results['img'] = img
        results['img_shape'] = img.shape
        
        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            gt_seg_map = gt_seg_map[top:top + crop_h, left:left + crop_w]
            results['gt_seg_map'] = gt_seg_map
        
        return results


def build_multimodal_pipeline(modality: str = 'optical',
                             crop_size: Tuple[int, int] = (512, 512),
                             training: bool = True) -> List[Dict]:
    """构建多模态预处理管道。
    
    Args:
        modality: 图像模态类型
        crop_size: 裁剪尺寸
        training: 是否为训练模式
    
    Returns:
        预处理管道配置列表
    """
    pipeline = []
    
    # 基础变换
    pipeline.extend([
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations') if training else dict(type='LoadAnnotations', reduce_zero_label=True),
    ])
    
    # 多模态缩放
    pipeline.append(
        dict(
            type='MultiModalResize',
            scale=crop_size,
            modality=modality,
            keep_ratio=True
        )
    )
    
    if training:
        # 训练时的数据增强
        pipeline.extend([
            dict(
                type='MultiModalRandomCrop',
                crop_size=crop_size,
                modality=modality
            ),
            dict(type='RandomFlip', prob=0.5),
        ])
        
        # 模态特定增强
        if modality == 'sar':
            pipeline.append(
                dict(
                    type='SARSpecificAugmentation',
                    speckle_prob=0.3,
                    contrast_prob=0.5
                )
            )
        elif modality == 'infrared':
            pipeline.append(
                dict(
                    type='InfraredSpecificAugmentation',
                    thermal_noise_prob=0.3,
                    temperature_shift_prob=0.4
                )
            )
    
    # 归一化
    pipeline.append(
        dict(
            type='MultiModalNormalize',
            modality=modality,
            to_rgb=True
        )
    )
    
    # 打包输入
    pipeline.append(
        dict(
            type='PackSegInputs',
            meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 
                      'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                      'modality', 'task_type')
        )
    )
    
    return pipeline