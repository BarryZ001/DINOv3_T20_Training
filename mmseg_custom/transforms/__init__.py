# 导入多模态transforms
try:
    from .multimodal_transforms import (
        MultiModalNormalize,
        MultiModalResize,
        SARSpecificAugmentation,
        InfraredSpecificAugmentation,
        MultiModalRandomCrop,
        RandomCrop,
        PhotoMetricDistortion,
        build_multimodal_pipeline,
        PackSegInputs
    )
    multimodal_available = True
except ImportError as e:
    print(f"Warning: Failed to import multimodal_transforms: {e}")
    multimodal_available = False

# 导入标准transforms兼容性模块
try:
    from .standard_transforms import (
        LoadImageFromFile, LoadAnnotations, Resize, RandomFlip,
        Normalize, Pad, ImageToTensor, DefaultFormatBundle, Collect
    )
    standard_available = True
except ImportError as e:
    print(f"Warning: Failed to import standard_transforms: {e}")
    print("This may be due to missing mmcv dependency. Please install mmcv-full.")
    standard_available = False

# 动态构建__all__列表
__all__ = []

if multimodal_available:
    __all__.extend([
        'MultiModalNormalize',
        'MultiModalResize', 
        'SARSpecificAugmentation',
        'InfraredSpecificAugmentation',
        'MultiModalRandomCrop',
        'RandomCrop',
        'PhotoMetricDistortion',
        'build_multimodal_pipeline',
        'PackSegInputs'
    ])

if standard_available:
    __all__.extend([
        'LoadImageFromFile', 'LoadAnnotations', 'Resize', 'RandomFlip',
        'Normalize', 'Pad', 'ImageToTensor', 'DefaultFormatBundle', 'Collect'
    ])