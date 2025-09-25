"""LoveDA Dataset implementation for remote sensing image segmentation."""

import os
import os.path as osp
from typing import List, Dict, Any, Optional

from mmengine.dataset import BaseDataset
# 导入正确的注册表
try:
    from mmseg.registry import DATASETS
    print("✅ 使用MMSeg DATASETS注册表")
except ImportError:
    from mmengine.registry import DATASETS
    print("⚠️ 回退到MMEngine DATASETS注册表")


@DATASETS.register_module()
class LoveDADataset(BaseDataset):
    """LoveDA dataset for remote sensing image segmentation.
    
    The LoveDA dataset contains 7 classes:
    - background (0)
    - building (1) 
    - road (2)
    - water (3)
    - barren (4)
    - forest (5)
    - agriculture (6)
    """
    
    METAINFO = {
        'classes': ('background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture'),
        'palette': [
            [255, 255, 255],  # background - white
            [255, 0, 0],      # building - red
            [255, 255, 0],    # road - yellow
            [0, 0, 255],      # water - blue
            [159, 129, 183],  # barren - purple
            [0, 255, 0],      # forest - green
            [255, 195, 128]   # agriculture - orange
        ]
    }
    
    def __init__(self,
                 data_root: str,
                 data_prefix: Optional[Dict[str, str]] = None,
                 img_suffix: str = '.png',
                 seg_map_suffix: str = '.png',
                 **kwargs):
        """Initialize LoveDA dataset.
        
        Args:
            data_root (str): Root directory of the dataset.
            data_prefix (dict, optional): Prefix for data paths.
            img_suffix (str): Suffix of images. Default: '.png'
            seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        """
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        
        if data_prefix is None:
            data_prefix = {}
            
        super().__init__(data_root=data_root, data_prefix=data_prefix, **kwargs)
    
    def load_data_list(self) -> List[Dict[str, Any]]:
        """Load annotation file to get data list.
        
        Returns:
            list[dict]: A list of annotation.
        """
        data_list = []
        
        # Get image and segmentation map directories with proper type handling
        img_path_prefix = self.data_prefix.get('img_path') or ''
        seg_path_prefix = self.data_prefix.get('seg_map_path') or ''
        
        # Ensure string types for path joining
        img_path_str = str(img_path_prefix) if img_path_prefix is not None else ''
        seg_path_str = str(seg_path_prefix) if seg_path_prefix is not None else ''
        
        img_dir = osp.join(str(self.data_root), img_path_str)
        seg_dir = osp.join(str(self.data_root), seg_path_str)
        
        # Debug information
        print(f"🔍 Loading LoveDA dataset from: {self.data_root}")
        print(f"📁 Image directory: {img_dir}")
        print(f"📁 Segmentation directory: {seg_dir}")
        
        # Try different LoveDA dataset structures
        possible_structures = [
            # Standard LoveDA structure
            ['Train/Rural', 'Train/Urban', 'Val/Rural', 'Val/Urban'],
            # Alternative structures
            ['train/Rural', 'train/Urban', 'val/Rural', 'val/Urban'],
            ['Rural', 'Urban'],
            ['train', 'val'],
            # Direct structure
            ['']
        ]
        
        found_data = False
        
        for structure in possible_structures:
            if found_data:
                break
                
            for subdir in structure:
                current_img_dir = osp.join(img_dir, subdir) if subdir else img_dir
                current_seg_dir = osp.join(seg_dir, subdir) if subdir else seg_dir
                
                # Check for images_png and masks_png subdirectories (LoveDA structure)
                for sub_subdir in ['images_png', '']:
                    img_path = osp.join(current_img_dir, sub_subdir) if sub_subdir else current_img_dir
                    seg_path = osp.join(current_seg_dir, 'masks_png' if sub_subdir else '')
                    
                    if not seg_path.endswith('masks_png') and sub_subdir == '':
                        seg_path = current_seg_dir
                    
                    if osp.exists(img_path) and osp.isdir(img_path):
                        img_files = [f for f in os.listdir(img_path) if f.endswith(self.img_suffix)]
                        
                        if img_files:
                            print(f"✅ Found {len(img_files)} images in {img_path}")
                            found_data = True
                            
                            for img_file in img_files:
                                img_full_path = osp.join(img_path, img_file)
                                seg_file = img_file.replace(self.img_suffix, self.seg_map_suffix)
                                seg_full_path = osp.join(seg_path, seg_file)
                                
                                # Create data info
                                data_info = {
                                    'img_path': img_full_path,
                                    'seg_map_path': seg_full_path,
                                    'label_map': None,
                                    'reduce_zero_label': False,
                                    'seg_fields': []
                                }
                                data_list.append(data_info)
        
        # If no real data found, raise an error instead of creating dummy data
        if not data_list:
            raise FileNotFoundError(
                f"No LoveDA data found in {self.data_root}. "
                f"Please ensure the LoveDA dataset is properly installed and the data_root path is correct. "
                f"Expected structure: {self.data_root}/[Train|Val|Test]/[Urban|Rural]/images_png/ and masks_png/"
            )
        else:
            print(f"✅ Successfully loaded {len(data_list)} LoveDA samples")
        
        return data_list