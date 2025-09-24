import json
import os
import os.path as osp
from typing import Dict, List, Optional, Union

try:
    import mmcv
except ImportError:
    # 如果mmcv不可用，使用基础的cv2或PIL替代
    mmcv = None
    print("Warning: mmcv not available, using fallback image loading")

import numpy as np
from mmengine.fileio import get_local_path
from PIL import Image

from mmengine.dataset import BaseDataset

# 导入正确的注册表
try:
    from mmseg.registry import DATASETS
    print("✅ 使用MMSeg DATASETS注册表")
except ImportError:
    from mmengine.registry import DATASETS
    print("⚠️ 回退到MMEngine DATASETS注册表")


@DATASETS.register_module()
class MMRS1MDataset(BaseDataset):
    """MMRS-1M多模态遥感数据集，支持分类、检测、VQA、视觉定位等任务。
    
    数据集结构:
    mmrs1m/
    └── data/
        ├── caption/          # 图像描述任务
        ├── classification/   # 分类任务  
        ├── detection/        # 目标检测任务
        ├── json/            # 标注文件
        ├── RSVG/            # 视觉定位任务
        └── VQA/             # 视觉问答任务
    """
    
    METAINFO = dict(
        classes=('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural'),
        palette=[[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255], 
                [159, 129, 183], [0, 255, 0], [255, 195, 128]]
    )
    
    def __init__(self,
                 task_type: str = 'classification',
                 modality: str = 'optical',
                 instruction_format: bool = True,
                 **kwargs):
        """初始化MMRS-1M数据集。
        
        Args:
            task_type (str): 任务类型，支持 'classification', 'detection', 'caption', 'vqa', 'rsvg'
            modality (str): 图像模态，支持 'optical', 'sar', 'infrared'
            instruction_format (bool): 是否使用指令跟随格式
        """
        self.task_type = task_type
        self.modality = modality
        self.instruction_format = instruction_format
        
        # 过滤掉'type'参数，避免传递给父类BaseDataset
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'type'}
        
        super().__init__(**filtered_kwargs)
        
    def load_data_list(self) -> List[dict]:
        """加载数据列表。"""
        data_list = []
        
        # 根据任务类型加载不同的数据
        if self.task_type == 'classification':
            data_list = self._load_classification_data()
        elif self.task_type == 'detection':
            data_list = self._load_detection_data()
        elif self.task_type == 'segmentation':
            data_list = self._load_segmentation_data()
        elif self.task_type == 'caption':
            data_list = self._load_caption_data()
        elif self.task_type == 'vqa':
            data_list = self._load_vqa_data()
        elif self.task_type == 'rsvg':
            data_list = self._load_rsvg_data()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
            
        print(f"[MMRS1M] 加载了 {len(data_list)} 个数据样本，任务类型: {self.task_type}")
        
        # 如果没有找到任何真实数据，创建一个占位符数据项以避免空数据集错误
        if not data_list:
            print(f"[MMRS1M] 警告：未找到数据，使用占位符数据")
            # 使用项目根目录下的test_image.jpg作为占位符
            project_root = osp.dirname(osp.dirname(osp.dirname(__file__)))
            placeholder_img = osp.join(project_root, 'test_image.jpg')
            
            if osp.exists(placeholder_img):
                # 为占位符创建一个简单的分割图（如果需要的话）
                placeholder_seg = None
                if self.task_type in ['segmentation', 'detection']:
                    # 对于需要分割标注的任务，使用图像本身作为占位符分割图
                    placeholder_seg = placeholder_img
                
                placeholder_data = {
                    'img_path': placeholder_img,
                    'seg_map_path': placeholder_seg,
                    'label': 0,
                    'dataset': 'placeholder',
                    'modality': self.modality,
                    'task_type': self.task_type,
                    'seg_fields': []
                }
                
                # 根据任务类型添加特定字段
                if self.task_type == 'classification':
                    placeholder_data['category'] = 'unknown'
                elif self.task_type == 'detection':
                    placeholder_data['ann_file'] = None
                elif self.task_type == 'caption':
                    placeholder_data['caption'] = 'No real data available'
                elif self.task_type == 'vqa':
                    placeholder_data['question'] = 'What is in this image?'
                    placeholder_data['answer'] = 'No real data available'
                elif self.task_type == 'rsvg':
                    placeholder_data['expression'] = 'locate object'
                    placeholder_data['bbox'] = [0, 0, 100, 100]
                
                if self.instruction_format:
                    placeholder_data['instruction'] = 'This is placeholder data'
                    placeholder_data['response'] = 'No real data available'
                
                data_list = [placeholder_data]
            else:
                raise FileNotFoundError(f"No data found and placeholder image does not exist: {placeholder_img}")
        else:
            print(f"[MMRS1M] 成功加载数据，前3个样本路径:")
            for i, item in enumerate(data_list[:3]):
                print(f"  {i+1}. {item.get('img_path', 'N/A')}")
            
        return data_list
    
    def _load_classification_data(self) -> List[dict]:
        """加载分类任务数据。"""
        data_list = []
        
        # 确保data_root不为None
        if not self.data_root:
            print(f"[MMRS1M] data_root为空，无法加载数据")
            return data_list
            
        print(f"[MMRS1M] 尝试从路径加载分类数据: {self.data_root}")
        
        # 根据真实MMRS1M数据结构加载分类数据
        classification_dir = osp.join(self.data_root, 'classification')
        json_dir = osp.join(self.data_root, 'json', 'classification')
        
        print(f"[MMRS1M] 检查分类目录: {classification_dir}")
        print(f"[MMRS1M] 分类目录存在: {osp.exists(classification_dir)}")
        
        if not osp.exists(classification_dir):
            print(f"[MMRS1M] 分类目录不存在: {classification_dir}")
            return data_list
            
        # 真实的分类数据集列表
        classification_datasets = [
            'DCSR', 'EuroSAT_split', 'FGSCR_split', 'NWPU-RESISC45_split',
            'RSSCN7_split', 'UCMerced_split', 'WHU-RS19_split'
        ]
        
        print(f"[MMRS1M] 开始遍历分类数据集...")
        
        # 遍历每个分类数据集
        for dataset_name in classification_datasets:
            dataset_path = osp.join(classification_dir, dataset_name)
            print(f"[MMRS1M] 检查数据集: {dataset_name} -> {dataset_path}")
            
            if not osp.exists(dataset_path):
                print(f"[MMRS1M] 数据集不存在: {dataset_name}")
                continue
                
            print(f"[MMRS1M] 找到数据集: {dataset_name}")
            
            # 尝试加载对应的JSON标注文件
            json_file = None
            json_mapping = {
                'DCSR': 'DSCR_cls.json',
                'EuroSAT_split': 'EuroSAT_cls.json', 
                'FGSCR_split': 'FGSCR_cls.json',
                'NWPU-RESISC45_split': 'NWPU-RESISC45_cls.json',
                'RSSCN7_split': 'RSSCN7_cls.json',
                'UCMerced_split': 'UCMerced_cls.json',
                'WHU-RS19_split': 'WHU-RS19_cls.json'
            }
            
            if dataset_name in json_mapping:
                json_path = osp.join(json_dir, json_mapping[dataset_name])
                if osp.exists(json_path):
                    json_file = json_path
                    print(f"[MMRS1M] 找到JSON文件: {json_file}")
            
            # 检查数据集结构类型
            train_path = osp.join(dataset_path, 'train')
            test_path = osp.join(dataset_path, 'test') 
            images_path = osp.join(dataset_path, 'images')
            
            print(f"[MMRS1M] 检查数据集结构:")
            print(f"  train目录: {osp.exists(train_path)}")
            print(f"  test目录: {osp.exists(test_path)}")
            print(f"  images目录: {osp.exists(images_path)}")
            
            if osp.exists(train_path) or osp.exists(test_path):
                # 类型1: 有train/test分割的数据集
                print(f"[MMRS1M] 使用train/test结构加载数据集: {dataset_name}")
                for split in ['train', 'test']:
                    split_path = osp.join(dataset_path, split)
                    if osp.exists(split_path):
                        print(f"[MMRS1M] 处理{split}分割...")
                        # 遍历每个类别目录
                        try:
                            categories = os.listdir(split_path)
                            print(f"[MMRS1M] 找到{len(categories)}个类别: {categories[:5]}...")
                            
                            for category_dir in categories:
                                category_path = osp.join(split_path, category_dir)
                                if osp.isdir(category_path):
                                    # 遍历类别目录中的图像文件
                                    try:
                                        img_files = [f for f in os.listdir(category_path) 
                                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                                        print(f"[MMRS1M] 类别{category_dir}有{len(img_files)}张图像")
                                        
                                        for img_file in img_files:
                                            img_path = osp.join(category_path, img_file)
                                            
                                            data_info = {
                                                'img_path': img_path,
                                                'seg_map_path': None,
                                                'label': self._get_class_id_from_dataset(dataset_name, category_dir),
                                                'category': category_dir,
                                                'dataset': dataset_name,
                                                'split': split,
                                                'modality': self.modality,
                                                'task_type': self.task_type,
                                                'json_file': json_file,
                                                'seg_fields': []
                                            }
                                            
                                            if self.instruction_format:
                                                data_info['instruction'] = f"What is the category of this {dataset_name} remote sensing image? Answer using a single word or phrase."
                                                data_info['response'] = category_dir
                                                
                                            data_list.append(data_info)
                                    except Exception as e:
                                        print(f"[MMRS1M] 处理类别{category_dir}时出错: {e}")
                                        continue
                        except Exception as e:
                            print(f"[MMRS1M] 处理{split}分割时出错: {e}")
                            continue
                                        
            elif osp.exists(images_path):
                # 类型2: 只有images目录的数据集
                print(f"[MMRS1M] 使用images结构加载数据集: {dataset_name}")
                try:
                    img_files = [f for f in os.listdir(images_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                    print(f"[MMRS1M] images目录有{len(img_files)}张图像")
                    
                    for img_file in img_files:
                        img_path = osp.join(images_path, img_file)
                        
                        # 从JSON文件中获取标签信息（如果存在）
                        category = 'unknown'
                        if json_file and osp.exists(json_file):
                            try:
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    json_data = json.load(f)
                                    # 根据文件名查找对应的标签
                                    for item in json_data:
                                        if isinstance(item, dict) and 'image' in item and item['image'] == img_file:
                                            category = item.get('category', 'unknown')
                                            break
                            except:
                                pass
                        
                        data_info = {
                            'img_path': img_path,
                            'seg_map_path': None,
                            'label': self._get_class_id_from_dataset(dataset_name, category),
                            'category': category,
                            'dataset': dataset_name,
                            'split': 'all',
                            'modality': self.modality,
                            'task_type': self.task_type,
                            'json_file': json_file,
                            'seg_fields': []
                        }
                        
                        if self.instruction_format:
                            data_info['instruction'] = f"What is the category of this {dataset_name} remote sensing image? Answer using a single word or phrase."
                            data_info['response'] = category
                            
                        data_list.append(data_info)
                except Exception as e:
                    print(f"[MMRS1M] 处理images目录时出错: {e}")
            else:
                print(f"[MMRS1M] 数据集{dataset_name}结构不符合预期")
        
        print(f"[MMRS1M] 分类数据加载完成，共{len(data_list)}个样本")
        return data_list
    
    def _load_detection_data(self) -> List[dict]:
        """加载检测任务数据。"""
        data_list = []
        
        if not self.data_root:
            return data_list
            
        # 根据真实MMRS1M数据结构加载检测数据
        detection_dir = osp.join(self.data_root, 'detection')
        json_dir = osp.join(self.data_root, 'json', 'detection')
        
        if not osp.exists(detection_dir):
            return data_list
            
        # 真实的检测数据集列表
        detection_datasets = [
            'dior', 'dotav2', 'FAR1M', 'HIT-UAV', 'HRRSD', 'HRSID_HBB', 'HRSID_OBB',
            'infrared_ship_fusion', 'infrared_ship_lwir', 'infrared_ship_swir',
            'IR_peoplecar', 'IR_security', 'IR_ship', 'IR_streetscene',
            'NWPUVHR10', 'RSOD', 'SARV2', 'SSDD', 'UCAS', 'VisDrone'
        ]
        
        # 遍历每个检测数据集
        for dataset_name in detection_datasets:
            dataset_path = osp.join(detection_dir, dataset_name)
            if not osp.exists(dataset_path):
                continue
                
            # 查找对应的JSON标注目录
            json_dataset_dir = osp.join(json_dir, dataset_name)
            if not osp.exists(json_dataset_dir):
                # 尝试一些常见的映射
                json_mapping = {
                    'dior': 'dior_detection',
                    'dotav2': 'dota',
                    'HRSID_OBB': 'HRISD_OBB',
                    'infrared_ship_fusion': 'IR',
                    'infrared_ship_lwir': 'IR', 
                    'infrared_ship_swir': 'IR',
                    'IR_peoplecar': 'IR',
                    'IR_security': 'IR',
                    'IR_ship': 'IR',
                    'IR_streetscene': 'IR'
                }
                if dataset_name in json_mapping:
                    json_dataset_dir = osp.join(json_dir, json_mapping[dataset_name])
            
            # 查找图像目录
            images_path = osp.join(dataset_path, 'images')
            if not osp.exists(images_path):
                continue
                
            # 遍历图像文件
            for img_file in os.listdir(images_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    img_path = osp.join(images_path, img_file)
                    
                    # 查找对应的标注文件
                    annotation_file = None
                    if osp.exists(json_dataset_dir):
                        # 查找JSON标注文件
                        for json_file in os.listdir(json_dataset_dir):
                            if json_file.endswith('.json'):
                                json_path = osp.join(json_dataset_dir, json_file)
                                try:
                                    with open(json_path, 'r', encoding='utf-8') as f:
                                        json_data = json.load(f)
                                        # 检查是否包含当前图像的标注
                                        if isinstance(json_data, list):
                                            for item in json_data:
                                                if isinstance(item, dict) and 'image' in item and item['image'] == img_file:
                                                    annotation_file = json_path
                                                    break
                                        elif isinstance(json_data, dict) and img_file in json_data:
                                            annotation_file = json_path
                                            break
                                    if annotation_file:
                                        break
                                except:
                                    continue
                    
                    data_info = {
                        'img_path': img_path,
                        'seg_map_path': None,
                        'label': None,  # 检测任务使用bbox标注
                        'dataset': dataset_name,
                        'annotation_file': annotation_file,
                        'modality': self.modality,
                        'task_type': self.task_type,
                        'seg_fields': []
                    }
                    
                    if self.instruction_format:
                        data_info['instruction'] = f"Detect and locate all objects in this {dataset_name} remote sensing image. Provide bounding boxes and class labels."
                        data_info['response'] = "Objects detected with bounding boxes and labels."
                        
                    data_list.append(data_info)
        
        return data_list
    
    def _load_segmentation_data(self) -> List[dict]:
        """加载语义分割任务数据。"""
        data_list = []
        
        if not self.data_root:
            return data_list
            
        print(f"[MMRS1M] 尝试从路径加载分割数据: {self.data_root}")
        
        # 根据真实MMRS1M数据结构加载分割数据
        # 通常分割数据可能在classification目录中，或者有专门的segmentation目录
        segmentation_dir = osp.join(self.data_root, 'segmentation')
        classification_dir = osp.join(self.data_root, 'classification')
        
        # 优先检查是否有专门的分割目录
        if osp.exists(segmentation_dir):
            print(f"[MMRS1M] 找到专门的分割目录: {segmentation_dir}")
            data_list = self._load_segmentation_from_dir(segmentation_dir, 'segmentation')
        elif osp.exists(classification_dir):
            print(f"[MMRS1M] 使用分类目录进行分割任务: {classification_dir}")
            # 使用分类数据集，但将其适配为分割任务
            data_list = self._load_segmentation_from_classification(classification_dir)
        else:
            print(f"[MMRS1M] 未找到合适的分割数据目录")
        
        print(f"[MMRS1M] 分割数据加载完成，共{len(data_list)}个样本")
        return data_list
    
    def _load_segmentation_from_dir(self, seg_dir: str, dataset_name: str) -> List[dict]:
        """从专门的分割目录加载数据。"""
        data_list = []
        
        # 查找图像和标注目录
        images_dir = osp.join(seg_dir, 'images')
        labels_dir = osp.join(seg_dir, 'labels')
        masks_dir = osp.join(seg_dir, 'masks')
        annotations_dir = osp.join(seg_dir, 'annotations')
        
        # 确定标注目录
        seg_map_dir = None
        if osp.exists(labels_dir):
            seg_map_dir = labels_dir
        elif osp.exists(masks_dir):
            seg_map_dir = masks_dir
        elif osp.exists(annotations_dir):
            seg_map_dir = annotations_dir
        
        if not osp.exists(images_dir):
            print(f"[MMRS1M] 分割数据集{dataset_name}缺少images目录")
            return data_list
            
        # 遍历图像文件
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                img_path = osp.join(images_dir, img_file)
                
                # 查找对应的分割标注
                seg_map_path = None
                if seg_map_dir:
                    # 尝试不同的标注文件扩展名
                    base_name = osp.splitext(img_file)[0]
                    for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
                        seg_file = base_name + ext
                        seg_path = osp.join(seg_map_dir, seg_file)
                        if osp.exists(seg_path):
                            seg_map_path = seg_path
                            break
                
                data_info = {
                    'img_path': img_path,
                    'seg_map_path': seg_map_path,
                    'label': None,  # 分割任务使用seg_map_path
                    'dataset': dataset_name,
                    'modality': self.modality,
                    'task_type': self.task_type,
                    'seg_fields': ['gt_semantic_seg'] if seg_map_path else []
                }
                
                if self.instruction_format:
                    data_info['instruction'] = f"Perform semantic segmentation on this {dataset_name} remote sensing image. Identify and segment different land cover types."
                    data_info['response'] = "Semantic segmentation mask with different land cover classes."
                    
                data_list.append(data_info)
        
        return data_list
    
    def _load_segmentation_from_classification(self, classification_dir: str) -> List[dict]:
        """从分类数据集适配为分割任务。"""
        data_list = []
        
        # 使用分类数据集的结构，但适配为分割任务
        # 这里我们将分类标签转换为简单的分割标签
        classification_datasets = [
            'DCSR', 'EuroSAT_split', 'FGSCR_split', 'NWPU-RESISC45_split',
            'RSSCN7_split', 'UCMerced_split', 'WHU-RS19_split'
        ]
        
        for dataset_name in classification_datasets:
            dataset_path = osp.join(classification_dir, dataset_name)
            if not osp.exists(dataset_path):
                continue
                
            print(f"[MMRS1M] 适配分类数据集为分割任务: {dataset_name}")
            
            # 检查数据集结构
            train_path = osp.join(dataset_path, 'train')
            test_path = osp.join(dataset_path, 'test')
            images_path = osp.join(dataset_path, 'images')
            
            if osp.exists(train_path) or osp.exists(test_path):
                # 有train/test分割的数据集
                for split in ['train', 'test']:
                    split_path = osp.join(dataset_path, split)
                    if osp.exists(split_path):
                        # 遍历每个类别目录
                        try:
                            categories = os.listdir(split_path)
                            for category_dir in categories:
                                category_path = osp.join(split_path, category_dir)
                                if osp.isdir(category_path):
                                    class_id = self._get_class_id_from_dataset(dataset_name, category_dir)
                                    
                                    # 遍历类别目录中的图像文件
                                    try:
                                        img_files = [f for f in os.listdir(category_path) 
                                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                                        
                                        for img_file in img_files:
                                            img_path = osp.join(category_path, img_file)
                                            
                                            data_info = {
                                                'img_path': img_path,
                                                'seg_map_path': None,  # 分类数据没有分割标注
                                                'label': class_id,
                                                'category': category_dir,
                                                'dataset': dataset_name,
                                                'split': split,
                                                'modality': self.modality,
                                                'task_type': self.task_type,
                                                'seg_fields': []  # 没有真实的分割标注
                                            }
                                            
                                            if self.instruction_format:
                                                data_info['instruction'] = f"Perform semantic segmentation on this {dataset_name} remote sensing image showing {category_dir}."
                                                data_info['response'] = f"Segmentation mask for {category_dir} land cover type."
                                                
                                            data_list.append(data_info)
                                    except Exception as e:
                                        print(f"[MMRS1M] 处理类别{category_dir}时出错: {e}")
                                        continue
                        except Exception as e:
                            print(f"[MMRS1M] 处理{split}分割时出错: {e}")
                            continue
                            
            elif osp.exists(images_path):
                # 只有images目录的数据集
                try:
                    img_files = [f for f in os.listdir(images_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                    
                    for img_file in img_files:
                        img_path = osp.join(images_path, img_file)
                        
                        data_info = {
                            'img_path': img_path,
                            'seg_map_path': None,
                            'label': 0,  # 默认类别
                            'category': 'unknown',
                            'dataset': dataset_name,
                            'split': 'all',
                            'modality': self.modality,
                            'task_type': self.task_type,
                            'seg_fields': []
                        }
                        
                        if self.instruction_format:
                            data_info['instruction'] = f"Perform semantic segmentation on this {dataset_name} remote sensing image."
                            data_info['response'] = "Semantic segmentation mask with land cover classes."
                            
                        data_list.append(data_info)
                except Exception as e:
                    print(f"[MMRS1M] 处理images目录时出错: {e}")
        
        return data_list

    def _load_caption_data(self) -> List[dict]:
        """加载图像描述任务数据。"""
        data_list = []
        
        if not self.data_root:
            return data_list
            
        # 根据真实MMRS1M数据结构加载caption数据
        caption_dir = osp.join(self.data_root, 'caption')
        json_dir = osp.join(self.data_root, 'json', 'caption')
        
        if not osp.exists(caption_dir):
            return data_list
            
        # 真实的caption数据集列表
        caption_datasets = [
            'RSICD', 'Sydney_captions', 'UCM_captions'
        ]
        
        # JSON文件映射
        json_mapping = {
            'RSICD': 'RSICD.json',
            'Sydney_captions': 'Sydney_captions.json',
            'UCM_captions': 'UCM_captions.json'
        }
        
        # 遍历每个caption数据集
        for dataset_name in caption_datasets:
            dataset_path = osp.join(caption_dir, dataset_name)
            if not osp.exists(dataset_path):
                continue
                
            # 查找对应的JSON标注文件
            json_file = None
            if dataset_name in json_mapping:
                json_path = osp.join(json_dir, json_mapping[dataset_name])
                if osp.exists(json_path):
                    json_file = json_path
            
            # 查找图像目录
            images_path = osp.join(dataset_path, 'images')
            if not osp.exists(images_path):
                continue
                
            # 加载JSON标注数据
            caption_data = {}
            if json_file:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        if isinstance(json_data, list):
                            for item in json_data:
                                if isinstance(item, dict) and 'image' in item:
                                    img_name = item['image']
                                    if img_name not in caption_data:
                                        caption_data[img_name] = []
                                    caption_data[img_name].append(item.get('caption', 'Remote sensing image'))
                        elif isinstance(json_data, dict):
                            caption_data = json_data
                except:
                    pass
                
            # 遍历图像文件
            for img_file in os.listdir(images_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    img_path = osp.join(images_path, img_file)
                    
                    # 获取对应的描述
                    captions = caption_data.get(img_file, ['Remote sensing image'])
                    caption = captions[0] if captions else 'Remote sensing image'
                    
                    data_info = {
                        'img_path': img_path,
                        'seg_map_path': None,
                        'label': None,
                        'caption': caption,
                        'captions': captions,
                        'dataset': dataset_name,
                        'modality': self.modality,
                        'task_type': self.task_type,
                        'json_file': json_file,
                        'seg_fields': []
                    }
                    
                    if self.instruction_format:
                        data_info['instruction'] = "Describe what you see in this remote sensing image"
                        data_info['response'] = caption
                        
                    data_list.append(data_info)
        
        return data_list
    
    def _load_vqa_data(self) -> List[dict]:
        """加载视觉问答任务数据。"""
        data_list = []
        
        if not self.data_root:
            return data_list
            
        # 根据真实MMRS1M数据结构加载VQA数据
        vqa_dir = osp.join(self.data_root, 'VQA')
        json_dir = osp.join(self.data_root, 'json', 'VQA')
        
        if not osp.exists(vqa_dir):
            return data_list
            
        # 真实的VQA数据集列表
        vqa_datasets = [
            'floodnet', 'MQVQA_dataset', 'RSIVQA', 'rsvqa_high'
        ]
        
        # JSON文件映射
        json_mapping = {
            'floodnet': 'floodnet.json',
            'MQVQA_dataset': 'MQVQA_train.json',
            'RSIVQA': 'sydney_vqa.json',  # 可能需要根据实际情况调整
            'rsvqa_high': 'rsvqa_high.json'
        }
        
        # 遍历每个VQA数据集
        for dataset_name in vqa_datasets:
            dataset_path = osp.join(vqa_dir, dataset_name)
            if not osp.exists(dataset_path):
                continue
                
            # 查找对应的JSON标注文件
            json_file = None
            if dataset_name in json_mapping:
                json_path = osp.join(json_dir, json_mapping[dataset_name])
                if osp.exists(json_path):
                    json_file = json_path
            
            # 查找图像目录
            images_path = osp.join(dataset_path, 'images')
            if not osp.exists(images_path):
                continue
                
            # 加载JSON标注数据
            vqa_data = {}
            if json_file:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        if isinstance(json_data, list):
                            for item in json_data:
                                if isinstance(item, dict) and 'image' in item:
                                    img_name = item['image']
                                    if img_name not in vqa_data:
                                        vqa_data[img_name] = []
                                    
                                    qa_pair = {
                                        'question': item.get('question', 'What do you see in this image?'),
                                        'answer': item.get('answer', 'This is a remote sensing image.')
                                    }
                                    vqa_data[img_name].append(qa_pair)
                        elif isinstance(json_data, dict):
                            vqa_data = json_data
                except:
                    pass
                
            # 遍历图像文件
            for img_file in os.listdir(images_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    img_path = osp.join(images_path, img_file)
                    
                    # 获取对应的问答对
                    qa_pairs = vqa_data.get(img_file, [])
                    if not qa_pairs:
                        # 跳过没有标注的图像
                        continue
                    
                    # 为每个问答对创建一个数据项
                    for qa_pair in qa_pairs:
                        data_info = {
                            'img_path': img_path,
                            'seg_map_path': None,
                            'label': None,
                            'question': qa_pair['question'],
                            'answer': qa_pair['answer'],
                            'dataset': dataset_name,
                            'modality': self.modality,
                            'task_type': self.task_type,
                            'json_file': json_file,
                            'seg_fields': []
                        }
                        
                        if self.instruction_format:
                            data_info['instruction'] = qa_pair['question']
                            data_info['response'] = qa_pair['answer']
                            
                        data_list.append(data_info)
        
        return data_list
    
    def _load_rsvg_data(self) -> List[dict]:
        """加载视觉定位任务数据。"""
        data_list = []
        
        if not self.data_root:
            return data_list
            
        # 根据真实MMRS1M数据结构加载RSVG数据
        rsvg_dir = osp.join(self.data_root, 'RSVG')
        json_dir = osp.join(self.data_root, 'json', 'RSVG')
        
        if not osp.exists(rsvg_dir):
            return data_list
            
        # RSVG数据集目录
        dior_rsvg_dir = osp.join(rsvg_dir, 'DIOR_RSVG')
        if not osp.exists(dior_rsvg_dir):
            return data_list
            
        # 查找图像目录
        images_path = osp.join(dior_rsvg_dir, 'images')
        if not osp.exists(images_path):
            return data_list
            
        # 查找JSON标注文件
        json_file = osp.join(json_dir, 'rsvg_trainval.json')
        
        # 加载JSON标注数据
        rsvg_data = {}
        if osp.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        for item in json_data:
                            if isinstance(item, dict) and 'image' in item:
                                img_name = item['image']
                                rsvg_data[img_name] = {
                                    'expression': item.get('expression', 'Locate the object in this image'),
                                    'bbox': item.get('bbox', [0, 0, 100, 100]),
                                    'target': item.get('target', 'object')
                                }
                    elif isinstance(json_data, dict):
                        rsvg_data = json_data
            except Exception as e:
                print(f"Error loading RSVG annotation file {json_file}: {e}")
                
        # 遍历图像文件
        for img_file in os.listdir(images_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                img_path = osp.join(images_path, img_file)
                
                # 获取对应的RSVG标注
                rsvg_info = rsvg_data.get(img_file, {})
                if not rsvg_info:
                    # 跳过没有标注的图像
                    continue
                    
                expression = rsvg_info.get('expression', 'Locate the main object in this remote sensing image')
                bbox = rsvg_info.get('bbox', [0, 0, 100, 100])
                target = rsvg_info.get('target', 'object')
                
                data_info = {
                    'img_path': img_path,
                    'seg_map_path': None,
                    'label': None,
                    'expression': expression,
                    'bbox': bbox,
                    'target': target,
                    'dataset': 'DIOR_RSVG',
                    'modality': self.modality,
                    'task_type': self.task_type,
                    'json_file': json_file if osp.exists(json_file) else None,
                    'seg_fields': []
                }
                
                if self.instruction_format:
                    data_info['instruction'] = expression
                    data_info['response'] = f"The {target} is located at coordinates {bbox}"
                    
                data_list.append(data_info)
        
        return data_list
    
    def _get_class_id(self, category: str) -> int:
        """获取类别ID。"""
        class_mapping = {
            'background': 0,
            'building': 1, 
            'road': 2,
            'water': 3,
            'barren': 4,
            'forest': 5,
            'agricultural': 6
        }
        return class_mapping.get(category, 0)
    
    def _get_class_id_from_dataset(self, dataset_name: str, category: str) -> int:
        """根据数据集名称和类别获取类别ID。"""
        # 为不同数据集定义类别映射
        dataset_mappings = {
            'EuroSAT_split': {
                'AnnualCrop': 0, 'Forest': 1, 'HerbaceousVegetation': 2, 'Highway': 3,
                'Industrial': 4, 'Pasture': 5, 'PermanentCrop': 6, 'Residential': 7,
                'River': 8, 'SeaLake': 9
            },
            'NWPU-RESISC45_split': {
                'airplane': 0, 'airport': 1, 'baseball_diamond': 2, 'basketball_court': 3,
                'beach': 4, 'bridge': 5, 'chaparral': 6, 'church': 7, 'circular_farmland': 8,
                'cloud': 9, 'commercial_area': 10, 'dense_residential': 11, 'desert': 12,
                'forest': 13, 'freeway': 14, 'golf_course': 15, 'ground_track_field': 16,
                'harbor': 17, 'industrial_area': 18, 'intersection': 19, 'island': 20,
                'lake': 21, 'meadow': 22, 'medium_residential': 23, 'mobile_home_park': 24,
                'mountain': 25, 'overpass': 26, 'palace': 27, 'parking_lot': 28,
                'railway': 29, 'railway_station': 30, 'rectangular_farmland': 31,
                'river': 32, 'roundabout': 33, 'runway': 34, 'sea_ice': 35,
                'ship': 36, 'snowberg': 37, 'sparse_residential': 38, 'stadium': 39,
                'storage_tank': 40, 'tennis_court': 41, 'terrace': 42, 'thermal_power_station': 43,
                'wetland': 44
            },
            'UCMerced_split': {
                'agricultural': 0, 'airplane': 1, 'baseballdiamond': 2, 'beach': 3,
                'buildings': 4, 'chaparral': 5, 'denseresidential': 6, 'forest': 7,
                'freeway': 8, 'golfcourse': 9, 'harbor': 10, 'intersection': 11,
                'mediumresidential': 12, 'mobilehomepark': 13, 'overpass': 14,
                'parkinglot': 15, 'river': 16, 'runway': 17, 'sparseresidential': 18,
                'storagetanks': 19, 'tenniscourt': 20
            }
        }
        
        # 如果数据集有特定映射，使用特定映射
        if dataset_name in dataset_mappings:
            mapping = dataset_mappings[dataset_name]
            return mapping.get(category, len(mapping))  # 未知类别返回最大ID+1
        
        # 否则使用通用映射
        return self._get_class_id(category)
    
    def get_gt_seg_map_by_idx(self, index: int) -> np.ndarray:
        """获取分割标注图。"""
        data_info = self.data_list[index]
        
        if data_info.get('seg_map_path') is None:
            # 对于分类等任务，创建虚拟分割图
            return np.zeros((512, 512), dtype=np.uint8)
            
        # 加载实际分割图
        seg_map_path = data_info['seg_map_path']
        if mmcv is not None:
            seg_map = mmcv.imread(seg_map_path, flag='unchanged', backend='pillow')
        else:
            # 使用PIL作为fallback
            seg_map = np.array(Image.open(seg_map_path))
        return seg_map.squeeze() if seg_map.ndim == 3 else seg_map