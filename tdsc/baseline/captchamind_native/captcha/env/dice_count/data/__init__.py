import json
import os
import glob
from typing import Any, List

# 修改这里：指定你的数据集根目录
DATASET_ROOT = "/mnt/workspace/workgroup/wangpengcheng/data/dice_counting/output_test"  # 改成你的数据集路径

# 缓存文件夹路径和当前样本路径
_sample_folders = None
_current_sample_path = None


def load_data() -> List[Any]:
    """加载所有样本数据"""
    global _sample_folders
    
    # 扫描所有sample文件夹
    sample_pattern = os.path.join(DATASET_ROOT, "sample*")
    _sample_folders = sorted(glob.glob(sample_pattern))
    
    if not _sample_folders:
        raise ValueError(f"在 {DATASET_ROOT} 中没有找到sample*文件夹")
    
    print(f"找到 {len(_sample_folders)} 个样本文件夹")
    
    # 加载所有样本的配置文件
    result = []
    for i, sample_folder in enumerate(_sample_folders):
        json_path = os.path.join(sample_folder, "dice_counting.json")
        
        if not os.path.exists(json_path):
            print(f"警告: 样本 {i} 缺少配置文件: {json_path}")
            continue
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            
            # 添加样本路径信息，用于后续图片路径构建
            sample_data['_sample_path'] = sample_folder
            result.append(sample_data)
            
        except Exception as e:
            print(f"警告: 加载样本 {i} 失败: {e}")
            continue
    
    print(f"成功加载 {len(result)} 个样本")
    return result


def get_image_path(image_name: str) -> str:
    """获取图片的完整路径"""
    global _current_sample_path
    
    if _current_sample_path is None:
        # 回退到原始行为，用于加载数字图片等公共资源
        FOLDER_PATH = os.path.dirname(__file__)
        return os.path.join(FOLDER_PATH, image_name)
    
    # 如果是相对路径（包含'/'），可能是公共资源
    if '/' in image_name or image_name.startswith('digit') or image_name == 'white_box.png':
        FOLDER_PATH = os.path.dirname(__file__)
        return os.path.join(FOLDER_PATH, image_name)
    
    # 否则是样本特定的图片
    return os.path.join(_current_sample_path, image_name)


def set_current_sample_path(sample_path: str):
    """设置当前样本路径（给env.py调用）"""
    global _current_sample_path
    _current_sample_path = sample_path


def get_tmp_file_path(filename: str) -> str:
    """获取临时文件路径"""
    global _current_sample_path
    
    if _current_sample_path is None:
        # 回退到原始行为
        FOLDER_PATH = os.path.dirname(__file__)
        tmp_dir = os.path.join(FOLDER_PATH, 'tmp')
    else:
        tmp_dir = os.path.join(_current_sample_path, 'tmp')
        
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        
    file_path = os.path.join(tmp_dir, filename)
    return file_path