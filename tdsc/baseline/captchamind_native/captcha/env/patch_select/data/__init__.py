import json
import os
import glob
from typing import Any, List

# 修改这里：指定你的数据集根目录
DATASET_ROOT = "/mnt/workspace/workgroup/wangpengcheng//data/patch_select/patch_select"  # 改成你的数据集路径

# tick图片的固定路径
TICK_IMAGE_PATH = "/mnt/workspace/workgroup/wangpengcheng/captcha_solve/captcha/env/patch_select/data/tick.png"

# 缓存文件夹路径和当前样本路径
_sample_folders = None
_current_sample_path = None


def load_data() -> List[Any]:
    """加载指定范围的样本数据 (sample_1801 到 sample_2000)"""
    global _sample_folders
    
    # 扫描指定范围的sample文件夹
    _sample_folders = []
    for i in range(1801, 2001):  # 1801到2000，共200个样本
        sample_folder = os.path.join(DATASET_ROOT, f"sample_{i:04d}")
        if os.path.exists(sample_folder):
            _sample_folders.append(sample_folder)
    
    if not _sample_folders:
        raise ValueError(f"在 {DATASET_ROOT} 中没有找到sample_1801到sample_2000文件夹")
    
    print(f"找到 {len(_sample_folders)} 个样本文件夹 (sample_1801 到 sample_2000)")
    
    # 加载所有样本的配置文件
    result = []
    for i, sample_folder in enumerate(_sample_folders):
        json_path = os.path.join(sample_folder, "task.json")
        
        if not os.path.exists(json_path):
            print(f"警告: 样本 {os.path.basename(sample_folder)} 缺少配置文件: {json_path}")
            continue
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            
            # 添加样本路径信息，用于后续图片路径构建
            sample_data['_sample_path'] = sample_folder
            # 添加固定的tick_image路径
            sample_data['tick_image'] = 'tick.png'  # 这里只存文件名，实际路径由get_tick_image_path()处理
            result.append(sample_data)
            
        except Exception as e:
            print(f"警告: 加载样本 {os.path.basename(sample_folder)} 失败: {e}")
            continue
    
    print(f"成功加载 {len(result)} 个样本")
    return result


def get_image_path(image_name: str) -> str:
    """获取样本图片的完整路径"""
    global _current_sample_path
    
    if _current_sample_path is None:
        raise ValueError("当前样本路径未设置，请先调用 reset()")
    
    return os.path.join(_current_sample_path, image_name)


def get_tick_image_path() -> str:
    """获取固定tick图片的完整路径"""
    return TICK_IMAGE_PATH


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