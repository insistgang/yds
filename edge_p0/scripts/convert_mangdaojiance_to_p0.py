#!/usr/bin/env python3
"""
将 mangdaojianceYOLO 16类数据集转换为 LinkAble P0 四类格式
用法: python convert_mangdaojiance_to_p0.py [--output_dir OUTPUT_DIR] [--split SPLIT]
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List
import yaml


# 类别映射配置
LABEL_MAP = {
    'blind track': 'blind_road_occupied',
    'ashcan': 'road_obstacle',
    'car': 'road_obstacle',
    'bicycle': 'road_obstacle',
    'person': 'road_obstacle',
    'spherical_roadblock': 'road_obstacle',
    'pole': 'road_obstacle',
    'fire_hydrant': 'road_obstacle',
    'stop_sign': None,  # 不映射
    'truck': 'road_obstacle',
    'dog': 'road_obstacle',
    'motorbike': 'road_obstacle',
    'warning_column': 'road_obstacle',
    'bus': 'road_obstacle',
    'tricycle': 'road_obstacle',
    'reflective_cone': 'road_obstacle',
}

# P0 四类定义
P0_CLASSES = {
    'blind_road_occupied': 0,
    'stairs': 1,
    'ramp': 2,
    'road_obstacle': 3,
}

# mangdaojianceYOLO 原始类别索引
SOURCE_CLASSES = [
    'blind track', 'ashcan', 'car', 'bicycle', 'person',
    'spherical_roadblock', 'pole', 'fire_hydrant', 'stop_sign',
    'truck', 'dog', 'motorbike', 'warning_column', 'bus',
    'tricycle', 'reflective_cone'
]

SOURCE_CLASS_TO_IDX = {name: idx for idx, name in enumerate(SOURCE_CLASSES)}


def convert_label_file(src_label_path: Path, dst_label_path: Path) -> bool:
    """
    转换单个标签文件
    YOLO 格式: class_id x_center y_center width height
    """
    if not src_label_path.exists():
        return False
    
    converted_lines = []
    with open(src_label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                src_class_idx = int(parts[0])
                if src_class_idx < 0 or src_class_idx >= len(SOURCE_CLASSES):
                    continue
                
                src_class_name = SOURCE_CLASSES[src_class_idx]
                target_class = LABEL_MAP.get(src_class_name)
                
                # 跳过未映射的类别
                if target_class is None:
                    continue
                
                # 获取目标类别索引
                target_class_idx = P0_CLASSES[target_class]
                
                # 重新构建标签行
                new_line = f"{target_class_idx} {' '.join(parts[1:])}"
                converted_lines.append(new_line)
                
            except (ValueError, IndexError):
                continue
    
    # 写入转换后的标签
    dst_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted_lines) + '\n' if converted_lines else '')
    
    return len(converted_lines) > 0


def convert_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
    label_map: Dict[str, str]
) -> Dict[str, int]:
    """
    转换一个数据集分割（train/val/test）
    """
    src_images = source_dir / split / 'images'
    src_labels = source_dir / split / 'labels'
    
    dst_images = output_dir / 'images' / split
    dst_labels = output_dir / 'labels' / split
    
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_images': 0,
        'converted_images': 0,
        'skipped_images': 0,
        'blind_road_occupied': 0,
        'stairs': 0,
        'ramp': 0,
        'road_obstacle': 0,
    }
    
    # 遍历所有图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    for img_path in sorted(src_images.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue
        
        stats['total_images'] += 1
        
        # 对应的标签文件
        label_path = src_labels / (img_path.stem + '.txt')
        
        # 复制图片
        dst_img_path = dst_images / img_path.name
        shutil.copy2(img_path, dst_img_path)
        
        # 转换标签
        dst_label_path = dst_labels / (img_path.stem + '.txt')
        if convert_label_file(label_path, dst_label_path):
            stats['converted_images'] += 1
            
            # 统计各类别数量
            with open(dst_label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_idx = int(parts[0])
                        if class_idx == 0:
                            stats['blind_road_occupied'] += 1
                        elif class_idx == 1:
                            stats['stairs'] += 1
                        elif class_idx == 2:
                            stats['ramp'] += 1
                        elif class_idx == 3:
                            stats['road_obstacle'] += 1
        else:
            stats['skipped_images'] += 1
            # 如果没有有效标签，删除复制的图片
            if dst_img_path.exists():
                dst_img_path.unlink()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='转换 mangdaojianceYOLO 到 P0 四类格式')
    parser.add_argument(
        '--source_dir',
        type=Path,
        default=Path('D:/000/yds/mangdaojianceYOLO'),
        help='mangdaojianceYOLO 数据集目录'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('D:/000/yds/edge_p0/datasets/p0_yolo'),
        help='输出目录'
    )
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test', 'all'],
        default='all',
        help='转换哪个分割'
    )
    args = parser.parse_args()
    
    print(f"源目录: {args.source_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"类别映射: {len(LABEL_MAP)} 个源类别 -> {len(P0_CLASSES)} 个目标类别")
    print()
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换各分割
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    
    total_stats = {
        'total_images': 0,
        'converted_images': 0,
        'skipped_images': 0,
        'blind_road_occupied': 0,
        'stairs': 0,
        'ramp': 0,
        'road_obstacle': 0,
    }
    
    for split in splits:
        print(f"转换 {split} 分割...")
        stats = convert_split(args.source_dir, args.output_dir, split, LABEL_MAP)
        
        for key in total_stats:
            total_stats[key] += stats[key]
        
        print(f"  总图片: {stats['total_images']}")
        print(f"  转换成功: {stats['converted_images']}")
        print(f"  跳过（无有效标签）: {stats['skipped_images']}")
        print(f"  盲道占用框数: {stats['blind_road_occupied']}")
        print(f"  障碍物框数: {stats['road_obstacle']}")
        print()
    
    # 创建 data.yaml
    data_yaml = {
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 4,
        'names': list(P0_CLASSES.keys()),
    }
    
    with open(args.output_dir / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print("=" * 50)
    print("转换完成！")
    print(f"总图片: {total_stats['total_images']}")
    print(f"转换成功: {total_stats['converted_images']}")
    print(f"跳过: {total_stats['skipped_images']}")
    print()
    print("各类别框数统计:")
    print(f"  blind_road_occupied: {total_stats['blind_road_occupied']}")
    print(f"  stairs: {total_stats['stairs']} (需从其他数据源补充)")
    print(f"  ramp: {total_stats['ramp']} (需从其他数据源补充)")
    print(f"  road_obstacle: {total_stats['road_obstacle']}")
    print()
    print(f"输出目录: {args.output_dir}")
    print(f"配置文件: {args.output_dir / 'data.yaml'}")
    print()
    print("注意: stairs 和 ramp 类别需要从以下数据源补充:")
    print("  1. Zenodo accessibility_barriers 数据集")
    print("  2. Roboflow stairs_detection 数据集")
    print("  3. HuggingFace RampNet 数据集")


if __name__ == '__main__':
    main()
