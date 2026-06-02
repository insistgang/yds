#!/usr/bin/env python3
"""
将 stairs_ramps 数据集合并到 P0 四类数据集

用法: python merge_stairs_ramps_dataset.py [--source SOURCE_DIR] [--target TARGET_DIR]
"""

import argparse
import shutil
from pathlib import Path


# 类别索引映射
# stairs_ramps 数据集: ramp=0, stairs=1
# P0 数据集: blind_road_occupied=0, stairs=1, ramp=2, road_obstacle=3
CLASS_MAP = {
    0: 2,  # ramp: 0 -> 2
    1: 1,  # stairs: 1 -> 1 (保持不变)
}


def convert_label_file(src_label_path: Path, dst_label_path: Path):
    """转换单个标签文件的类别索引"""
    if not src_label_path.exists():
        return 0
    
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
                if src_class_idx in CLASS_MAP:
                    dst_class_idx = CLASS_MAP[src_class_idx]
                    new_line = f"{dst_class_idx} {' '.join(parts[1:])}"
                    converted_lines.append(new_line)
            except ValueError:
                continue
    
    # 写入转换后的标签
    dst_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted_lines) + '\n' if converted_lines else '')
    
    return len(converted_lines)


def merge_split(
    source_dir: Path,
    target_dir: Path,
    split: str,
    stats: dict
):
    """合并一个数据集分割"""
    # Roboflow 使用 'valid' 而不是 'val'
    source_split = 'valid' if split == 'val' else split
    
    src_images = source_dir / source_split / 'images'
    src_labels = source_dir / source_split / 'labels'
    
    dst_images = target_dir / 'images' / split
    dst_labels = target_dir / 'labels' / split
    
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    
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
        num_boxes = convert_label_file(label_path, dst_label_path)
        
        if num_boxes > 0:
            stats['converted_images'] += 1
            
            # 统计各类别数量
            with open(dst_label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_idx = int(parts[0])
                        if class_idx == 1:
                            stats['stairs'] += 1
                        elif class_idx == 2:
                            stats['ramp'] += 1
        else:
            stats['skipped_images'] += 1
            # 如果没有有效标签，删除复制的图片
            if dst_img_path.exists():
                dst_img_path.unlink()


def main():
    parser = argparse.ArgumentParser(description='合并 stairs_ramps 数据集到 P0 四类')
    parser.add_argument(
        '--source',
        type=Path,
        default=Path('D:/000/yds/stair&ramps'),
        help='stairs_ramps 数据集目录'
    )
    parser.add_argument(
        '--target',
        type=Path,
        default=Path('D:/000/yds/edge_p0/datasets/p0_yolo'),
        help='P0 数据集目标目录'
    )
    args = parser.parse_args()
    
    print(f"源目录: {args.source}")
    print(f"目标目录: {args.target}")
    print()
    
    # 检查源目录
    if not args.source.exists():
        print(f"错误: 源目录不存在 {args.source}")
        return 1
    
    # 统计信息
    stats = {
        'total_images': 0,
        'converted_images': 0,
        'skipped_images': 0,
        'stairs': 0,
        'ramp': 0,
    }
    
    # 合并各分割
    for split in ['train', 'val', 'test']:
        print(f"合并 {split} 分割...")
        merge_split(args.source, args.target, split, stats)
    
    print()
    print("=" * 60)
    print("合并完成！")
    print("=" * 60)
    print(f"总图片: {stats['total_images']}")
    print(f"转换成功: {stats['converted_images']}")
    print(f"跳过: {stats['skipped_images']}")
    print()
    print("新增类别框数:")
    print(f"  stairs: {stats['stairs']}")
    print(f"  ramp: {stats['ramp']}")
    print()
    print(f"数据已合并到: {args.target}")
    
    # 显示合并后的总统计
    print()
    print("=" * 60)
    print("P0 数据集总统计")
    print("=" * 60)
    
    total_stats = {
        'blind_road_occupied': 0,
        'stairs': 0,
        'ramp': 0,
        'road_obstacle': 0,
    }
    
    labels_dir = args.target / 'labels' / 'train'
    if labels_dir.exists():
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_idx = int(parts[0])
                        if class_idx == 0:
                            total_stats['blind_road_occupied'] += 1
                        elif class_idx == 1:
                            total_stats['stairs'] += 1
                        elif class_idx == 2:
                            total_stats['ramp'] += 1
                        elif class_idx == 3:
                            total_stats['road_obstacle'] += 1
    
    train_images = len(list((args.target / 'images' / 'train').glob("*")))
    print(f"训练图片总数: {train_images}")
    print()
    print("各类别标注框数:")
    for name, count in total_stats.items():
        print(f"  {name}: {count}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
