#!/usr/bin/env python3
"""
重新划分 P0 数据集
目标：Train 86%, Val 9%, Test 5%

用法: python resplit_dataset.py
"""

import shutil
import random
from pathlib import Path

# 配置
P0_DIR = Path('D:/000/yds/edge_p0/datasets/p0_yolo')
RANDOM_SEED = 42

# 划分比例
SPLITS = {
    'train': 0.86,
    'val': 0.09,
    'test': 0.05,
}

def main():
    random.seed(RANDOM_SEED)
    
    print("=" * 60)
    print("重新划分 P0 数据集")
    print("=" * 60)
    
    # 获取所有图片
    train_images = list((P0_DIR / 'images' / 'train').glob('*'))
    val_images = list((P0_DIR / 'images' / 'val').glob('*'))
    test_images = list((P0_DIR / 'images' / 'test').glob('*'))
    
    all_images = train_images + val_images + test_images
    total = len(all_images)
    
    print(f"总图片数: {total}")
    print(f"划分比例: Train {SPLITS['train']*100}%, Val {SPLITS['val']*100}%, Test {SPLITS['test']*100}%")
    
    # 打乱顺序
    random.shuffle(all_images)
    
    # 计算划分点
    train_end = int(total * SPLITS['train'])
    val_end = train_end + int(total * SPLITS['val'])
    
    # 划分数据集
    train_files = all_images[:train_end]
    val_files = all_images[train_end:val_end]
    test_files = all_images[val_end:]
    
    print(f"\n划分结果:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")
    
    # 创建临时目录
    temp_dir = P0_DIR / 'temp_split'
    for split in ['train', 'val', 'test']:
        (temp_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (temp_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 复制文件到临时目录
    print(f"\n正在移动文件...")
    
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for img_path in files:
            # 复制图片
            dst_img = temp_dir / 'images' / split_name / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # 复制对应的标签
            label_path = P0_DIR / 'labels' / img_path.parent.parent.name / (img_path.stem + '.txt')
            if label_path.exists():
                dst_label = temp_dir / 'labels' / split_name / (img_path.stem + '.txt')
                shutil.copy2(label_path, dst_label)
    
    # 删除旧目录，重命名临时目录
    print(f"\n正在更新目录...")
    
    # 备份旧目录
    backup_dir = P0_DIR / 'backup'
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    # 移动旧目录到备份
    for split in ['train', 'val', 'test']:
        old_img_dir = P0_DIR / 'images' / split
        old_lbl_dir = P0_DIR / 'labels' / split
        
        if old_img_dir.exists():
            backup_img = backup_dir / 'images' / split
            backup_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_img_dir), str(backup_img))
        
        if old_lbl_dir.exists():
            backup_lbl = backup_dir / 'labels' / split
            backup_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_lbl_dir), str(backup_lbl))
    
    # 移动临时目录到正式目录
    for split in ['train', 'val', 'test']:
        src_img = temp_dir / 'images' / split
        dst_img = P0_DIR / 'images' / split
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))
        
        src_lbl = temp_dir / 'labels' / split
        dst_lbl = P0_DIR / 'labels' / split
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dst_lbl))
    
    # 删除临时目录
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # 验证结果
    print(f"\n" + "=" * 60)
    print(f"验证结果")
    print(f"=" * 60)
    
    for split in ['train', 'val', 'test']:
        img_count = len(list((P0_DIR / 'images' / split).glob('*')))
        lbl_count = len(list((P0_DIR / 'labels' / split).glob('*.txt')))
        match = "✓" if img_count == lbl_count else "✗"
        print(f"  {split}: {img_count} images, {lbl_count} labels {match}")
    
    # 统计类别分布
    print(f"\n类别分布 (train):")
    stats = {0: 0, 1: 0, 2: 0, 3: 0}
    for label_file in (P0_DIR / 'labels' / 'train').glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_idx = int(parts[0])
                    if class_idx in stats:
                        stats[class_idx] += 1
    
    classNames = ['blind_road_occupied', 'stairs', 'ramp', 'road_obstacle']
    total_boxes = sum(stats.values())
    for i, name in enumerate(classNames):
        pct = stats[i] / total_boxes * 100 if total_boxes > 0 else 0
        print(f"  {name}: {stats[i]} ({pct:.1f}%)")
    
    print(f"\n备份位置: {backup_dir}")
    print(f"提示: 确认无误后可删除备份目录")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
