#!/usr/bin/env python3
"""
LinkAble P0 四类模型训练脚本
整合 mangdaojianceYOLO 数据集和其他公共数据集

用法:
    python train_p0_model.py --mode prepare   # 准备数据
    python train_p0_model.py --mode train     # 训练模型
    python train_p0_model.py --mode evaluate  # 评估模型
    python train_p0_model.py --mode export    # 导出模型
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent  # yds 目录
MANGDAOJIANE_DIR = PROJECT_ROOT / "mangdaojianceYOLO"
EDGE_P0_DIR = PROJECT_ROOT / "edge_p0"
DATASETS_DIR = EDGE_P0_DIR / "datasets"
P0_YOLO_DIR = DATASETS_DIR / "p0_yolo"
CONFIGS_DIR = EDGE_P0_DIR / "configs"

# P0 四类定义
P0_CLASSES = {
    'blind_road_occupied': 0,
    'stairs': 1,
    'ramp': 2,
    'road_obstacle': 3,
}

# 训练配置
TRAIN_CONFIG = {
    'model': 'yolo11n.pt',  # 使用 YOLOv11n 轻量级模型
    'epochs': 150,
    'imgsz': 640,
    'batch': 16,
    'patience': 30,
    'device': 0,  # GPU
    'workers': 4,
    'optimizer': 'auto',
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'erasing': 0.4,
    'augment': True,
    'verbose': True,
    'seed': 42,
    'deterministic': True,
    'single_cls': False,
    'rect': False,
    'cos_lr': False,
    'close_mosaic': 10,
    'amp': True,
    'fraction': 1.0,
    'multi_scale': False,
    'overlap_mask': True,
    'mask_ratio': 4,
    'dropout': 0.0,
    'val': True,
    'save_json': False,
    'save_hybrid': False,
    'conf': None,
    'iou': 0.7,
    'max_det': 300,
    'half': False,
    'dnn': False,
    'plots': True,
    'show': False,
    'save_frames': False,
    'save_txt': False,
    'save_conf': False,
    'save_crop': False,
    'show_labels': True,
    'show_conf': True,
    'show_boxes': True,
    'line_width': None,
    'format': 'torchscript',
    'keras': False,
    'optimize': False,
    'int8': False,
    'dynamic': False,
    'simplify': True,
    'opset': None,
    'workspace': 4,
    'nms': False,
    'profile': False,
    'freeze': None,
    'label_smoothing': 0.0,
    'nbs': 64,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'bgr': 0.0,
    'auto_augment': 'randaugment',
    'crop_fraction': 1.0,
    'cfg': None,
    'tracker': 'botsort.yaml',
}


def check_mangdaojiance_dataset():
    """检查 P0 数据集是否存在"""
    data_yaml = P0_YOLO_DIR / "data.yaml"
    train_images = P0_YOLO_DIR / "images" / "train"
    
    if not data_yaml.exists():
        print(f"错误: 找不到 {data_yaml}")
        return False
    
    if not train_images.exists() or not any(train_images.iterdir()):
        print(f"错误: 找不到训练图片 {train_images}")
        return False
    
    train_count = len(list(train_images.iterdir()))
    print(f"[OK] P0 数据集检查通过")
    print(f"  训练图片: {train_count} 张")
    return True


def prepare_data():
    """准备 P0 四类数据集"""
    print("=" * 60)
    print("步骤 1: 转换 mangdaojianceYOLO 数据集到 P0 四类格式")
    print("=" * 60)
    
    # 运行转换脚本
    convert_script = EDGE_P0_DIR / "scripts" / "convert_mangdaojiance_to_p0.py"
    
    if not convert_script.exists():
        print(f"错误: 找不到转换脚本 {convert_script}")
        return False
    
    cmd = [
        sys.executable,
        str(convert_script),
        "--source_dir", str(MANGDAOJIANE_DIR),
        "--output_dir", str(P0_YOLO_DIR),
        "--split", "all",
    ]
    
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("错误: 数据转换失败")
        return False
    
    print("\n" + "=" * 60)
    print("步骤 2: 检查转换结果")
    print("=" * 60)
    
    # 检查转换结果
    data_yaml = P0_YOLO_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"错误: 找不到 {data_yaml}")
        return False
    
    with open(data_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ 数据配置: {config}")
    
    # 统计各类别数量
    labels_dir = P0_YOLO_DIR / "labels" / "train"
    if labels_dir.exists():
        class_counts = {name: 0 for name in P0_CLASSES}
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_idx = int(parts[0])
                        for name, idx in P0_CLASSES.items():
                            if idx == class_idx:
                                class_counts[name] += 1
                                break
        
        print("\n各类别标注框数:")
        for name, count in class_counts.items():
            print(f"  {name}: {count}")
    
    print("\n" + "=" * 60)
    print("步骤 3: 数据集统计")
    print("=" * 60)
    
    # 统计各类别数量
    labels_dir = P0_YOLO_DIR / "labels" / "train"
    class_counts = {name: 0 for name in P0_CLASSES}
    if labels_dir.exists():
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_idx = int(parts[0])
                        for name, idx in P0_CLASSES.items():
                            if idx == class_idx:
                                class_counts[name] += 1
                                break
    
    print("\n各类别标注框数:")
    total_boxes = sum(class_counts.values())
    for name, count in class_counts.items():
        percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"  {name}: {count} ({percentage:.1f}%)")
    
    print(f"\n总计: {total_boxes} 个标注框")
    
    # 检查类别平衡
    print("\n类别平衡分析:")
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    if min_count > 0:
        imbalance_ratio = max_count / min_count
        print(f"  最小类别: {min(class_counts, key=class_counts.get)} ({min_count})")
        print(f"  最大类别: {max(class_counts, key=class_counts.get)} ({max_count})")
        print(f"  不平衡比例: {imbalance_ratio:.1f}x")
        
        if imbalance_ratio > 10:
            print("  警告: 类别严重不平衡，建议优先补采少数类或做数据重采样")
            print("  注意: 当前 Ultralytics 版本不支持 class_weights 训练参数，脚本不会传入该参数")
    
    return True


def train_model():
    """训练 P0 四类模型"""
    print("=" * 60)
    print("开始训练 P0 四类 YOLO 模型")
    print("=" * 60)
    
    # 检查数据集
    data_yaml = P0_YOLO_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"错误: 找不到数据配置 {data_yaml}")
        print("请先运行: python train_p0_model.py --mode prepare")
        return False
    
    # 加载数据配置
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print(f"数据配置: {data_config}")
    
    # 检查训练图片
    train_images = P0_YOLO_DIR / "images" / "train"
    if not train_images.exists() or not any(train_images.iterdir()):
        print(f"错误: 找不到训练图片 {train_images}")
        return False
    
    train_count = len(list(train_images.iterdir()))
    print(f"训练图片数量: {train_count}")
    
    if train_count < 100:
        print("警告: 训练图片数量过少，建议至少 500 张")
        print("是否继续训练？(y/n)")
        response = input().strip().lower()
        if response != 'y':
            return False
    
    # 构建训练命令
    model_path = MANGDAOJIANE_DIR / "yolo11n.pt"
    if not model_path.exists():
        model_path = "yolo11n.pt"  # 使用默认预训练权重
    
    output_dir = EDGE_P0_DIR / "runs" / "train" / "p0_mangdaojiance"
    
    # 计算类别权重（解决类别不平衡问题）
    # 基于各类别数量计算权重，数量越少权重越高
    labels_dir = P0_YOLO_DIR / "labels" / "train"
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    if labels_dir.exists():
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_idx = int(parts[0])
                        if class_idx in class_counts:
                            class_counts[class_idx] += 1
    
    # 计算权重：总数 / (类别数 * 该类别数量)
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    class_weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total / (num_classes * count) if count > 0 else 1.0
        class_weights.append(round(weight, 2))
    
    print(f"\n类别分布:")
    for i, count in class_counts.items():
        print(f"  类别 {i}: {count} 样本, 参考逆频权重: {class_weights[i]}")
    print("注意: 参考权重仅用于判断类别不平衡；当前 Ultralytics CLI 不传 class_weights。")
    
    cmd = [
        sys.executable, "-m", "ultralytics", "detect", "train",
        f"model={model_path}",
        f"data={data_yaml}",
        f"epochs={TRAIN_CONFIG['epochs']}",
        f"imgsz={TRAIN_CONFIG['imgsz']}",
        f"batch={TRAIN_CONFIG['batch']}",
        f"patience={TRAIN_CONFIG['patience']}",
        f"device={TRAIN_CONFIG['device']}",
        f"workers={TRAIN_CONFIG['workers']}",
        f"optimizer={TRAIN_CONFIG['optimizer']}",
        f"lr0={TRAIN_CONFIG['lr0']}",
        f"lrf={TRAIN_CONFIG['lrf']}",
        f"momentum={TRAIN_CONFIG['momentum']}",
        f"weight_decay={TRAIN_CONFIG['weight_decay']}",
        f"warmup_epochs={TRAIN_CONFIG['warmup_epochs']}",
        f"mosaic={TRAIN_CONFIG['mosaic']}",
        f"mixup={TRAIN_CONFIG['mixup']}",
        f"erasing={TRAIN_CONFIG['erasing']}",
        f"augment={TRAIN_CONFIG['augment']}",
        f"verbose={TRAIN_CONFIG['verbose']}",
        f"seed={TRAIN_CONFIG['seed']}",
        f"deterministic={TRAIN_CONFIG['deterministic']}",
        f"amp={TRAIN_CONFIG['amp']}",
        f"val={TRAIN_CONFIG['val']}",
        f"plots={TRAIN_CONFIG['plots']}",
        f"project={output_dir.parent}",
        f"name={output_dir.name}",
        "exist_ok=True",
    ]
    
    print(f"\n训练命令:")
    print(" ".join(cmd))
    print(f"\n输出目录: {output_dir}")
    print(f"\n开始训练... (预计需要 2-4 小时)")
    
    # 运行训练
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("错误: 训练失败")
        return False
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    best_weights = output_dir / "weights" / "best.pt"
    last_weights = output_dir / "weights" / "last.pt"
    
    print(f"最佳权重: {best_weights}")
    print(f"最后权重: {last_weights}")
    
    return True


def evaluate_model():
    """评估训练好的模型"""
    print("=" * 60)
    print("评估 P0 四类模型")
    print("=" * 60)
    
    # 查找最佳权重
    best_weights = EDGE_P0_DIR / "runs" / "train" / "p0_mangdaojiance" / "weights" / "best.pt"
    
    if not best_weights.exists():
        print(f"错误: 找不到模型权重 {best_weights}")
        print("请先运行训练: python train_p0_model.py --mode train")
        return False
    
    # 数据配置
    data_yaml = P0_YOLO_DIR / "data.yaml"
    
    # 运行评估
    cmd = [
        sys.executable, "-m", "ultralytics", "detect", "val",
        f"model={best_weights}",
        f"data={data_yaml}",
        "split=test",
        "plots=True",
        "save_json=True",
    ]
    
    print(f"评估命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("错误: 评估失败")
        return False
    
    print("\n评估完成！")
    return True


def export_model():
    """导出模型用于边缘部署"""
    print("=" * 60)
    print("导出模型用于边缘部署")
    print("=" * 60)
    
    # 查找最佳权重
    best_weights = EDGE_P0_DIR / "runs" / "train" / "p0_mangdaojiance" / "weights" / "best.pt"
    
    if not best_weights.exists():
        print(f"错误: 找不到模型权重 {best_weights}")
        print("请先运行训练: python train_p0_model.py --mode train")
        return False
    
    # 导出格式
    formats = {
        'torchscript': 'TorchScript (推荐用于 PyTorch)',
        'onnx': 'ONNX (通用格式)',
        'engine': 'TensorRT (Jetson 部署)',
        'openvino': 'OpenVINO (Intel 设备)',
        'coreml': 'CoreML (Apple 设备)',
    }
    
    print("可用导出格式:")
    for fmt, desc in formats.items():
        print(f"  {fmt}: {desc}")
    
    print("\n请输入要导出的格式 (默认: onnx):")
    fmt = input().strip().lower() or 'onnx'
    
    if fmt not in formats:
        print(f"错误: 不支持的格式 {fmt}")
        return False
    
    # 运行导出
    cmd = [
        sys.executable, "-m", "ultralytics", "export",
        f"model={best_weights}",
        f"format={fmt}",
        "imgsz=640",
        "simplify=True",
    ]
    
    print(f"导出命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("错误: 导出失败")
        return False
    
    print("\n导出完成！")
    print(f"导出文件位于: {best_weights.parent}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='LinkAble P0 四类模型训练')
    parser.add_argument(
        '--mode',
        choices=['prepare', 'train', 'evaluate', 'export'],
        required=True,
        help='运行模式'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("LinkAble P0 四类模型训练工具")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"项目根目录: {PROJECT_ROOT}")
    print()
    
    # 检查 mangdaojianceYOLO 数据集
    if not check_mangdaojiance_dataset():
        return 1
    
    if args.mode == 'prepare':
        success = prepare_data()
    elif args.mode == 'train':
        success = train_model()
    elif args.mode == 'evaluate':
        success = evaluate_model()
    elif args.mode == 'export':
        success = export_model()
    else:
        print(f"错误: 未知模式 {args.mode}")
        return 1
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
