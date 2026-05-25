"""
工具函数模块

提供日志配置、时间戳生成、路径工具、图像处理辅助函数等通用功能。
"""

import logging
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np


def setup_logger(name: str, config: dict) -> logging.Logger:
    """
    配置日志器。
    
    支持控制台输出和文件输出，根据配置参数进行灵活设置。
    
    Args:
        name: 日志器名称
        config: 日志配置字典，包含level、save_to_file、log_dir等键
        
    Returns:
        logging.Logger: 配置好的日志器实例
        
    Example:
        >>> config = {"level": "INFO", "save_to_file": True}
        >>> logger = setup_logger("MyApp", config)
        >>> logger.info("程序启动")
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 设置日志级别
    level_str = config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler（可选）
    if config.get("save_to_file", False):
        log_dir = Path(config.get("log_dir", "output/logs"))
        ensure_dir(log_dir)
        
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.get("max_bytes", 10485760),  # 默认10MB
            backupCount=config.get("backup_count", 5),
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def generate_timestamp() -> str:
    """
    生成ISO 8601格式时间戳。
    
    Returns:
        str: ISO 8601格式的时间戳字符串
        
    Example:
        >>> generate_timestamp()
        '2024-01-15T10:30:45.123456'
    """
    return datetime.utcnow().isoformat()


def generate_alarm_id() -> str:
    """
    生成告警ID。
    
    格式：alarm_YYYYMMDD_HHMMSS_random4digits
    
    Returns:
        str: 唯一的告警ID字符串
        
    Example:
        >>> generate_alarm_id()
        'alarm_20240115_103045_3281'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_digits = random.randint(1000, 9999)
    return f"alarm_{timestamp}_{random_digits}"


def ensure_dir(path: Path) -> Path:
    """
    确保目录存在，不存在则创建。
    
    Args:
        path: 目录路径（Path对象）
        
    Returns:
        Path: 目录路径（与输入相同）
        
    Example:
        >>> ensure_dir(Path("output/snapshots"))
        PosixPath('output/snapshots')
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_with_aspect_ratio(image: np.ndarray, scale: float,
                              interpolation=cv2.INTER_CUBIC) -> np.ndarray:
    """
    按比例缩放图像。
    
    Args:
        image: 输入图像（numpy数组）
        scale: 缩放比例（>1为放大，<1为缩小）
        interpolation: 插值方法，默认cv2.INTER_CUBIC
        
    Returns:
        np.ndarray: 缩放后的图像
        
    Example:
        >>> resized = resize_with_aspect_ratio(image, 2.0)  # 放大2倍
    """
    if scale == 1.0:
        return image
    
    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def draw_text_with_background(frame: np.ndarray, text: str,
                              position: Tuple[int, int], font_scale: float = 0.5,
                              color: Tuple[int, int, int] = (255, 255, 255),
                              bg_color: Tuple[int, int, int] = (0, 0, 0),
                              padding: int = 5) -> np.ndarray:
    """
    在图像上绘制带背景的文字。
    
    提高文字在复杂背景下的可读性。
    
    Args:
        frame: 输入图像帧
        text: 要绘制的文字
        position: 文字左上角坐标 (x, y)
        font_scale: 字体大小缩放因子
        color: 文字颜色 (B, G, R)
        bg_color: 背景颜色 (B, G, R)
        padding: 文字与背景边缘的 padding 像素数
        
    Returns:
        np.ndarray: 绘制后的图像
        
    Example:
        >>> frame = draw_text_with_background(frame, "ALARM!", (50, 50),
        ...                                  color=(0, 0, 255), bg_color=(255, 255, 255))
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    
    # 获取文字尺寸
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    # 背景矩形坐标
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding
    
    # 绘制背景矩形
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # 绘制文字
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame


def calculate_iou(box1: Tuple[int, int, int, int],
                  box2: Tuple[int, int, int, int]) -> float:
    """
    计算两个边界框的IOU（Intersection over Union）。
    
    Args:
        box1: 第一个边界框 (x1, y1, x2, y2)
        box2: 第二个边界框 (x1, y1, x2, y2)
        
    Returns:
        float: IOU值，范围[0, 1]
        
    Example:
        >>> iou = calculate_iou((0, 0, 100, 100), (50, 50, 150, 150))
        >>> print(f"{iou:.3f}")
        0.142
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集区域
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # 如果没有交集
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # 计算交集面积
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算并集面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    # 计算IOU
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    限制数值范围在[min_val, max_val]之间。
    
    Args:
        value: 输入数值
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        float: 限制后的值
        
    Example:
        >>> clamp(150, 0, 100)
        100
        >>> clamp(-10, 0, 100)
        0
        >>> clamp(50, 0, 100)
        50
    """
    return max(min_val, min(value, max_val))


def normalize_points(points: List[dict], width: int, height: int) -> List[Tuple[int, int]]:
    """
    将归一化坐标转换为实际像素坐标。
    
    Args:
        points: 归一化坐标列表 [{"x": 0.1, "y": 0.2}, ...]
        width: 图像宽度
        height: 图像高度
        
    Returns:
        List[Tuple[int, int]]: 像素坐标列表 [(x, y), ...]
        
    Example:
        >>> pts = normalize_points([{"x": 0.5, "y": 0.5}], 1920, 1080)
        >>> pts
        [(960, 540)]
    """
    pixel_points = []
    for pt in points:
        x = int(pt["x"] * width)
        y = int(pt["y"] * height)
        pixel_points.append((x, y))
    return pixel_points


def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """
    判断点是否在多边形内（Ray Casting算法）。
    
    Args:
        point: 待判断的点 (x, y)
        polygon: 多边形顶点列表 [(x1, y1), (x2, y2), ...]
        
    Returns:
        bool: 点在多边形内返回True，否则返回False
        
    Example:
        >>> polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        >>> point_in_polygon((50, 50), polygon)
        True
        >>> point_in_polygon((150, 150), polygon)
        False
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    px1, py1 = polygon[0]
    for i in range(1, n + 1):
        px2, py2 = polygon[i % n]
        
        if y > min(py1, py2):
            if y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xinters = (y - py1) * (px2 - px1) / (py2 - py1) + px1
                    
                    if px1 == px2 or x <= xinters:
                        inside = not inside
        
        px1, py1 = px2, py2
    
    return inside


def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    计算边界框的中心点坐标。
    
    Args:
        bbox: 边界框 (x1, y1, x2, y2)
        
    Returns:
        Tuple[int, int]: 中心点坐标 (cx, cy)
        
    Example:
        >>> get_bbox_center((0, 0, 100, 100))
        (50, 50)
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)


def calculate_solidity(contour: np.ndarray, area: float) -> float:
    """
    计算轮廓的密实度（Solidity）。
    
    密实度 = 轮廓面积 / 凸包面积
    
    Args:
        contour: OpenCV轮廓（numpy数组）
        area: 轮廓面积
        
    Returns:
        float: 密实度值，范围(0, 1]
        
    Example:
        >>> solidity = calculate_solidity(contour, 1000)
    """
    if area <= 0:
        return 0.0
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area <= 0:
        return 0.0
    
    return area / hull_area


def time_it(func):
    """
    装饰器：测量函数执行时间。
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
        
    Example:
        >>> @time_it
        ... def my_function():
        ...     time.sleep(1)
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.4f}秒")
        return result
    return wrapper
