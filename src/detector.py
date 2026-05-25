"""
目标检测模块

基于前景掩膜的目标检测，支持多维过滤（面积、宽高比、密实度）
以及区域过滤功能。

Author: 寇豆码 (Alex)
Date: 2024
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2

from src.utils import setup_logger, calculate_solidity


@dataclass
class DetectionResult:
    """
    检测结果数据结构。
    
    封装单个检测目标的所有相关信息，包括边界框、面积、质心等。
    
    Attributes:
        x: 左上角X坐标（像素）
        y: 左上角Y坐标（像素）
        w: 宽度（像素）
        h: 高度（像素）
        area: 面积（像素数）
        centroid: 质心坐标 (cx, cy)
        confidence: 置信度（传统算法固定为1.0）
        
    Example:
        >>> det = DetectionResult(x=100, y=100, w=50, h=50, area=2500, centroid=(125, 125))
        >>> print(det.to_xyxy())
        (100, 100, 150, 150)
    """
    
    x: int               # 左上角X
    y: int               # 左上角Y
    w: int               # 宽度
    h: int               # 高度
    area: int            # 面积（像素数）
    centroid: Tuple[int, int]   # 质心坐标
    confidence: float = 1.0     # 置信度（传统算法固定为1.0）
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """
        转换为 [x1, y1, x2, y2] 格式（SORT输入）。
        
        Returns:
            Tuple[int, int, int, int]: 边界框坐标 (x1, y1, x2, y2)
            
        Example:
            >>> det = DetectionResult(x=100, y=100, w=50, h=50, area=2500, centroid=(125, 125))
            >>> det.to_xyxy()
            (100, 100, 150, 150)
        """
        return (self.x, self.y, self.x + self.w, self.y + self.h)
    
    def to_xyah(self) -> Tuple[int, int, float, float]:
        """
        转换为 [x, y, a, h] 格式（中心点和宽高比）。
        
        该格式常用于目标跟踪和轨迹分析。
        
        Returns:
            Tuple[int, int, float, float]: (cx, cy, aspect_ratio, height)
            
        Example:
            >>> det = DetectionResult(x=100, y=100, w=50, h=50, area=2500, centroid=(125, 125))
            >>> det.to_xyah()
            (125, 125, 1.0, 50)
        """
        return (self.x + self.w//2, self.y + self.h//2, self.w/self.h, self.h)


class Detector:
    """
    基于前景掩膜的目标检测器。
    
    处理流程：
    1. 轮廓检测（cv2.findContours）
    2. 多维过滤：
       - 面积过滤（min_area, max_area）
       - 宽高比过滤（min_aspect, max_aspect）
       - 密实度过滤（min_solidity）
    3. 区域过滤（调用RegionManager.filter_detections()）
    
    密实度说明：
        - 密实度 = 轮廓面积 / 外接矩形面积
        - 飞虫等松散轮廓 → 密实度低（< 0.3）
        - 抛物实体轮廓 → 密实度高（> 0.3）
    
    Attributes:
        config (dict): 配置字典。
        min_area (int): 最小面积阈值（像素）。
        max_area (int): 最大面积阈值（像素）。
        min_aspect (float): 最小宽高比。
        max_aspect (float): 最大宽高比。
        min_solidity (float): 最小密实度。
        region_manager (Optional[RegionManager]): 区域管理器实例。
        logger (logging.Logger): 日志器实例。
        
    Example:
        >>> config = {"detector": {"min_area": 16, "max_area": 5000}}
        >>> detector = Detector(config, region_manager=None)
        >>> fg_mask = np.zeros((480, 640), dtype=np.uint8)
        >>> fg_mask[100:200, 200:400] = 255
        >>> detections = detector.detect(fg_mask, frame=np.zeros((480, 640, 3)))
        >>> print(f"Detections: {len(detections)}")
    """
    
    def __init__(self, config: dict, region_manager=None) -> None:
        """
        初始化Detector。
        
        Args:
            config: 配置字典，应包含 "detector" 节
            region_manager: RegionManager实例（可选）
            
        Example:
            >>> config = {
            ...     "detector": {
            ...         "min_area": 16,
            ...         "max_area": 5000,
            ...         "min_aspect": 0.1,
            ...         "max_aspect": 10.0,
            ...         "min_solidity": 0.3
            ...     }
            ... }
            >>> detector = Detector(config, region_manager=None)
        """
        self.config = config
        self.logger = setup_logger(__name__, config)
        
        # 从配置中提取检测参数
        det_config = config.get("detector", {})
        self.min_area = det_config.get("min_area", 16)
        self.max_area = det_config.get("max_area", 5000)
        self.min_aspect = det_config.get("min_aspect", 0.1)
        self.max_aspect = det_config.get("max_aspect", 10.0)
        self.min_solidity = det_config.get("min_solidity", 0.3)
        
        # 区域管理器
        self.region_manager = region_manager
        
        self.logger.info(
            f"Detector initialized: "
            f"min_area={self.min_area}, max_area={self.max_area}, "
            f"min_aspect={self.min_aspect}, max_aspect={self.max_aspect}, "
            f"min_solidity={self.min_solidity}"
        )
    
    def detect(self, fg_mask: np.ndarray, frame: np.ndarray) -> List[DetectionResult]:
        """
        检测前景掩膜中的目标。
        
        处理流程：
        1. 轮廓检测
        2. 面积过滤
        3. 宽高比过滤
        4. 密实度过滤
        5. 区域过滤（如果配置了RegionManager）
        
        Args:
            fg_mask: 二值前景掩膜 (0 or 255)，形状 (H, W)
            frame: 原始帧 (BGR, 用于可视化)，形状 (H, W, 3)
            
        Returns:
            List[DetectionResult]: 检测结果列表
            
        Example:
            >>> fg_mask = np.zeros((480, 640), dtype=np.uint8)
            >>> fg_mask[100:200, 200:400] = 255
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> detections = detector.detect(fg_mask, frame)
            >>> len(detections) > 0
            True
        """
        if fg_mask is None or fg_mask.size == 0:
            self.logger.warning("Empty foreground mask")
            return []
        
        if len(fg_mask.shape) != 2:
            self.logger.error(f"Invalid fg_mask shape: {fg_mask.shape}, expected (H, W)")
            return []
        
        # Step1: 轮廓检测
        # RETR_EXTERNAL: 只检测外部轮廓
        # CHAIN_APPROX_SIMPLE: 压缩水平、垂直、对角线方向，只保留端点
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            self.logger.debug("No contours found")
            return []
        
        self.logger.debug(f"Found {len(contours)} contours")
        
        # Step2: 轮廓筛选
        detections = []
        filtered_by_area = 0
        filtered_by_aspect = 0
        filtered_by_solidity = 0
        
        for contour in contours:
            # 面积过滤
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                filtered_by_area += 1
                continue
            
            # 外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 宽高比过滤
            aspect = w / max(h, 1)
            if aspect < self.min_aspect or aspect > self.max_aspect:
                filtered_by_aspect += 1
                continue
            
            # 密实度过滤（轮廓面积/外接矩形面积）
            rect_area = w * h
            solidity = area / max(rect_area, 1)
            if solidity < self.min_solidity:
                filtered_by_solidity += 1
                continue
            
            # 质心计算
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # 如果矩为0，使用外接矩形中心
                cx, cy = x + w // 2, y + h // 2
            
            # 创建检测结果
            detections.append(DetectionResult(
                x=x, y=y, w=w, h=h,
                area=int(area),
                centroid=(cx, cy),
                confidence=1.0
            ))
        
        self.logger.debug(
            f"Contour filtering: total={len(contours)}, "
            f"filtered_by_area={filtered_by_area}, "
            f"filtered_by_aspect={filtered_by_aspect}, "
            f"filtered_by_solidity={filtered_by_solidity}, "
            f"remaining={len(detections)}"
        )
        
        # Step3: 区域过滤
        if self.region_manager and self.region_manager.has_regions():
            before_filter = len(detections)
            detections = self.region_manager.filter_detections(
                detections, frame.shape
            )
            after_filter = len(detections)
            self.logger.debug(
                f"Region filtering: {before_filter} -> {after_filter} detections"
            )
        
        return detections
    
    def set_region_manager(self, region_manager) -> None:
        """
        设置区域管理器。
        
        Args:
            region_manager: RegionManager实例
            
        Example:
            >>> detector.set_region_manager(region_manager)
        """
        self.region_manager = region_manager
        self.logger.debug("Region manager set")
    
    def update_config(self, new_config: dict) -> None:
        """
        更新检测参数。
        
        Args:
            new_config: 新的配置字典
            
        Example:
            >>> new_config = {"detector": {"min_area": 32}}
            >>> detector.update_config(new_config)
        """
        det_config = new_config.get("detector", {})
        
        if "min_area" in det_config:
            self.min_area = det_config["min_area"]
        if "max_area" in det_config:
            self.max_area = det_config["max_area"]
        if "min_aspect" in det_config:
            self.min_aspect = det_config["min_aspect"]
        if "max_aspect" in det_config:
            self.max_aspect = det_config["max_aspect"]
        if "min_solidity" in det_config:
            self.min_solidity = det_config["min_solidity"]
        
        self.logger.info(
            f"Detector config updated: "
            f"min_area={self.min_area}, max_area={self.max_area}, "
            f"min_aspect={self.min_aspect}, max_aspect={self.max_aspect}, "
            f"min_solidity={self.min_solidity}"
        )
