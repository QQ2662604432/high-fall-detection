"""
区域管理模块

管理多边形检测区域和屏蔽区域，支持：
1. 多边形区域定义（归一化坐标0~1）
2. 区域掩膜预计算（加速判断）
3. 区域过滤逻辑（质心在检测区域内 且 不在屏蔽区域内）
4. 区域可视化绘制（半透明叠加+边界线）

Author: 寇豆码 (Alex)
Date: 2024
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2

from src.utils import setup_logger


@dataclass
class PolygonRegion:
    """
    多边形区域数据结构。
    
    用于定义检测区域和屏蔽区域。
    
    Attributes:
        region_id: 区域ID（唯一标识）
        region_type: 区域类型（"detect" 或 "shield"）
        points: 多边形顶点（归一化坐标0~1）
        color: 显示颜色 (BGR格式)
        
    Example:
        >>> region = PolygonRegion(
        ...     region_id="region_1",
        ...     region_type="detect",
        ...     points=[(0.1, 0.2), (0.3, 0.2), (0.3, 0.8), (0.1, 0.8)],
        ...     color=(0, 255, 0)
        ... )
        >>> print(region.region_id)
        'region_1'
    """
    
    region_id: str                       # 区域ID
    region_type: str                     # "detect" 或 "shield"
    points: List[Tuple[float, float]]   # 多边形顶点（归一化坐标0~1）
    color: Tuple[int, int, int]         # 显示颜色 (BGR)
    
    def contains(self, point: Tuple[int, int], frame_shape: Tuple) -> bool:
        """
        判断点是否在多边形内。
        
        使用cv2.pointPolygonTest进行精确判断。
        
        Args:
            point: 像素坐标 (x, y)
            frame_shape: 帧形状 (h, w, c)
            
        Returns:
            bool: 是否在区域内
            
        Example:
            >>> region = PolygonRegion(
            ...     region_id="r1",
            ...     region_type="detect",
            ...     points=[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)],
            ...     color=(0, 255, 0)
            ... )
            >>> region.contains((100, 100), (480, 640, 3))
            True
        """
        if not frame_shape or len(frame_shape) < 2:
            self.logger.error("Invalid frame_shape")
            return False
        
        h, w = frame_shape[:2]
        
        # 将归一化坐标转换为像素坐标
        pts = np.array(
            [[int(p[0] * w), int(p[1] * h)] for p in self.points],
            dtype=np.int32
        )
        
        # 使用cv2.pointPolygonTest判断点是否在多边形内
        # 返回值：
        #   > 0: 点在多边形内
        #   = 0: 点在多边形边界上
        #   < 0: 点在多边形外
        result = cv2.pointPolygonTest(pts, point, False)
        return result >= 0


class RegionManager:
    """
    检测/屏蔽区域管理器。
    
    功能：
    1. 从配置中解析检测区域和屏蔽区域
    2. 预计算区域掩膜（加速后续点-in-区域判断）
    3. 过滤检测结果（保留在检测区域内且不在屏蔽区域内的目标）
    4. 可视化区域（半透明叠加+边界线）
    
    预计算掩膜的优势：
        - 每次判断点是否在区域内，无需遍历多边形顶点
        - 直接查表：mask[y, x] > 0 → O(1) 时间复杂度
        - 特别适合每一帧需要判断大量检测框的场景
    
    Attributes:
        detect_regions (List[PolygonRegion]): 检测区域列表
        shield_regions (List[PolygonRegion]): 屏蔽区域列表
        _region_mask_detect (Optional[np.ndarray]): 检测区域掩膜（预计算）
        _region_mask_shield (Optional[np.ndarray]): 屏蔽区域掩膜（预计算）
        _frame_size (Optional[Tuple[int, int]]): 帧尺寸缓存
        logger (logging.Logger): 日志器实例
        
    Example:
        >>> config = {
        ...     "regions": {
        ...         "detect_regions": [
        ...             {"id": "r1", "points": [{"x": 0.1, "y": 0.1}, ...], "color": [0, 255, 0]}
        ...         ]
        ...     }
        ... }
        >>> rm = RegionManager(config)
        >>> point = (100, 100)
        >>> in_region = rm.is_in_detect_region(point, (480, 640, 3))
    """
    
    def __init__(self, config: dict) -> None:
        """
        初始化RegionManager。
        
        Args:
            config: 配置字典，应包含 "regions" 节
            
        Example:
            >>> config = {"regions": {"detect_regions": [...], "shield_regions": [...]}}
            >>> rm = RegionManager(config)
        """
        self.logger = setup_logger(__name__, config)
        
        # 区域列表
        self.detect_regions: List[PolygonRegion] = []
        self.shield_regions: List[PolygonRegion] = []
        
        # 预计算掩膜
        self._region_mask_detect: Optional[np.ndarray] = None
        self._region_mask_shield: Optional[np.ndarray] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        
        # 解析区域配置
        regions_config = config.get("regions", {})
        
        # 解析检测区域
        for region_cfg in regions_config.get("detect_regions", []):
            self.detect_regions.append(self._parse_polygon(region_cfg, "detect"))
            self.logger.info(f"Loaded detect region: {region_cfg.get('id', 'unknown')}")
        
        # 解析屏蔽区域
        for region_cfg in regions_config.get("shield_regions", []):
            self.shield_regions.append(self._parse_polygon(region_cfg, "shield"))
            self.logger.info(f"Loaded shield region: {region_cfg.get('id', 'unknown')}")
        
        self.logger.info(
            f"RegionManager initialized: "
            f"detect_regions={len(self.detect_regions)}, "
            f"shield_regions={len(self.shield_regions)}"
        )
    
    def _parse_polygon(self, cfg: dict, region_type: str) -> PolygonRegion:
        """
        解析多边形配置。
        
        Args:
            cfg: 区域配置字典
            region_type: 区域类型（"detect" 或 "shield"）
            
        Returns:
            PolygonRegion: 解析后的多边形区域对象
            
        Raises:
            KeyError: 如果配置缺少必要字段
            
        Example:
            >>> cfg = {
            ...     "id": "r1",
            ...     "points": [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.1}],
            ...     "color": [0, 255, 0]
            ... }
            >>> region = rm._parse_polygon(cfg, "detect")
        """
        region_id = cfg.get("id", f"region_{len(self.detect_regions) + len(self.shield_regions)}")
        points = [(p["x"], p["y"]) for p in cfg.get("points", [])]
        
        # 默认颜色：检测区域绿色，屏蔽区域红色
        default_color = (0, 255, 0) if region_type == "detect" else (0, 0, 255)
        color = tuple(cfg.get("color", list(default_color)))
        
        return PolygonRegion(
            region_id=region_id,
            region_type=region_type,
            points=points,
            color=color
        )
    
    def _build_masks(self, shape: Tuple[int, int, int]) -> None:
        """
        预计算区域掩膜（加速后续判断）。
        
        为检测区域和屏蔽区域分别创建二值掩膜：
        - 检测区域掩膜：白色（255）= 检测区域
        - 屏蔽区域掩膜：白色（255）= 屏蔽区域
        
        Args:
            shape: 帧形状 (h, w, c)
            
        Example:
            >>> rm._build_masks((480, 640, 3))
            >>> rm._region_mask_detect.shape
            (480, 640)
        """
        h, w = shape[:2]
        self._frame_size = (h, w)
        
        self.logger.debug(f"Building region masks for frame size: {w}x{h}")
        
        # 检测区域掩膜: 白色=检测区域
        if self.detect_regions:
            self._region_mask_detect = np.zeros((h, w), dtype=np.uint8)
            for region in self.detect_regions:
                pts = np.array(
                    [[int(p[0]*w), int(p[1]*h)] for p in region.points],
                    dtype=np.int32
                )
                cv2.fillPoly(self._region_mask_detect, [pts], 255)
            
            self.logger.debug("Detect region mask built")
        else:
            self._region_mask_detect = None
        
        # 屏蔽区域掩膜: 白色=屏蔽区域
        if self.shield_regions:
            self._region_mask_shield = np.zeros((h, w), dtype=np.uint8)
            for region in self.shield_regions:
                pts = np.array(
                    [[int(p[0]*w), int(p[1]*h)] for p in region.points],
                    dtype=np.int32
                )
                cv2.fillPoly(self._region_mask_shield, [pts], 255)
            
            self.logger.debug("Shield region mask built")
        else:
            self._region_mask_shield = None
    
    def has_regions(self) -> bool:
        """
        判断是否配置了区域。
        
        Returns:
            bool: 如果配置了检测区域或屏蔽区域，返回True
            
        Example:
            >>> rm = RegionManager({"regions": {}})
            >>> rm.has_regions()
            False
        """
        return len(self.detect_regions) > 0 or len(self.shield_regions) > 0
    
    def is_in_detect_region(self, point: Tuple[int, int], frame_shape: Tuple) -> bool:
        """
        判断点是否在检测区域内。
        
        判断逻辑：
        - 未配置检测区域 → 返回True（全画面检测）
        - 配置了检测区域 → 点在任一检测区域内返回True
        
        Args:
            point: 像素坐标 (x, y)
            frame_shape: 帧形状 (h, w, c)
            
        Returns:
            bool: 是否在检测区域内
            
        Example:
            >>> rm = RegionManager(config_with_detect_region)
            >>> rm.is_in_detect_region((100, 100), (480, 640, 3))
            True
        """
        # 如果帧尺寸变化，重新构建掩膜
        if self._frame_size != frame_shape[:2]:
            self._build_masks(frame_shape)
        
        # 无检测区域限制 → 全画面检测
        if self._region_mask_detect is None:
            return True
        
        x, y = point
        return bool(self._region_mask_detect[y, x] > 0)
    
    def is_in_shield_region(self, point: Tuple[int, int], frame_shape: Tuple) -> bool:
        """
        判断点是否在屏蔽区域内。
        
        Args:
            point: 像素坐标 (x, y)
            frame_shape: 帧形状 (h, w, c)
            
        Returns:
            bool: 是否在屏蔽区域内
            
        Example:
            >>> rm = RegionManager(config_with_shield_region)
            >>> rm.is_in_shield_region((100, 100), (480, 640, 3))
            False
        """
        # 如果帧尺寸变化，重新构建掩膜
        if self._frame_size != frame_shape[:2]:
            self._build_masks(frame_shape)
        
        # 无屏蔽区域 → 返回False
        if self._region_mask_shield is None:
            return False
        
        x, y = point
        return bool(self._region_mask_shield[y, x] > 0)
    
    def filter_detections(self, detections: List, 
                          frame_shape: Tuple) -> List:
        """
        过滤检测结果。
        
        过滤逻辑：
        - 保留：质心在检测区域内 且 不在屏蔽区域内
        - 移除：质心不在检测区域内 或 在屏蔽区域内
        
        Args:
            detections: 检测结果列表（DetectionResult对象）
            frame_shape: 帧形状 (h, w, c)
            
        Returns:
            List: 过滤后的检测结果列表
            
        Example:
            >>> detections = [DetectionResult(...), ...]
            >>> filtered = rm.filter_detections(detections, (480, 640, 3))
        """
        # 如果帧尺寸变化，重新构建掩膜
        if self._frame_size != frame_shape[:2]:
            self._build_masks(frame_shape)
        
        filtered = []
        for det in detections:
            # 质心在检测区域内？
            in_detect = self.is_in_detect_region(det.centroid, frame_shape)
            
            # 质心不在屏蔽区域内？
            not_in_shield = not self.is_in_shield_region(det.centroid, frame_shape)
            
            if in_detect and not_in_shield:
                filtered.append(det)
        
        self.logger.debug(
            f"Region filter: {len(detections)} -> {len(filtered)} detections"
        )
        
        return filtered
    
    def draw_regions(self, frame: np.ndarray) -> np.ndarray:
        """
        在帧上绘制区域边界（可视化用）。
        
        绘制内容：
        - 检测区域：绿色半透明填充 + 绿色边界线
        - 屏蔽区域：红色半透明填充 + 红色边界线
        
        Args:
            frame: 输入帧 (BGR)
            
        Returns:
            np.ndarray: 绘制后的帧
            
        Example:
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> frame_with_regions = rm.draw_regions(frame)
        """
        if frame is None or frame.size == 0:
            self.logger.error("Invalid frame for drawing regions")
            return frame
        
        # 如果帧尺寸变化，重新构建掩膜
        if self._frame_size != frame.shape[:2]:
            self._build_masks(frame.shape)
        
        overlay = frame.copy()
        
        # 绘制检测区域（绿色半透明）
        for region in self.detect_regions:
            pts = np.array(
                [[int(p[0]*frame.shape[1]), int(p[1]*frame.shape[0])]
                 for p in region.points],
                dtype=np.int32
            )
            # 半透明填充
            cv2.fillPoly(overlay, [pts], region.color)
            # 边界线
            cv2.polylines(overlay, [pts], True, region.color, 2)
        
        # 绘制屏蔽区域（红色半透明）
        for region in self.shield_regions:
            pts = np.array(
                [[int(p[0]*frame.shape[1]), int(p[1]*frame.shape[0])]
                 for p in region.points],
                dtype=np.int32
            )
            # 半透明填充
            cv2.fillPoly(overlay, [pts], region.color)
            # 边界线
            cv2.polylines(overlay, [pts], True, region.color, 2)
        
        # 半透明叠加
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame
    
    def add_detect_region(self, region: PolygonRegion) -> None:
        """
        添加检测区域。
        
        Args:
            region: PolygonRegion对象
            
        Example:
            >>> region = PolygonRegion(...)
            >>> rm.add_detect_region(region)
        """
        self.detect_regions.append(region)
        self._region_mask_detect = None  # 标记需要重建掩膜
        self.logger.info(f"Added detect region: {region.region_id}")
    
    def add_shield_region(self, region: PolygonRegion) -> None:
        """
        添加屏蔽区域。
        
        Args:
            region: PolygonRegion对象
            
        Example:
            >>> region = PolygonRegion(...)
            >>> rm.add_shield_region(region)
        """
        self.shield_regions.append(region)
        self._region_mask_shield = None  # 标记需要重建掩膜
        self.logger.info(f"Added shield region: {region.region_id}")
    
    def clear_regions(self) -> None:
        """
        清除所有区域。
        
        Example:
            >>> rm.clear_regions()
        """
        self.detect_regions.clear()
        self.shield_regions.clear()
        self._region_mask_detect = None
        self._region_mask_shield = None
        self.logger.info("All regions cleared")
    
    def __repr__(self) -> str:
        """返回RegionManager的字符串表示。"""
        return (
            f"RegionManager("
            f"detect_regions={len(self.detect_regions)}, "
            f"shield_regions={len(self.shield_regions)})"
        )
