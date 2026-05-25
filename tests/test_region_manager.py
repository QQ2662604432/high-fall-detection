"""
RegionManager模块单元测试

测试区域判断和掩膜生成功能，包括：
1. 多边形区域定义
2. 点-in-区域判断
3. 掩膜预计算
4. 区域过滤逻辑

Author: 寇豆码 (Alex)
Date: 2024
"""

import unittest
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.region_manager import RegionManager, PolygonRegion


class TestPolygonRegion(unittest.TestCase):
    """测试PolygonRegion数据结构。"""
    
    def test_init(self) -> None:
        """测试PolygonRegion初始化。"""
        points = [(0.1, 0.2), (0.3, 0.2), (0.3, 0.8), (0.1, 0.8)]
        region = PolygonRegion(
            region_id="test_region",
            region_type="detect",
            points=points,
            color=(0, 255, 0)
        )
        
        self.assertEqual(region.region_id, "test_region")
        self.assertEqual(region.region_type, "detect")
        self.assertEqual(len(region.points), 4)
        self.assertEqual(region.color, (0, 255, 0))
    
    def test_contains_inside(self) -> None:
        """测试点在多边形内。"""
        # 创建一个矩形区域（归一化坐标）
        points = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]
        region = PolygonRegion(
            region_id="r1",
            region_type="detect",
            points=points,
            color=(0, 255, 0)
        )
        
        # 点在区域内 (归一化坐标 (0.3, 0.3) -> 像素坐标 (192, 144) for 640x480)
        point = (192, 144)  # 在区域内
        frame_shape = (480, 640, 3)
        
        result = region.contains(point, frame_shape)
        self.assertTrue(result)
    
    def test_contains_outside(self) -> None:
        """测试点在多边形外。"""
        # 创建一个矩形区域
        points = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]
        region = PolygonRegion(
            region_id="r1",
            region_type="detect",
            points=points,
            color=(0, 255, 0)
        )
        
        # 点在区域外
        point = (500, 300)  # 在区域外
        frame_shape = (480, 640, 3)
        
        result = region.contains(point, frame_shape)
        self.assertFalse(result)
    
    def test_contains_boundary(self) -> None:
        """测试点在多边形边界上。"""
        # 创建一个矩形区域
        points = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]
        region = PolygonRegion(
            region_id="r1",
            region_type="detect",
            points=points,
            color=(0, 255, 0)
        )
        
        # 点在边界上 (0.1, 0.3) -> (64, 144)
        point = (64, 144)
        frame_shape = (480, 640, 3)
        
        result = region.contains(point, frame_shape)
        self.assertTrue(result)  # pointPolygonTest返回0表示在边界上


class TestRegionManager(unittest.TestCase):
    """测试RegionManager类。"""
    
    @classmethod
    def setUpClass(cls) -> None:
        """设置测试类。"""
        # 创建测试配置
        cls.config = {
            "regions": {
                "detect_regions": [
                    {
                        "id": "region_1",
                        "points": [
                            {"x": 0.1, "y": 0.1},
                            {"x": 0.5, "y": 0.1},
                            {"x": 0.5, "y": 0.5},
                            {"x": 0.1, "y": 0.5}
                        ],
                        "color": [0, 255, 0]
                    }
                ],
                "shield_regions": [
                    {
                        "id": "shield_1",
                        "points": [
                            {"x": 0.6, "y": 0.6},
                            {"x": 0.9, "y": 0.6},
                            {"x": 0.9, "y": 0.9},
                            {"x": 0.6, "y": 0.9}
                        ],
                        "color": [0, 0, 255]
                    }
                ]
            }
        }
    
    def setUp(self) -> None:
        """每个测试用例前的设置。"""
        self.region_manager = RegionManager(self.config)
    
    def test_init(self) -> None:
        """测试RegionManager初始化。"""
        self.assertEqual(len(self.region_manager.detect_regions), 1)
        self.assertEqual(len(self.region_manager.shield_regions), 1)
        self.assertTrue(self.region_manager.has_regions())
    
    def test_has_regions(self) -> None:
        """测试has_regions()方法。"""
        # 有区域
        self.assertTrue(self.region_manager.has_regions())
        
        # 无区域
        empty_config = {"regions": {}}
        rm_empty = RegionManager(empty_config)
        self.assertFalse(rm_empty.has_regions())
    
    def test_build_masks(self) -> None:
        """测试掩膜构建。"""
        frame_shape = (480, 640, 3)
        
        # 构建掩膜
        self.region_manager._build_masks(frame_shape)
        
        # 检查掩膜是否创建
        self.assertIsNotNone(self.region_manager._region_mask_detect)
        self.assertIsNotNone(self.region_manager._region_mask_shield)
        
        # 检查掩膜形状
        self.assertEqual(self.region_manager._region_mask_detect.shape, (480, 640))
        self.assertEqual(self.region_manager._region_mask_shield.shape, (480, 640))
    
    def test_is_in_detect_region_inside(self) -> None:
        """测试点在检测区域内。"""
        frame_shape = (480, 640, 3)
        
        # 点在检测区域内 (0.3, 0.3) -> (192, 144)
        point = (192, 144)
        
        result = self.region_manager.is_in_detect_region(point, frame_shape)
        self.assertTrue(result)
    
    def test_is_in_detect_region_outside(self) -> None:
        """测试点在检测区域外。"""
        frame_shape = (480, 640, 3)
        
        # 点在检测区域外 (0.6, 0.6) -> (384, 288)
        point = (384, 288)
        
        result = self.region_manager.is_in_detect_region(point, frame_shape)
        self.assertFalse(result)
    
    def test_is_in_detect_region_no_limit(self) -> None:
        """测试无检测区域限制（全画面检测）。"""
        # 创建无检测区域的配置
        config_no_detect = {
            "regions": {
                "shield_regions": []
            }
        }
        rm = RegionManager(config_no_detect)
        
        frame_shape = (480, 640, 3)
        point = (320, 240)  # 任意点
        
        # 无检测区域限制 → 返回True（全画面检测）
        result = rm.is_in_detect_region(point, frame_shape)
        self.assertTrue(result)
    
    def test_is_in_shield_region_inside(self) -> None:
        """测试点在屏蔽区域内。"""
        frame_shape = (480, 640, 3)
        
        # 点在屏蔽区域内 (0.7, 0.7) -> (448, 336)
        point = (448, 336)
        
        result = self.region_manager.is_in_shield_region(point, frame_shape)
        self.assertTrue(result)
    
    def test_is_in_shield_region_outside(self) -> None:
        """测试点在屏蔽区域外。"""
        frame_shape = (480, 640, 3)
        
        # 点在屏蔽区域外 (0.3, 0.3) -> (192, 144)
        point = (192, 144)
        
        result = self.region_manager.is_in_shield_region(point, frame_shape)
        self.assertFalse(result)
    
    def test_filter_detections(self) -> None:
        """测试区域过滤逻辑。"""
        from src.detector import DetectionResult
        
        frame_shape = (480, 640, 3)
        
        # 创建检测结果
        # 检测区域: (0.1,0.1) to (0.5,0.5) -> 像素 (64,48) to (320,240)
        # 屏蔽区域: (0.6,0.6) to (0.9,0.9) -> 像素 (384,288) to (576,432)
        
        detections = [
            # 在检测区域内，不在屏蔽区域内 → 应保留
            DetectionResult(x=100, y=100, w=50, h=50, area=2500, centroid=(125, 125)),
            # 在检测区域外 → 应过滤
            DetectionResult(x=400, y=400, w=50, h=50, area=2500, centroid=(425, 425)),
            # 在屏蔽区域内 → 应过滤 (注意：y=325在480范围内)
            DetectionResult(x=450, y=325, w=50, h=50, area=2500, centroid=(475, 350))
        ]
        
        filtered = self.region_manager.filter_detections(detections, frame_shape)
        
        # 只有第一个应该保留
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].centroid, (125, 125))
    
    def test_draw_regions(self) -> None:
        """测试区域绘制。"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 绘制区域
        frame_with_regions = self.region_manager.draw_regions(frame)
        
        # 检查返回值
        self.assertIsNotNone(frame_with_regions)
        self.assertEqual(frame_with_regions.shape, (480, 640, 3))
    
    def test_add_detect_region(self) -> None:
        """测试添加检测区域。"""
        initial_count = len(self.region_manager.detect_regions)
        
        # 创建新区域
        new_region = PolygonRegion(
            region_id="new_region",
            region_type="detect",
            points=[(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)],
            color=(0, 255, 0)
        )
        
        self.region_manager.add_detect_region(new_region)
        
        self.assertEqual(len(self.region_manager.detect_regions), initial_count + 1)
        self.assertIsNone(self.region_manager._region_mask_detect)  # 应标记为需要重建
    
    def test_add_shield_region(self) -> None:
        """测试添加屏蔽区域。"""
        initial_count = len(self.region_manager.shield_regions)
        
        # 创建新区域
        new_region = PolygonRegion(
            region_id="new_shield",
            region_type="shield",
            points=[(0.1, 0.6), (0.4, 0.6), (0.4, 0.9), (0.1, 0.9)],
            color=(0, 0, 255)
        )
        
        self.region_manager.add_shield_region(new_region)
        
        self.assertEqual(len(self.region_manager.shield_regions), initial_count + 1)
        self.assertIsNone(self.region_manager._region_mask_shield)  # 应标记为需要重建
    
    def test_clear_regions(self) -> None:
        """测试清除所有区域。"""
        # 确认有区域
        self.assertTrue(self.region_manager.has_regions())
        
        # 清除区域
        self.region_manager.clear_regions()
        
        # 确认已清除
        self.assertFalse(self.region_manager.has_regions())
        self.assertEqual(len(self.region_manager.detect_regions), 0)
        self.assertEqual(len(self.region_manager.shield_regions), 0)
    
    def test_mask_rebuild_on_frame_size_change(self) -> None:
        """测试帧尺寸变化时重建掩膜。"""
        # 初始构建掩膜
        frame_shape1 = (480, 640, 3)
        self.region_manager._build_masks(frame_shape1)
        mask1 = self.region_manager._region_mask_detect.copy()
        
        # 改变帧尺寸
        frame_shape2 = (720, 1280, 3)
        self.region_manager.is_in_detect_region((100, 100), frame_shape2)
        
        # 掩膜应该被重建
        mask2 = self.region_manager._region_mask_detect
        
        # 检查掩膜形状是否改变
        self.assertEqual(mask2.shape, (720, 1280))
        self.assertNotEqual(mask1.shape, mask2.shape)


if __name__ == "__main__":
    unittest.main()
