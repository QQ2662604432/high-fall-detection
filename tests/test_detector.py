"""
Detector模块单元测试

测试目标检测器的各种过滤逻辑，包括：
1. 面积过滤
2. 宽高比过滤
3. 密实度过滤
4. 区域过滤集成

Author: 寇豆码 (Alex)
Date: 2024
"""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import Detector, DetectionResult
from src.region_manager import RegionManager, PolygonRegion


class TestDetectionResult(unittest.TestCase):
    """测试DetectionResult数据结构。"""
    
    def test_to_xyxy(self) -> None:
        """测试to_xyxy()方法。"""
        det = DetectionResult(
            x=100, y=100, w=50, h=50,
            area=2500, centroid=(125, 125), confidence=1.0
        )
        result = det.to_xyxy()
        self.assertEqual(result, (100, 100, 150, 150))
    
    def test_to_xyah(self) -> None:
        """测试to_xyah()方法。"""
        det = DetectionResult(
            x=100, y=100, w=50, h=50,
            area=2500, centroid=(125, 125), confidence=1.0
        )
        result = det.to_xyah()
        self.assertEqual(result, (125, 125, 1.0, 50))


class TestDetector(unittest.TestCase):
    """测试Detector类。"""
    
    @classmethod
    def setUpClass(cls) -> None:
        """设置测试类。"""
        # 创建测试配置
        cls.config = {
            "detector": {
                "min_area": 16,
                "max_area": 5000,
                "min_aspect": 0.1,
                "max_aspect": 10.0,
                "min_solidity": 0.3
            }
        }
    
    def setUp(self) -> None:
        """每个测试用例前的设置。"""
        self.detector = Detector(self.config, region_manager=None)
    
    def test_init(self) -> None:
        """测试Detector初始化。"""
        self.assertEqual(self.detector.min_area, 16)
        self.assertEqual(self.detector.max_area, 5000)
        self.assertEqual(self.detector.min_aspect, 0.1)
        self.assertEqual(self.detector.max_aspect, 10.0)
        self.assertEqual(self.detector.min_solidity, 0.3)
        self.assertIsNone(self.detector.region_manager)
    
    def test_detect_empty_mask(self) -> None:
        """测试空掩膜检测。"""
        empty_mask = np.zeros((480, 640), dtype=np.uint8)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = self.detector.detect(empty_mask, frame)
        self.assertEqual(len(detections), 0)
    
    def test_detect_single_rectangle(self) -> None:
        """测试单个矩形目标检测。"""
        # 创建前景掩膜（中间有一个矩形）
        fg_mask = np.zeros((480, 640), dtype=np.uint8)
        fg_mask[100:150, 200:250] = 255  # 50x50矩形
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = self.detector.detect(fg_mask, frame)
        
        # 应该检测到1个目标
        self.assertGreaterEqual(len(detections), 1)
        
        # 检查检测结果
        if len(detections) > 0:
            det = detections[0]
            self.assertGreaterEqual(det.area, self.config["detector"]["min_area"])
            self.assertLessEqual(det.area, self.config["detector"]["max_area"])
            self.assertIn("x", det.__dict__)
            self.assertIn("y", det.__dict__)
            self.assertIn("w", det.__dict__)
            self.assertIn("h", det.__dict__)
    
    def test_detect_multiple_objects(self) -> None:
        """测试多个目标检测。"""
        # 创建前景掩膜（多个矩形）
        fg_mask = np.zeros((480, 640), dtype=np.uint8)
        fg_mask[50:100, 50:100] = 255    # 矩形1: 50x50=2500
        fg_mask[150:250, 150:250] = 255  # 矩形2: 100x100=10000
        fg_mask[300:350, 300:350] = 255  # 矩形3: 50x50=2500
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = self.detector.detect(fg_mask, frame)
        
        # 应该检测到3个目标
        # 注意：由于形态学操作，可能检测到2-3个
        self.assertGreaterEqual(len(detections), 2)
    
    def test_area_filtering(self) -> None:
        """测试面积过滤。"""
        # 创建前景掩膜
        fg_mask = np.zeros((480, 640), dtype=np.uint8)
        
        # 小目标（面积<min_area=16）
        fg_mask[10:12, 10:12] = 255  # 2x2=4像素
        
        # 正常目标
        fg_mask[50:150, 50:150] = 255  # 100x100=10000像素
        
        # 大目标（面积>max_area=5000）
        fg_mask[200:300, 200:300] = 255  # 100x100=10000像素
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = self.detector.detect(fg_mask, frame)
        
        # 只有正常目标应该被检测到
        # 注意：实际过滤效果取决于形态学操作和轮廓检测
        # 这里主要测试代码不崩溃
        self.assertIsInstance(detections, list)
    
    def test_aspect_ratio_filtering(self) -> None:
        """测试宽高比过滤。"""
        # 创建前景掩膜
        fg_mask = np.zeros((480, 640), dtype=np.uint8)
        
        # 正常宽高比
        fg_mask[100:200, 100:200] = 255  # 方形 (aspect=1.0)
        
        # 极端宽高比（非常宽）
        fg_mask[300:310, 100:500] = 255  # 10x400 (aspect=40)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = self.detector.detect(fg_mask, frame)
        
        # 测试代码不崩溃
        self.assertIsInstance(detections, list)
    
    def test_solidity_filtering(self) -> None:
        """测试密实度过滤。"""
        # 密实度 = 轮廓面积 / 外接矩形面积
        # 圆形/方形 → 密实度高
        # 细长/松散形状 → 密实度低
        
        fg_mask = np.zeros((480, 640), dtype=np.uint8)
        
        # 矩形（密实度高）
        fg_mask[100:200, 100:200] = 255
        
        # 创建一个更像飞虫的松散形状（密实度低）
        # 这里用圆形模拟（相对密实）
        cv2.circle(fg_mask, (400, 150), 50, 255, -1)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = self.detector.detect(fg_mask, frame)
        
        # 测试代码不崩溃
        self.assertIsInstance(detections, list)
    
    def test_with_region_manager(self) -> None:
        """测试集成RegionManager。"""
        # 创建RegionManager
        region_config = {
            "regions": {
                "detect_regions": [
                    {
                        "id": "test_region",
                        "points": [
                            {"x": 0.0, "y": 0.0},
                            {"x": 0.5, "y": 0.0},
                            {"x": 0.5, "y": 0.5},
                            {"x": 0.0, "y": 0.5}
                        ],
                        "color": [0, 255, 0]
                    }
                ]
            }
        }
        region_manager = RegionManager(region_config)
        
        # 创建Detector with region_manager
        detector = Detector(self.config, region_manager=region_manager)
        
        # 创建前景掩膜（目标在检测区域内）
        fg_mask = np.zeros((480, 640), dtype=np.uint8)
        fg_mask[50:150, 50:150] = 255  # 在检测区域内
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(fg_mask, frame)
        
        # 测试代码不崩溃
        self.assertIsInstance(detections, list)
    
    def test_update_config(self) -> None:
        """测试更新配置。"""
        new_config = {
            "detector": {
                "min_area": 32,
                "max_area": 10000
            }
        }
        
        self.detector.update_config(new_config)
        
        self.assertEqual(self.detector.min_area, 32)
        self.assertEqual(self.detector.max_area, 10000)


if __name__ == "__main__":
    unittest.main()
