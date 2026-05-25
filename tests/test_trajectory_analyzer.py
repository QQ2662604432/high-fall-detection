"""
TrajectoryAnalyzer 单元测试

测试轨迹分析模块的各项功能：
1. 抛物线拟合正确性
2. 位移分析（X/Y轴分离）
3. 速度/加速度分析
4. 综合判定逻辑
5. 边界条件处理

Author: 寇豆码 (Alex)
Date: 2024
"""

import unittest
import time
import numpy as np
from datetime import datetime
from typing import List, Tuple

from src.tracker import Track
from src.trajectory_analyzer import TrajectoryAnalyzer, TrajectoryResult


class TestTrajectoryAnalyzer(unittest.TestCase):
    """TrajectoryAnalyzer 测试用例。"""
    
    def setUp(self) -> None:
        """测试前准备。"""
        # 最小化配置
        self.config = {
            "trajectory_analyzer": {
                "min_points_for_fit": 5,
                "parabola_a_threshold": 0.01,
                "min_vertical_speed": 2.0,
                "max_horizontal_ratio": 2.0,
                "min_track_length": 5,
                "min_acceleration": 0.1,
                "fit_recent_n": 8
            },
            "logging": {
                "level": "DEBUG",
                "save_to_file": False
            }
        }
        self.analyzer = TrajectoryAnalyzer(self.config)
    
    def _create_parabola_track(self, track_id: int = 1) -> Track:
        """
        创建抛物线轨迹（模拟高空抛物）。
        
        轨迹方程: y = 0.1 * x^2 (a > 0，符合抛物下落)
        转换为像素坐标：x从0到100，y从0到1000
        
        Args:
            track_id: 轨迹ID
            
        Returns:
            Track: 抛物线轨迹对象
        """
        track = Track(track_id=track_id)
        
        # 生成抛物线上的点 (y = 0.1 * x^2)
        for i in range(10):
            x = i * 10  # 0, 10, 20, ..., 90
            y = int(0.1 * x ** 2)  # 0, 10, 40, ..., 810
            bbox = (x - 10, y - 10, 20, 20)
            track.add_point((x, y), time.time(), i, bbox)
        
        return track
    
    def _create_straight_line_track(self, track_id: int = 2) -> Track:
        """
        创建直线轨迹（模拟水平运动，非抛物）。
        
        Args:
            track_id: 轨迹ID
            
        Returns:
            Track: 直线轨迹对象
        """
        track = Track(track_id=track_id)
        
        # 生成直线上的点 (y = 5，恒定高度)
        for i in range(10):
            x = i * 10
            y = 5  # 恒定高度
            bbox = (x - 10, y - 10, 20, 20)
            track.add_point((x, y), time.time(), i, bbox)
        
        return track
    
    def _create_vertical_fall_track(self, track_id: int = 3) -> Track:
        """
        创建垂直下落轨迹（纯垂直运动）。
        
        Args:
            track_id: 轨迹ID
            
        Returns:
            Track: 垂直下落轨迹对象
        """
        track = Track(track_id=track_id)
        
        # 生成垂直下落的点 (x恒定，y增加)
        for i in range(10):
            x = 100  # 恒定x
            y = i * 20  # 匀速下落
            bbox = (x - 10, y - 10, 20, 20)
            track.add_point((x, y), time.time(), i, bbox)
        
        return track
    
    def test_parabola_detection(self) -> None:
        """测试抛物线拟合正确性。"""
        # 创建抛物线轨迹
        track = self._create_parabola_track()
        
        # 分析轨迹
        result = self.analyzer.analyze(track)
        
        # 验证：抛物线系数 a 应该 > 0
        self.assertTrue(
            result.parabola_a > self.config["trajectory_analyzer"]["parabola_a_threshold"],
            f"Expected parabola_a > 0, got {result.parabola_a}"
        )
        self.assertTrue(result.is_parabola, "Should detect parabola")
        
        # 验证置信度
        self.assertGreaterEqual(result.confidence, 0.0, "Confidence should be >= 0")
        self.assertLessEqual(result.confidence, 1.0, "Confidence should be <= 1")
    
    def test_straight_line_not_parabola(self) -> None:
        """测试直线轨迹不被误判为抛物线。"""
        # 创建直线轨迹
        track = self._create_straight_line_track()
        
        # 分析轨迹
        result = self.analyzer.analyze(track)
        
        # 验证：直线不应该被判定为抛物线
        # 注意：由于浮点误差，a可能很小但不为0
        # 我们检查是否小于阈值
        self.assertFalse(result.is_parabola, "Straight line should not be detected as parabola")
    
    def test_vertical_fall_detection(self) -> None:
        """测试垂直下落检测。"""
        # 创建垂直下落轨迹
        track = self._create_vertical_fall_track()
        
        # 分析轨迹
        result = self.analyzer.analyze(track)
        
        # 验证：垂直速度应该 > 0（向下）
        self.assertGreater(result.vertical_speed, 0, "Vertical speed should be > 0 for falling object")
        
        # 验证：方向应该是 "down" 或 "diagonal"
        self.assertIn(result.direction, ["down", "diagonal"], "Direction should be down or diagonal")
    
    def test_insufficient_points(self) -> None:
        """测试点数不足的情况。"""
        # 创建只有3个点的轨迹（少于min_points=5）
        track = Track(track_id=99)
        for i in range(3):
            x = i * 10
            y = i * 20
            bbox = (x - 10, y - 10, 20, 20)
            track.add_point((x, y), time.time(), i, bbox)
        
        # 分析轨迹
        result = self.analyzer.analyze(track)
        
        # 验证：应该返回默认值
        self.assertFalse(result.is_parabola, "Should not detect parabola with insufficient points")
        self.assertEqual(result.confidence, 0.0, "Confidence should be 0 for insufficient points")
        self.assertEqual(result.direction, "unknown", "Direction should be unknown")
    
    def test_is_falling_object(self) -> None:
        """测试最终判定逻辑。"""
        # 创建一个高置信度的抛物轨迹结果
        result = TrajectoryResult(
            track_id=1,
            is_parabola=True,
            parabola_a=0.05,
            parabola_b=0.0,
            parabola_c=0.0,
            vertical_speed=5.0,  # > min_vertical_speed
            horizontal_speed=1.0,
            acceleration=0.2,
            confidence=0.7,  # > 0.6
            direction="down"
        )
        
        # 验证：应该判定为抛物
        self.assertTrue(
            self.analyzer.is_falling_object(result),
            "Should detect falling object with high confidence"
        )
        
        # 创建一个低置信度的结果
        result_low = TrajectoryResult(
            track_id=2,
            is_parabola=False,
            parabola_a=0.0,
            parabola_b=0.0,
            parabola_c=0.0,
            vertical_speed=1.0,  # < min_vertical_speed
            horizontal_speed=0.0,
            acceleration=0.0,
            confidence=0.3,  # < 0.6
            direction="horizontal"
        )
        
        # 验证：不应该判定为抛物
        self.assertFalse(
            self.analyzer.is_falling_object(result_low),
            "Should not detect falling object with low confidence"
        )
    
    def test_direction_detection(self) -> None:
        """测试方向判断逻辑。"""
        # 创建向下轨迹
        track_down = Track(track_id=10)
        for i in range(10):
            x = i * 2  # 小水平位移
            y = i * 10  # 大垂直位移
            bbox = (x - 10, y - 10, 20, 20)
            track_down.add_point((x, y), time.time(), i, bbox)
        
        result_down = self.analyzer.analyze(track_down)
        self.assertEqual(result_down.direction, "down", "Should detect downward direction")
        
        # 创建对角线轨迹
        track_diag = Track(track_id=11)
        for i in range(10):
            x = i * 10  # 大水平位移
            y = i * 10  # 大垂直位移
            bbox = (x - 10, y - 10, 20, 20)
            track_diag.add_point((x, y), time.time(), i, bbox)
        
        result_diag = self.analyzer.analyze(track_diag)
        self.assertEqual(result_diag.direction, "diagonal", "Should detect diagonal direction")
        
        # 创建向上轨迹
        track_up = Track(track_id=12)
        for i in range(10):
            x = i * 10
            y = 1000 - i * 10  # y减小（向上）
            bbox = (x - 10, y - 10, 20, 20)
            track_up.add_point((x, y), time.time(), i, bbox)
        
        result_up = self.analyzer.analyze(track_up)
        self.assertEqual(result_up.direction, "up", "Should detect upward direction")
    
    def test_acceleration_calculation(self) -> None:
        """测试加速度计算。"""
        # 创建加速轨迹（速度递增）
        track = Track(track_id=20)
        
        # 模拟加速运动
        for i in range(10):
            x = i * 10
            # 加速度运动：位移随t^2增长
            y = int(0.5 * (i ** 2))  # 加速度运动
            bbox = (x - 10, y - 10, 20, 20)
            track.add_point((x, y), time.time(), i, bbox)
        
        result = self.analyzer.analyze(track)
        
        # 验证：应该检测到加速度
        self.assertGreater(result.acceleration, 0, "Should detect acceleration")
    
    def test_batch_analyze(self) -> None:
        """测试批量分析功能。"""
        # 创建多个轨迹
        tracks = [
            self._create_parabola_track(track_id=1),
            self._create_vertical_fall_track(track_id=2),
            self._create_straight_line_track(track_id=3)
        ]
        
        # 批量分析
        results = self.analyzer.batch_analyze(tracks)
        
        # 验证：应该返回3个结果
        self.assertEqual(len(results), 3, "Should return 3 results")
        
        # 验证：每个结果都是TrajectoryResult实例
        for result in results:
            self.assertIsInstance(result, TrajectoryResult, "Result should be TrajectoryResult")
    
    def test_empty_track(self) -> None:
        """测试空轨迹处理。"""
        # 创建空轨迹
        track = Track(track_id=99)
        
        # 分析空轨迹
        result = self.analyzer.analyze(track)
        
        # 验证：应该返回默认值
        self.assertEqual(result.track_id, 99, "Track ID should match")
        self.assertEqual(result.confidence, 0.0, "Confidence should be 0 for empty track")
        self.assertEqual(result.direction, "unknown", "Direction should be unknown")
    
    def test_polyfit_exception_handling(self) -> None:
        """测试polyfit异常处理（共线点）。"""
        # 创建共线点（会导致polyfit警告但不会失败）
        track = Track(track_id=30)
        
        # 所有点都在同一位置（会导致条件数警告，但不会失败）
        for i in range(10):
            x = 100
            y = 100
            bbox = (x - 10, y - 10, 20, 20)
            track.add_point((x, y), time.time(), i, bbox)
        
        # 分析轨迹（应该不崩溃）
        result = self.analyzer.analyze(track)
        
        # 验证：应该返回TrajectoryResult（不崩溃）
        self.assertIsInstance(result, TrajectoryResult, "Should return TrajectoryResult even with polyfit warning")
        
        # 验证：由于所有点共线，is_parabola应该为False（a应该很小）
        self.assertFalse(result.is_parabola, "Collinear points should not be detected as parabola")
        self.assertEqual(result.confidence, 0.0, "Confidence should be 0 for collinear points")


if __name__ == "__main__":
    unittest.main()
