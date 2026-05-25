"""
Tracker模块单元测试

测试SORT集成和Track管理功能，包括：
1. SORT算法集成
2. Track对象生命周期管理
3. 目标跟踪流程
4. 轨迹过滤逻辑

Author: 寇豆码 (Alex)
Date: 2024
"""

import unittest
import numpy as np
import time
import sys
from pathlib import Path
from typing import List

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker import Tracker, Track
from src.detector import DetectionResult


class TestTrack(unittest.TestCase):
    """测试Track数据结构。"""
    
    def test_init(self) -> None:
        """测试Track初始化。"""
        track = Track(track_id=1)
        
        self.assertEqual(track.track_id, 1)
        self.assertEqual(len(track.points), 0)
        self.assertEqual(len(track.timestamps), 0)
        self.assertEqual(len(track.frame_ids), 0)
        self.assertEqual(len(track.bbox_history), 0)
        self.assertEqual(len(track.velocities), 0)
        self.assertEqual(track.age, 0)
        self.assertEqual(track.hits, 0)
        self.assertFalse(track.is_confirmed)
    
    def test_add_point(self) -> None:
        """测试添加轨迹点。"""
        track = Track(track_id=1)
        
        # 添加第一个点
        point = (100, 100)
        timestamp = time.time()
        frame_id = 1
        bbox = (90, 90, 110, 110)
        
        track.add_point(point, timestamp, frame_id, bbox)
        
        self.assertEqual(len(track.points), 1)
        self.assertEqual(track.points[0], point)
        self.assertEqual(len(track.timestamps), 1)
        self.assertEqual(len(track.frame_ids), 1)
        self.assertEqual(len(track.bbox_history), 1)
        self.assertEqual(track.bbox_history[0], bbox)
        self.assertEqual(len(track.velocities), 0)  # 第一个点没有速度
        self.assertEqual(track.hits, 1)
        self.assertEqual(track.age, 0)
        
        # 添加第二个点
        point2 = (110, 110)
        timestamp2 = time.time()
        frame_id2 = 2
        bbox2 = (100, 100, 120, 120)
        
        track.add_point(point2, timestamp2, frame_id2, bbox2)
        
        self.assertEqual(len(track.points), 2)
        self.assertEqual(track.points[1], point2)
        self.assertEqual(len(track.velocities), 1)  # 现在应该有速度了
        self.assertGreater(track.velocities[0], 0.0)  # 速度应该大于0
        self.assertEqual(track.hits, 2)
    
    def test_get_recent_points(self) -> None:
        """测试获取最近轨迹点。"""
        track = Track(track_id=1)
        
        # 添加5个点
        for i in range(5):
            track.add_point(
                (100 + i * 10, 100 + i * 10),
                time.time(),
                i + 1,
                (90 + i * 10, 90 + i * 10, 110 + i * 10, 110 + i * 10)
            )
        
        # 获取最近3个点
        recent = track.get_recent_points(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0], (120, 120))  # 第3个点
        self.assertEqual(recent[2], (140, 140))  # 第5个点
        
        # 获取超过实际数量的点
        all_points = track.get_recent_points(10)
        self.assertEqual(len(all_points), 5)  # 应该返回所有5个点
    
    def test_get_displacement(self) -> None:
        """测试获取总位移。"""
        track = Track(track_id=1)
        
        # 添加点
        track.add_point((100, 100), time.time(), 1, (90, 90, 110, 110))
        track.add_point((200, 300), time.time(), 2, (190, 290, 210, 310))
        
        dx, dy = track.get_displacement()
        self.assertEqual(dx, 100.0)  # 200 - 100
        self.assertEqual(dy, 200.0)  # 300 - 100
        
        # 只有一个点
        track2 = Track(track_id=2)
        track2.add_point((100, 100), time.time(), 1, (90, 90, 110, 110))
        dx2, dy2 = track2.get_displacement()
        self.assertEqual(dx2, 0.0)
        self.assertEqual(dy2, 0.0)
    
    def test_get_average_velocity(self) -> None:
        """测试获取平均速度。"""
        track = Track(track_id=1)
        
        # 添加多个点
        for i in range(5):
            track.add_point(
                (100 + i * 10, 100 + i * 10),
                time.time(),
                i + 1,
                (90 + i * 10, 90 + i * 10, 110 + i * 10, 110 + i * 10)
            )
        
        avg_vel = track.get_average_velocity(last_n=3)
        self.assertGreater(avg_vel, 0.0)  # 平均速度应该大于0
        
        # 没有速度数据
        track2 = Track(track_id=2)
        avg_vel2 = track2.get_average_velocity()
        self.assertEqual(avg_vel2, 0.0)
    
    def test_to_dict_and_from_dict(self) -> None:
        """测试序列化和反序列化。"""
        track = Track(track_id=1)
        track.add_point((100, 100), time.time(), 1, (90, 90, 110, 110))
        track.add_point((110, 110), time.time(), 2, (100, 100, 120, 120))
        
        # 转换为字典
        track_dict = track.to_dict()
        self.assertIn('track_id', track_dict)
        self.assertIn('points', track_dict)
        self.assertEqual(track_dict['track_id'], 1)
        
        # 从字典恢复
        track_restored = Track.from_dict(track_dict)
        self.assertEqual(track_restored.track_id, track.track_id)
        self.assertEqual(track_restored.points, track.points)
        self.assertEqual(track_restored.hits, track.hits)


class TestTracker(unittest.TestCase):
    """测试Tracker类。"""
    
    @classmethod
    def setUpClass(cls) -> None:
        """设置测试类。"""
        # 创建测试配置
        cls.config = {
            "tracker": {
                "max_age": 3,
                "min_hits": 5,
                "iou_threshold": 0.1
            }
        }
    
    def setUp(self) -> None:
        """每个测试用例前的设置。"""
        self.tracker = Tracker(self.config)
    
    def test_init(self) -> None:
        """测试Tracker初始化。"""
        self.assertEqual(self.tracker.max_age, 3)
        self.assertEqual(self.tracker.min_hits, 5)
        self.assertEqual(self.tracker.iou_threshold, 0.1)
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(self.tracker.frame_id, 0)
        self.assertIsNotNone(self.tracker.sort)
    
    def test_update_with_detections(self) -> None:
        """测试使用检测结果更新跟踪器。"""
        # 创建检测结果
        detections = [
            DetectionResult(
                x=100, y=100, w=50, h=50,
                area=2500, centroid=(125, 125), confidence=1.0
            )
        ]
        
        # 第一帧更新
        active_tracks = self.tracker.update(detections)
        
        # 第一帧不应该有确认的轨迹（需要min_hits=5帧）
        self.assertEqual(len(active_tracks), 0)
        self.assertEqual(len(self.tracker.tracks), 1)  # 但应该有1个track
        
        # 继续更新多帧（超过min_hits=5）
        for i in range(6):
            x_offset = i * 10  # 模拟移动
            detections = [
                DetectionResult(
                    x=100 + x_offset, y=100 + x_offset, w=50, h=50,
                    area=2500, centroid=(125 + x_offset, 125 + x_offset),
                    confidence=1.0
                )
            ]
            active_tracks = self.tracker.update(detections)
        
        # 现在应该有一个确认的轨迹
        self.assertGreaterEqual(len(active_tracks), 1)
        self.assertTrue(active_tracks[0].is_confirmed)
    
    def test_update_without_detections(self) -> None:
        """测试没有检测结果时更新跟踪器。"""
        # 创建一个轨迹
        detections = [
            DetectionResult(
                x=100, y=100, w=50, h=50,
                area=2500, centroid=(125, 125), confidence=1.0
            )
        ]
        self.tracker.update(detections)
        
        # 现在没有检测结果
        empty_detections = []
        active_tracks = self.tracker.update(empty_detections)
        
        # 应该没有活跃的确认轨迹
        self.assertEqual(len(active_tracks), 0)
        
        # 但track应该还在（age应该增加）
        self.assertEqual(len(self.tracker.tracks), 1)
        track_id = list(self.tracker.tracks.keys())[0]
        self.assertGreater(self.tracker.tracks[track_id].age, 0)
    
    def test_get_track(self) -> None:
        """测试获取指定ID的Track对象。"""
        # 创建检测结果并更新多次以创建track
        for i in range(6):
            x_offset = i * 10
            detections = [
                DetectionResult(
                    x=100 + x_offset, y=100 + x_offset, w=50, h=50,
                    area=2500, centroid=(125 + x_offset, 125 + x_offset),
                    confidence=1.0
                )
            ]
            self.tracker.update(detections)
        
        # 获取track
        track = self.tracker.get_track(0)  # SORT默认从0开始
        
        self.assertIsNotNone(track)
        self.assertEqual(track.track_id, 0)
        
        # 获取不存在的track
        non_existent = self.tracker.get_track(999)
        self.assertIsNone(non_existent)
    
    def test_get_all_tracks(self) -> None:
        """测试获取所有Track对象。"""
        # 创建两个检测结果并更新多次
        for i in range(6):
            x_offset = i * 10
            detections1 = [
                DetectionResult(
                    x=100 + x_offset, y=100 + x_offset, w=50, h=50,
                    area=2500, centroid=(125 + x_offset, 125 + x_offset),
                    confidence=1.0
                )
            ]
            self.tracker.update(detections1)
        
        all_tracks = self.tracker.get_all_tracks()
        self.assertGreaterEqual(len(all_tracks), 1)
    
    def test_get_confirmed_tracks(self) -> None:
        """测试获取已确认的Track对象。"""
        # 创建检测结果并更新多次（超过min_hits=5）
        for i in range(6):
            x_offset = i * 10
            detections = [
                DetectionResult(
                    x=100 + x_offset, y=100 + x_offset, w=50, h=50,
                    area=2500, centroid=(125 + x_offset, 125 + x_offset),
                    confidence=1.0
                )
            ]
            self.tracker.update(detections)
        
        confirmed = self.tracker.get_confirmed_tracks()
        self.assertGreaterEqual(len(confirmed), 1)
        self.assertTrue(all(t.is_confirmed for t in confirmed))
    
    def test_cleanup_stale_tracks(self) -> None:
        """测试清理过期Track。"""
        # 创建一个轨迹并更新多次
        for i in range(6):
            x_offset = i * 10
            detections = [
                DetectionResult(
                    x=100 + x_offset, y=100 + x_offset, w=50, h=50,
                    area=2500, centroid=(125 + x_offset, 125 + x_offset),
                    confidence=1.0
                )
            ]
            self.tracker.update(detections)
        
        # 模拟轨迹很久没有匹配（增加age）
        track_id = 0  # SORT默认从0开始
        self.tracker.tracks[track_id].age = self.tracker.max_age * 3  # 远超阈值
        
        # 手动触发清理
        self.tracker._cleanup_stale_tracks()
        
        # 轨迹应该被清理
        self.assertEqual(len(self.tracker.tracks), 0)
    
    def test_reset(self) -> None:
        """测试重置跟踪器。"""
        # 创建一些轨迹并更新多次
        for i in range(6):
            x_offset = i * 10
            detections = [
                DetectionResult(
                    x=100 + x_offset, y=100 + x_offset, w=50, h=50,
                    area=2500, centroid=(125 + x_offset, 125 + x_offset),
                    confidence=1.0
                )
            ]
            self.tracker.update(detections)
        
        # 确认有轨迹
        self.assertGreater(len(self.tracker.tracks), 0)
        
        # 重置
        self.tracker.reset()
        
        # 确认已清空
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(self.tracker.frame_id, 0)
    
    def test_get_statistics(self) -> None:
        """测试获取跟踪统计信息。"""
        # 创建检测结果并更新多次以确认轨迹（超过min_hits=5）
        for i in range(6):
            x_offset = i * 10
            detections = [
                DetectionResult(
                    x=100 + x_offset, y=100 + x_offset, w=50, h=50,
                    area=2500, centroid=(125 + x_offset, 125 + x_offset),
                    confidence=1.0
                )
            ]
            self.tracker.update(detections)
        
        stats = self.tracker.get_statistics()
        
        self.assertIn('frame_id', stats)
        self.assertIn('total_tracks', stats)
        self.assertIn('confirmed_tracks', stats)
        self.assertIn('unconfirmed_tracks', stats)
        self.assertGreaterEqual(stats['total_tracks'], 1)
        self.assertGreaterEqual(stats['confirmed_tracks'], 1)
    
    def test_tracking_moving_object(self) -> None:
        """测试跟踪移动对象。"""
        # 模拟一个移动的对象（从左到右）
        for i in range(10):
            x = 100 + i * 20  # 每帧向右移动20像素
            y = 100 + i * 5   # 每帧向下移动5像素
            
            detections = [
                DetectionResult(
                    x=x, y=y, w=50, h=50,
                    area=2500, centroid=(x + 25, y + 25),
                    confidence=1.0
                )
            ]
            
            active_tracks = self.tracker.update(detections)
        
        # 应该有一个确认的轨迹
        self.assertGreaterEqual(len(active_tracks), 1)
        
        # 检查轨迹点数量
        track = active_tracks[0]
        self.assertGreaterEqual(len(track.points), 5)  # 至少5个点
        
        # 检查位移（应该向右移动了）
        dx, dy = track.get_displacement()
        self.assertGreater(dx, 0)  # X方向应该有正位移
        self.assertGreater(dy, 0)  # Y方向应该有正位移
    
    def test_tracking_multiple_objects(self) -> None:
        """测试跟踪多个对象。"""
        # 第一帧：两个对象
        detections = [
            DetectionResult(
                x=100, y=100, w=50, h=50,
                area=2500, centroid=(125, 125), confidence=1.0
            ),
            DetectionResult(
                x=300, y=300, w=50, h=50,
                area=2500, centroid=(325, 325), confidence=1.0
            )
        ]
        
        active_tracks = self.tracker.update(detections)
        
        # 第一帧没有确认的轨迹
        self.assertEqual(len(active_tracks), 0)
        
        # 但应该有两个tracks
        self.assertEqual(len(self.tracker.tracks), 2)
        
        # 继续更新多帧以确认轨迹
        for i in range(6):
            x_offset1 = i * 10
            x_offset2 = i * 5
            detections = [
                DetectionResult(
                    x=100 + x_offset1, y=100 + x_offset1, w=50, h=50,
                    area=2500, centroid=(125 + x_offset1, 125 + x_offset1),
                    confidence=1.0
                ),
                DetectionResult(
                    x=300 + x_offset2, y=300 + x_offset2, w=50, h=50,
                    area=2500, centroid=(325 + x_offset2, 325 + x_offset2),
                    confidence=1.0
                )
            ]
            active_tracks = self.tracker.update(detections)
        
        # 现在应该有两个确认的轨迹
        self.assertGreaterEqual(len(active_tracks), 2)


if __name__ == "__main__":
    unittest.main()
