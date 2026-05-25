"""
AlarmHandler 单元测试

测试告警处理模块的各项功能：
1. 快照截取（ROI提取、4倍放大、轨迹点叠加）
2. 视频截取（前置帧、当前帧、后置帧）
3. 轨迹叠加图生成
4. 告警冷却机制（同一track_id不重复告警）
5. 告警事件构建

Author: 寇豆码 (Alex)
Date: 2024
"""

import unittest
import tempfile
import shutil
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock

from src.tracker import Track
from src.trajectory_analyzer import TrajectoryResult
from src.alarm_handler import AlarmHandler, AlarmEvent


class TestAlarmHandler(unittest.TestCase):
    """AlarmHandler 测试用例。"""
    
    def setUp(self) -> None:
        """测试前准备。"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 最小化配置
        self.config = {
            "alarm": {
                "output_dir": self.temp_dir,
                "pre_alarm_seconds": 5.0,
                "post_alarm_seconds": 5.0,
                "snapshot_scale": 4.0,
                "cooldown_seconds": 10.0
            },
            "logging": {
                "level": "DEBUG",
                "save_to_file": False
            },
            "qq_bot": {
                "enabled": False  # 测试时不启用QQ机器人
            }
        }
        
        # 创建模拟的QQ机器人
        self.mock_qq_bot = Mock()
        self.mock_qq_bot.enabled = False
        
        # 创建AlarmHandler实例
        self.handler = AlarmHandler(self.config, self.mock_qq_bot)
        
        # 创建测试帧（彩色图像 640x480）
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 创建测试轨迹
        self.track = self._create_test_track()
        
        # 创建测试结果
        self.result = TrajectoryResult(
            track_id=1,
            is_parabola=True,
            parabola_a=0.05,
            parabola_b=0.0,
            parabola_c=0.0,
            vertical_speed=5.0,
            horizontal_speed=1.0,
            acceleration=0.2,
            confidence=0.8,
            direction="down"
        )
    
    def tearDown(self) -> None:
        """测试后清理。"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_track(self, track_id: int = 1) -> Track:
        """
        创建测试轨迹。
        
        Args:
            track_id: 轨迹ID
            
        Returns:
            Track: 测试轨迹对象
        """
        track = Track(track_id=track_id)
        
        # 添加10个轨迹点
        for i in range(10):
            x = 100 + i * 10
            y = 100 + i * 20
            bbox = (x - 10, y - 10, 20, 20)
            track.add_point((x, y), time.time(), i, bbox)
        
        return track
    
    def _create_mock_reader(self):
        """创建模拟的VideoReader对象。"""
        mock_reader = Mock()
        mock_reader.get_buffered_frames = Mock(return_value=[
            (time.time() - 5, self.test_frame),
            (time.time() - 4, self.test_frame),
            (time.time() - 3, self.test_frame)
        ])
        return mock_reader
    
    def test_handle_alarm_success(self) -> None:
        """测试成功处理告警。"""
        # 处理告警
        alarm = self.handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            self._create_mock_reader()
        )
        
        # 验证：应该返回AlarmEvent
        self.assertIsNotNone(alarm, "Alarm should be triggered")
        self.assertIsInstance(alarm, AlarmEvent, "Should return AlarmEvent")
        
        # 验证：告警ID格式正确
        self.assertTrue(
            alarm.alarm_id.startswith("alarm_"),
            "Alarm ID should start with 'alarm_'"
        )
        
        # 验证：快照文件已生成
        snapshot_path = Path(alarm.snapshot_path)
        self.assertTrue(snapshot_path.exists(), f"Snapshot should be saved: {snapshot_path}")
        
        # 验证：轨迹图已生成
        traj_path = Path(alarm.trajectory_image_path)
        self.assertTrue(traj_path.exists(), f"Trajectory image should be saved: {traj_path}")
    
    def test_alarm_cooldown(self) -> None:
        """测试告警冷却机制。"""
        # 第一次告警
        alarm1 = self.handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            self._create_mock_reader()
        )
        self.assertIsNotNone(alarm1, "First alarm should be triggered")
        
        # 立即第二次告警（应该在冷却期内）
        alarm2 = self.handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            self._create_mock_reader()
        )
        self.assertIsNone(alarm2, "Second alarm should be blocked by cooldown")
        
        # 等待冷却时间结束
        self.handler._last_alarm_time[self.track.track_id] = time.time() - 11
        
        # 第三次告警（应该成功）
        alarm3 = self.handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            self._create_mock_reader()
        )
        self.assertIsNotNone(alarm3, "Third alarm should be triggered after cooldown")
    
    def test_different_track_no_cooldown(self) -> None:
        """测试不同轨迹ID不受冷却影响。"""
        # 第一次告警（track_id=1）
        alarm1 = self.handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            self._create_mock_reader()
        )
        self.assertIsNotNone(alarm1, "First alarm should be triggered")
        
        # 第二次告警（不同track_id=2）
        track2 = self._create_test_track(track_id=2)
        result2 = TrajectoryResult(
            track_id=2,
            is_parabola=True,
            parabola_a=0.05,
            parabola_b=0.0,
            parabola_c=0.0,
            vertical_speed=5.0,
            horizontal_speed=1.0,
            acceleration=0.2,
            confidence=0.8,
            direction="down"
        )
        
        alarm2 = self.handler.handle_alarm(
            track2,
            result2,
            self.test_frame,
            self._create_mock_reader()
        )
        self.assertIsNotNone(alarm2, "Different track should not be blocked by cooldown")
    
    def test_snapshot_capture(self) -> None:
        """测试快照截取功能。"""
        # 处理告警
        alarm = self.handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            self._create_mock_reader()
        )
        
        # 验证：快照文件存在
        snapshot_path = Path(alarm.snapshot_path)
        self.assertTrue(snapshot_path.exists(), "Snapshot file should exist")
        
        # 验证：快照是JPEG格式
        self.assertEqual(snapshot_path.suffix, ".jpg", "Snapshot should be JPEG")
        
        # 验证：快照已正确保存（可以读取）
        snapshot = cv2.imread(str(snapshot_path))
        self.assertIsNotNone(snapshot, "Snapshot should be readable")
        
        # 验证：快照尺寸应该比原ROI大（4倍放大）
        # 原bbox约为20x20，加padding后约30x30，放大4倍后约120x120
        self.assertGreater(snapshot.shape[0], 100, "Snapshot should be enlarged")
        self.assertGreater(snapshot.shape[1], 100, "Snapshot should be enlarged")
    
    def test_trajectory_image_generation(self) -> None:
        """测试轨迹叠加图生成。"""
        # 处理告警
        alarm = self.handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            self._create_mock_reader()
        )
        
        # 验证：轨迹图文件存在
        traj_path = Path(alarm.trajectory_image_path)
        self.assertTrue(traj_path.exists(), "Trajectory image should exist")
        
        # 验证：轨迹图是JPEG格式
        self.assertEqual(traj_path.suffix, ".jpg", "Trajectory image should be JPEG")
        
        # 验证：轨迹图已正确保存
        traj_img = cv2.imread(str(traj_path))
        self.assertIsNotNone(traj_img, "Trajectory image should be readable")
    
    def test_alarm_event_fields(self) -> None:
        """测试AlarmEvent字段正确性。"""
        # 处理告警
        alarm = self.handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            self._create_mock_reader()
        )
        
        # 验证：所有字段都已正确填充
        self.assertIsNotNone(alarm.alarm_id, "Alarm ID should not be None")
        self.assertIsInstance(alarm.timestamp, datetime, "Timestamp should be datetime")
        self.assertEqual(alarm.track_id, 1, "Track ID should match")
        self.assertIsNotNone(alarm.snapshot_path, "Snapshot path should not be None")
        self.assertIsNotNone(alarm.clip_path, "Clip path should not be None")
        self.assertIsNotNone(alarm.trajectory_image_path, "Trajectory image path should not be None")
        self.assertEqual(alarm.confidence, 0.8, "Confidence should match")
        self.assertIsNotNone(alarm.message, "Message should not be None")
        
        # 验证：消息包含关键信息
        self.assertIn("高空抛物", alarm.message, "Message should contain '高空抛物'")
        self.assertIn(alarm.alarm_id, alarm.message, "Message should contain alarm ID")
    
    def test_clear_cooldown(self) -> None:
        """测试清除冷却记录。"""
        # 触发一次告警
        self.handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            self._create_mock_reader()
        )
        
        # 验证：冷却记录存在
        self.assertIn(self.track.track_id, self.handler._last_alarm_time, "Cooldown record should exist")
        
        # 清除指定track的冷却
        self.handler.clear_cooldown(track_id=self.track.track_id)
        self.assertNotIn(self.track.track_id, self.handler._last_alarm_time, "Cooldown should be cleared")
        
        # 清除所有冷却
        self.handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            self._create_mock_reader()
        )
        self.handler.clear_cooldown()
        self.assertEqual(len(self.handler._last_alarm_time), 0, "All cooldowns should be cleared")
    
    def test_generate_alarm_id_format(self) -> None:
        """测试告警ID生成格式。"""
        alarm_id = self.handler._generate_alarm_id()
        
        # 验证：格式为 alarm_YYYYMMDD_HHMMSS_xxxx
        self.assertRegex(
            alarm_id,
            r"^alarm_\d{8}_\d{6}_\d{4}$",
            "Alarm ID format should be: alarm_YYYYMMDD_HHMMSS_xxxx"
        )
    
    def test_handle_alarm_without_reader(self) -> None:
        """测试没有VideoReader的情况。"""
        # 处理告警（不提供reader）
        alarm = self.handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            None
        )
        
        # 验证：仍然应该成功（只是没有前置帧）
        self.assertIsNotNone(alarm, "Alarm should be triggered even without reader")
        
        # 验证：快照和轨迹图仍然生成
        self.assertTrue(Path(alarm.snapshot_path).exists(), "Snapshot should still be generated")
        self.assertTrue(Path(alarm.trajectory_image_path).exists(), "Trajectory image should still be generated")
    
    def test_qq_bot_push_disabled(self) -> None:
        """测试QQ机器人未启用时不推送。"""
        # 创建未启用的QQ机器人
        mock_qq_bot = Mock()
        mock_qq_bot.enabled = False
        
        handler = AlarmHandler(self.config, mock_qq_bot)
        
        # 处理告警
        alarm = handler.handle_alarm(
            self.track,
            self.result,
            self.test_frame,
            self._create_mock_reader()
        )
        
        # 验证：告警仍然处理成功
        self.assertIsNotNone(alarm, "Alarm should be triggered even if QQ bot is disabled")
        
        # 验证：QQ机器人没有被调用
        mock_qq_bot.send_alarm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
