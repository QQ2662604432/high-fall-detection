"""
告警处理模块

实现告警事件处理，包括：
1. 快照截取（bbox+padding+4倍放大+轨迹点叠加）
2. 视频截取（前置帧从缓冲区取+后置帧继续采集）
3. 轨迹叠加图生成
4. 告警冷却机制（同一轨迹10秒内不重复告警）
5. QQ机器人推送集成

Author: 寇豆码 (Alex)
Date: 2024
"""

import logging
import time
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from src.utils import setup_logger, ensure_dir, generate_alarm_id
from src.tracker import Track
from src.trajectory_analyzer import TrajectoryResult


@dataclass
class AlarmEvent:
    """
    告警事件数据结构。
    
    Attributes:
        alarm_id: 告警唯一ID（时间戳+随机数）
        timestamp: 告警时间
        track_id: 关联轨迹ID
        snapshot_path: 快照图片路径
        clip_path: 告警视频路径
        trajectory_image_path: 轨迹叠加图路径
        confidence: 置信度
        message: 格式化告警消息
    """
    
    alarm_id: str                # 告警唯一ID（时间戳+随机数）
    timestamp: datetime          # 告警时间
    track_id: int                # 关联轨迹ID
    snapshot_path: str           # 快照图片路径
    clip_path: str               # 告警视频路径
    trajectory_image_path: str   # 轨迹叠加图路径
    confidence: float            # 置信度
    message: str                 # 格式化告警消息


class AlarmHandler:
    """
    告警处理器。职责: 截取快照(4倍放大) + 录制前后5秒视频 + 推送告警。
    
    处理流程：
    1. 检查告警冷却（同一track_id在冷却时间内不重复告警）
    2. 生成告警ID
    3. 截取抛物特写快照（4倍放大）
    4. 截取告警前后5秒视频
    5. 生成轨迹叠加图
    6. 构建告警事件
    7. 通过QQ机器人推送告警
    
    Attributes:
        output_dir: 输出目录（Path对象）
        pre_alarm_seconds: 告警前视频时长（秒）
        post_alarm_seconds: 告警后视频时长（秒）
        snapshot_scale: 快照放大倍数
        qq_bot: QQ机器人实例
        alarm_cooldown: 告警冷却时间（秒）
        _last_alarm_time: 上次告警时间记录 {track_id: timestamp}
        logger: 日志器实例
        
    Example:
        >>> handler = AlarmHandler(config, qq_bot)
        >>> alarm = handler.handle_alarm(track, result, frame, reader)
        >>> if alarm:
        ...     print(f"Alarm triggered: {alarm.alarm_id}")
    """
    
    def __init__(self, config: dict, qq_bot=None) -> None:
        """
        初始化告警处理器。
        
        Args:
            config: 配置字典，应包含 "alarm" 节
            qq_bot: QQ机器人实例（可选）
            
        Example:
            >>> config = {
            ...     "alarm": {
            ...         "output_dir": "output",
            ...         "pre_alarm_seconds": 5.0,
            ...         "post_alarm_seconds": 5.0,
            ...         "snapshot_scale": 4.0,
            ...         "cooldown_seconds": 10.0
            ...     }
            ... }
            >>> handler = AlarmHandler(config, qq_bot)
        """
        alarm_config = config.get("alarm", {})
        self.output_dir = Path(alarm_config.get("output_dir", "output"))
        self.pre_alarm_seconds = alarm_config.get("pre_alarm_seconds", 5.0)
        self.post_alarm_seconds = alarm_config.get("post_alarm_seconds", 5.0)
        self.snapshot_scale = alarm_config.get("snapshot_scale", 4.0)
        self.qq_bot = qq_bot
        
        # 创建输出目录
        ensure_dir(self.output_dir)
        ensure_dir(self.output_dir / "snapshots")
        ensure_dir(self.output_dir / "clips")
        
        # 防重复告警: 同一track_id的冷却时间
        self.alarm_cooldown = alarm_config.get("cooldown_seconds", 10.0)
        self._last_alarm_time: Dict[int, float] = {}
        
        # 日志器
        self.logger = setup_logger(__name__, config.get("logging", {}))
        
        self.logger.info(
            f"AlarmHandler initialized: "
            f"output_dir={self.output_dir}, "
            f"pre_alarm={self.pre_alarm_seconds}s, "
            f"post_alarm={self.post_alarm_seconds}s, "
            f"snapshot_scale={self.snapshot_scale}x, "
            f"cooldown={self.alarm_cooldown}s"
        )
    
    def handle_alarm(self, track: Track, result: TrajectoryResult,
                    frame: np.ndarray, reader=None) -> Optional[AlarmEvent]:
        """
        处理告警事件。
        
        处理流程：
        1. 冷却检查（同一track_id在冷却时间内不重复告警）
        2. 生成告警ID
        3. 截取抛物特写快照（4倍放大）
        4. 截取告警视频（前后5秒）
        5. 生成轨迹叠加图
        6. 构建告警事件
        7. QQ推送
        
        Args:
            track: 确认的抛物轨迹
            result: 轨迹分析结果
            frame: 当前帧 (BGR)
            reader: VideoReader实例（用于获取缓冲帧，可选）
            
        Returns:
            Optional[AlarmEvent]: 告警事件，如果冷却中则返回None
            
        Example:
            >>> alarm = handler.handle_alarm(track, result, frame, reader)
            >>> if alarm:
            ...     print(f"Alarm {alarm.alarm_id} triggered!")
        """
        # 冷却检查
        now = time.time()
        if track.track_id in self._last_alarm_time:
            time_since_last = now - self._last_alarm_time[track.track_id]
            if time_since_last < self.alarm_cooldown:
                self.logger.debug(
                    f"告警冷却中: track={track.track_id}, "
                    f"remaining={self.alarm_cooldown - time_since_last:.1f}s"
                )
                return None
        
        # 更新告警时间
        self._last_alarm_time[track.track_id] = now
        
        # Step 1: 生成告警ID
        alarm_id = self._generate_alarm_id()
        self.logger.info(f"Processing alarm {alarm_id} for track {track.track_id}")
        
        # Step 2: 截取抛物特写快照（4倍放大）
        latest_bbox = track.bbox_history[-1] if track.bbox_history else (0, 0, 0, 0)
        snapshot_path = self._capture_snapshot(frame, latest_bbox, track, alarm_id)
        
        # Step 3: 截取告警视频（前后5秒）
        clip_path = self._clip_video(reader, track, alarm_id, frame)
        
        # Step 4: 生成轨迹叠加图
        traj_img_path = self._save_trajectory_image(frame, track, alarm_id)
        
        # Step 5: 构建告警事件
        alarm = AlarmEvent(
            alarm_id=alarm_id,
            timestamp=datetime.now(),
            track_id=track.track_id,
            snapshot_path=str(snapshot_path),
            clip_path=str(clip_path),
            trajectory_image_path=str(traj_img_path),
            confidence=result.confidence,
            message=self._format_alarm_message(alarm_id, result, track)
        )
        
        # Step 6: QQ推送
        if self.qq_bot and self.qq_bot.enabled:
            try:
                self.qq_bot.send_alarm(alarm)
                self.logger.info(f"Alarm {alarm_id} sent to QQ bot")
            except Exception as e:
                self.logger.error(f"Failed to send alarm {alarm_id} to QQ bot: {e}")
        
        self.logger.info(
            f"告警已处理: {alarm_id}, track={track.track_id}, "
            f"confidence={result.confidence:.2f}"
        )
        
        return alarm
    
    def _generate_alarm_id(self) -> str:
        """
        生成告警ID: alarm_YYYYMMDD_HHMMSS_random4digits。
        
        Returns:
            str: 唯一的告警ID字符串
            
        Example:
            >>> alarm_id = self._generate_alarm_id()
            >>> print(alarm_id)
            'alarm_20240115_103045_3281'
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_digits = np.random.randint(1000, 9999)
        return f"alarm_{timestamp}_{random_digits}"
    
    def _capture_snapshot(self, frame: np.ndarray, bbox: Tuple,
                         track: Track, alarm_id: str) -> Path:
        """
        截取抛物特写放大图。
        
        处理流程：
        1. 从bbox提取ROI区域（带padding）
        2. 4倍放大（使用INTER_CUBIC插值）
        3. 在放大图上叠加轨迹点
        4. 保存为JPEG文件
        
        Args:
            frame: 当前帧
            bbox: 边界框 (x, y, w, h)
            track: Track对象（用于绘制轨迹点）
            alarm_id: 告警ID
            
        Returns:
            Path: 快照文件路径
            
        Example:
            >>> snapshot_path = self._capture_snapshot(frame, bbox, track, "alarm_xxx")
        """
        x, y, w, h = bbox
        
        # 扩展ROI（留一些上下文，padding为bbox长边的1/2）
        pad = max(w, h) // 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        
        # 提取ROI区域
        roi = frame[y1:y2, x1:x2].copy()
        
        # 4倍放大（使用INTER_CUBIC插值，效果比LINEAR好）
        h_roi, w_roi = roi.shape[:2]
        roi_scaled = cv2.resize(
            roi,
            (int(w_roi * self.snapshot_scale), int(h_roi * self.snapshot_scale)),
            interpolation=cv2.INTER_CUBIC
        )
        
        # 在放大图上叠加轨迹点（最近10个点）
        for pt in track.points[-10:]:
            # 将轨迹点坐标转换到放大图坐标系
            scaled_pt = (
                int((pt[0] - x1) * self.snapshot_scale),
                int((pt[1] - y1) * self.snapshot_scale)
            )
            # 确保点在图像范围内
            if 0 <= scaled_pt[0] < roi_scaled.shape[1] and 0 <= scaled_pt[1] < roi_scaled.shape[0]:
                cv2.circle(roi_scaled, scaled_pt, 3, (0, 0, 255), -1)
        
        # 保存快照
        path = self.output_dir / "snapshots" / f"{alarm_id}.jpg"
        cv2.imwrite(str(path), roi_scaled)
        self.logger.debug(f"快照已保存: {path}")
        
        return path
    
    def _clip_video(self, reader, track: Track,
                    alarm_id: str, current_frame: np.ndarray) -> Path:
        """
        截取告警前后5秒视频。
        
        处理流程：
        1. 获取前置帧（从VideoReader缓冲区）
        2. 添加当前帧
        3. 后置帧需要在主循环中继续采集（此处先占位）
        4. 叠加轨迹框到每帧
        5. 编码保存为MP4文件
        
        Args:
            reader: VideoReader实例（可选）
            track: Track对象
            alarm_id: 告警ID
            current_frame: 当前帧
            
        Returns:
            Path: 视频文件路径
            
        Note:
            后置帧的采集需要在主循环中完成，此处仅处理前置帧和当前帧。
            完整实现需要在main.py中协调后置帧的采集。
            
        Example:
            >>> clip_path = self._clip_video(reader, track, "alarm_xxx", frame)
        """
        all_frames = []
        
        # 获取前置帧
        if reader and hasattr(reader, 'get_buffered_frames'):
            try:
                pre_frames = reader.get_buffered_frames(self.pre_alarm_seconds)
                all_frames.extend(pre_frames)
                self.logger.debug(f"Got {len(pre_frames)} pre-alarm frames")
            except Exception as e:
                self.logger.warning(f"Failed to get pre-alarm frames: {e}")
        
        # 添加当前帧
        all_frames.append((time.time(), current_frame))
        
        # 后置帧需要在主循环中继续采集（此处先占位）
        # 实际实现时，可以在main.py中采集完post_frames后再调用此方法
        # 或者返回一个占位符，由main.py后续填充
        post_frames = []
        
        # 合并所有帧
        all_frames = all_frames + post_frames
        
        # 如果没有帧，至少保存当前帧
        if len(all_frames) == 0:
            all_frames = [(time.time(), current_frame)]
        
        # 叠加轨迹框到每帧
        overlay_frames = []
        for i, (ts, f) in enumerate(all_frames):
            overlay_frame = self._draw_track_overlay(f, track)
            overlay_frames.append((ts, overlay_frame))
        
        # 编码保存
        path = self.output_dir / "clips" / f"{alarm_id}.mp4"
        
        # 获取视频尺寸
        h, w = overlay_frames[0][1].shape[:2]
        
        # 使用mp4v编码器（跨平台兼容）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
        
        # 写入所有帧
        for ts, f in overlay_frames:
            writer.write(f)
        writer.release()
        
        self.logger.debug(f"告警视频已保存: {path} ({len(overlay_frames)} frames)")
        
        return path
    
    def _draw_track_overlay(self, frame: np.ndarray, track: Track) -> np.ndarray:
        """
        在帧上叠加轨迹点和边界框。
        
        Args:
            frame: 输入帧
            track: Track对象
            
        Returns:
            np.ndarray: 叠加后的帧
            
        Example:
            >>> overlay = self._draw_track_overlay(frame, track)
        """
        overlay = frame.copy()
        
        # 绘制历史轨迹点（最近20个点，绿色）
        for i, pt in enumerate(track.points[-20:]):
            color = (0, 255, 0)  # 绿色
            cv2.circle(overlay, pt, 3, color, -1)
        
        # 绘制轨迹线（最近20个点）
        if len(track.points) >= 2:
            pts = np.array(track.points[-20:], dtype=np.int32)
            cv2.polylines(overlay, [pts], False, (0, 255, 0), 2)
        
        # 绘制当前边界框（红色）
        if track.bbox_history:
            x, y, w, h = track.bbox_history[-1]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # 在边界框上方显示track_id
            label = f"Track {track.track_id}"
            cv2.putText(overlay, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return overlay
    
    def _save_trajectory_image(self, frame: np.ndarray, track: Track,
                               alarm_id: str) -> Path:
        """
        生成轨迹叠加图。
        
        Args:
            frame: 当前帧
            track: Track对象
            alarm_id: 告警ID
            
        Returns:
            Path: 轨迹图文件路径
            
        Example:
            >>> traj_path = self._save_trajectory_image(frame, track, "alarm_xxx")
        """
        # 生成轨迹叠加图
        traj_img = self._draw_track_overlay(frame, track)
        
        # 保存
        path = self.output_dir / "clips" / f"{alarm_id}_trajectory.jpg"
        cv2.imwrite(str(path), traj_img)
        self.logger.debug(f"轨迹图已保存: {path}")
        
        return path
    
    def _format_alarm_message(self, alarm_id: str, result: TrajectoryResult,
                              track: Track) -> str:
        """
        格式化告警消息。
        
        Args:
            alarm_id: 告警ID
            result: 轨迹分析结果
            track: Track对象
            
        Returns:
            str: 格式化的告警消息
            
        Example:
            >>> message = self._format_alarm_message("alarm_xxx", result, track)
        """
        return (
            f"🚨 高空抛物告警 🚨\n"
            f"告警ID: {alarm_id}\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"置信度: {result.confidence:.1%}\n"
            f"轨迹ID: {track.track_id}\n"
            f"垂直速度: {result.vertical_speed:.1f} 像素/帧\n"
            f"方向: {result.direction}\n"
            f"请及时处理！"
        )
    
    def clear_cooldown(self, track_id: int = None) -> None:
        """
        清除告警冷却记录。
        
        Args:
            track_id: 要清除的轨迹ID（如果为None，清除所有）
            
        Example:
            >>> handler.clear_cooldown(track_id=1)  # 清除指定track的冷却
            >>> handler.clear_cooldown()  # 清除所有冷却
        """
        if track_id is not None:
            if track_id in self._last_alarm_time:
                del self._last_alarm_time[track_id]
                self.logger.debug(f"Cleared cooldown for track {track_id}")
        else:
            self._last_alarm_time.clear()
            self.logger.debug("Cleared all cooldown records")
