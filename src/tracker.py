"""
轨迹跟踪模块

SORT算法集成层，负责：
1. SORT算法封装（Sort类封装）
2. 检测结果 → numpy数组 → SORT输入 → Track对象
3. Track对象生命周期管理（创建、更新、清理过期Track）
4. 提供update(detections)接口，返回活跃轨迹列表

Author: 寇豆码 (Alex)
Date: 2024
"""

import logging
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import numpy as np

from src.utils import setup_logger
from sort.sort import Sort


@dataclass
class Track:
    """
    轨迹数据结构。
    
    封装单个轨迹的所有历史信息，包括质心点列、时间戳、
    帧号、边界框历史、速度历史等。
    
    Attributes:
        track_id: 轨迹ID（唯一标识）
        points: 质心历史点列 [(x, y), ...]
        timestamps: 对应时间戳 [timestamp, ...]
        frame_ids: 对应帧号 [frame_id, ...]
        bbox_history: bbox历史 [x1, y1, x2, y2]
        velocities: 速度历史（像素/帧）
        age: 自上次匹配的帧数
        hits: 总匹配次数
        is_confirmed: 是否已确认
        
    Example:
        >>> track = Track(track_id=1)
        >>> track.add_point((100, 100), time.time(), 1, (90, 90, 110, 110))
        >>> track.is_confirmed
        False
    """
    
    track_id: int                        # 轨迹ID
    points: List[Tuple[int, int]] = field(default_factory=list)       # 质心历史点列
    timestamps: List[float] = field(default_factory=list)             # 对应时间戳
    frame_ids: List[int] = field(default_factory=list)                # 对应帧号
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)  # bbox历史 [x1,y1,x2,y2]
    velocities: List[float] = field(default_factory=list)             # 速度历史（像素/帧）
    age: int = 0                           # 自上次匹配的帧数
    hits: int = 0                          # 总匹配次数
    is_confirmed: bool = False             # 是否已确认
    
    def add_point(self, point: Tuple[int, int], timestamp: float,
                   frame_id: int, bbox: Tuple[int, int, int, int]) -> None:
        """
        添加轨迹点。
        
        Args:
            point: 质心坐标 (cx, cy)
            timestamp: 时间戳
            frame_id: 帧号
            bbox: 边界框 (x1, y1, x2, y2)
            
        Example:
            >>> track.add_point((100, 100), time.time(), 1, (90, 90, 110, 110))
        """
        self.points.append(point)
        self.timestamps.append(timestamp)
        self.frame_ids.append(frame_id)
        self.bbox_history.append(bbox)
        
        # 计算速度（与前一帧的差）
        if len(self.points) >= 2:
            dx = self.points[-1][0] - self.points[-2][0]
            dy = self.points[-1][1] - self.points[-2][1]
            velocity = np.sqrt(dx**2 + dy**2)
            self.velocities.append(velocity)
        
        self.hits += 1
        self.age = 0  # 重置age（因为刚匹配）
    
    def get_recent_points(self, n: int) -> List[Tuple[int, int]]:
        """
        获取最近n个轨迹点。
        
        Args:
            n: 要获取的点的数量
            
        Returns:
            List[Tuple[int, int]]: 最近n个点（如果不足n个，返回全部）
            
        Example:
            >>> points = track.get_recent_points(5)
            >>> len(points) <= 5
            True
        """
        return self.points[-n:] if len(self.points) >= n else self.points
    
    def get_displacement(self) -> Tuple[float, float]:
        """
        获取X/Y轴总位移。
        
        Returns:
            Tuple[float, float]: (dx, dy) 总位移
            
        Example:
            >>> dx, dy = track.get_displacement()
            >>> print(f"X位移: {dx}, Y位移: {dy}")
        """
        if len(self.points) < 2:
            return (0.0, 0.0)
        
        dx = self.points[-1][0] - self.points[0][0]
        dy = self.points[-1][1] - self.points[0][1]
        return (float(dx), float(dy))
    
    def get_average_velocity(self, last_n: int = 5) -> float:
        """
        获取平均速度。
        
        Args:
            last_n: 使用最近n个速度点计算平均
            
        Returns:
            float: 平均速度（像素/帧）
            
        Example:
            >>> avg_vel = track.get_average_velocity(last_n=5)
        """
        if len(self.velocities) == 0:
            return 0.0
        
        recent_velocities = self.velocities[-last_n:] if len(self.velocities) >= last_n else self.velocities
        return float(np.mean(recent_velocities))
    
    def to_dict(self) -> dict:
        """
        转换为字典格式（用于序列化）。
        
        Returns:
            dict: 包含所有轨迹信息的字典
            
        Example:
            >>> track_dict = track.to_dict()
        """
        return {
            'track_id': self.track_id,
            'points': self.points,
            'timestamps': self.timestamps,
            'frame_ids': self.frame_ids,
            'bbox_history': self.bbox_history,
            'velocities': self.velocities,
            'age': self.age,
            'hits': self.hits,
            'is_confirmed': self.is_confirmed
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Track':
        """
        从字典恢复Track对象。
        
        Args:
            data: 字典格式的轨迹数据
            
        Returns:
            Track: 恢复的Track对象
            
        Example:
            >>> track = Track.from_dict(track_dict)
        """
        track = cls(track_id=data['track_id'])
        track.points = data['points']
        track.timestamps = data['timestamps']
        track.frame_ids = data['frame_ids']
        track.bbox_history = data['bbox_history']
        track.velocities = data['velocities']
        track.age = data['age']
        track.hits = data['hits']
        track.is_confirmed = data['is_confirmed']
        return track


class Tracker:
    """
    SORT轨迹跟踪封装。
    
    功能：
    1. 封装SORT算法（Sort类）
    2. 检测结果 → numpy数组 → SORT输入 → Track对象
    3. Track对象生命周期管理
    4. 提供update(detections)接口
    
    处理流程：
    1. 将DetectionResult转换为SORT输入格式 [x1, y1, x2, y2, score]
    2. 调用SORT.update()获取跟踪结果
    3. 同步SORT内部trackers到Track对象
    4. 更新Track对象（创建新Track或更新已有Track）
    5. 清理过期Track
    6. 返回已确认的活跃轨迹
    
    Attributes:
        config (dict): 配置字典
        max_age (int): 丢失N帧后删除轨迹
        min_hits (int): 连续N帧确认目标
        iou_threshold (float): IOU匹配阈值
        sort (Sort): SORT跟踪器实例
        tracks (Dict[int, Track]): 轨迹管理: track_id → Track
        frame_id (int): 当前帧号
        logger (logging.Logger): 日志器实例
        
    Example:
        >>> config = {"tracker": {"max_age": 3, "min_hits": 5, "iou_threshold": 0.1}}
        >>> tracker = Tracker(config)
        >>> dets = [DetectionResult(x=100, y=100, w=50, h=50, area=2500, centroid=(125, 125))]
        >>> tracks = tracker.update(dets)
        >>> len(tracks)
        0  # 需要min_hits=5帧才能确认
    """
    
    def __init__(self, config: dict) -> None:
        """
        初始化Tracker。
        
        Args:
            config: 配置字典，应包含 "tracker" 节
            
        Example:
            >>> config = {
            ...     "tracker": {
            ...         "max_age": 3,
            ...         "min_hits": 5,
            ...         "iou_threshold": 0.1
            ...     }
            ... }
            >>> tracker = Tracker(config)
        """
        self.config = config
        self.logger = setup_logger(__name__, config)
        
        # 从配置中提取跟踪参数
        tracker_config = config.get("tracker", {})
        self.max_age = tracker_config.get("max_age", 3)
        self.min_hits = tracker_config.get("min_hits", 5)
        self.iou_threshold = tracker_config.get("iou_threshold", 0.1)
        
        # 初始化SORT跟踪器
        self.sort = Sort(
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold
        )
        
        # 轨迹管理: track_id → Track
        self.tracks: Dict[int, Track] = {}
        self.frame_id = 0
        
        self.logger.info(
            f"Tracker initialized: "
            f"max_age={self.max_age}, min_hits={self.min_hits}, "
            f"iou_threshold={self.iou_threshold}"
        )
    
    def update(self, detections: List) -> List[Track]:
        """
        使用SORT更新跟踪状态。
        
        处理流程：
        1. 转换检测结果为SORT输入格式
        2. 调用SORT.update()
        3. 同步SORT内部trackers到Track对象
        4. 更新Track对象
        5. 清理过期Track
        6. 返回已确认的活跃轨迹
        
        Args:
            detections: 当前帧检测结果（DetectionResult对象列表）
            
        Returns:
            List[Track]: 当前活跃的确认轨迹列表
        """
        self.frame_id += 1
        
        # 转换为SORT输入格式: [x1, y1, x2, y2, score]
        if len(detections) > 0:
            dets = np.array([list(d.to_xyxy()) + [d.confidence] for d in detections])
        else:
            dets = np.empty((0, 5))
        
        # SORT更新
        # trackers格式: [[x1, y1, x2, y2, track_id], ...]
        trackers = self.sort.update(dets)
        
        # 同步SORT内部的trackers到我们的Track对象
        # 关键：必须为SORT中的所有tracker调用add_point()
        # 而不是只给matched的调用

        # 首先，为所有SORT内部的tracker创建Track对象（如果不存在）
        for sort_tracker in self.sort.trackers:
            sort_track_id = sort_tracker.id

            if sort_track_id not in self.tracks:
                self.tracks[sort_track_id] = Track(track_id=sort_track_id)
                self.logger.debug(f"Created new track from SORT: ID={sort_track_id}")

        # 然后，从SORT的返回结果中找到matched的tracker，调用add_point()
        matched_track_ids = set()
        for t in trackers:
            x1, y1, x2, y2, tid = t
            tid = int(tid)
            matched_track_ids.add(tid)

        # 为matched的tracker更新点
        for t in trackers:
            x1, y1, x2, y2, tid = t
            track_id = int(tid)
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            timestamp = time.time()

            self.tracks[track_id].add_point(centroid, timestamp, self.frame_id, bbox)

            # 判断是否已确认
            if self.tracks[track_id].hits >= self.min_hits:
                self.tracks[track_id].is_confirmed = True

        # 为所有SORT内部的tracker更新age（未匹配的age+1）
        for sort_tracker in self.sort.trackers:
            sort_track_id = sort_tracker.id
            if sort_track_id not in matched_track_ids:
                self.tracks[sort_track_id].age += 1
        
        # 收集活跃的已确认轨迹
        active_tracks = [t for t in self.tracks.values() if t.is_confirmed]
        
        # 清理过期Track
        self._cleanup_stale_tracks()
        
        self.logger.debug(
            f"Frame {self.frame_id}: "
            f"detections={len(detections)}, "
            f"active_tracks={len(active_tracks)}, "
            f"total_tracks={len(self.tracks)}"
        )
        
        return active_tracks
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """
        获取指定ID的Track对象。
        
        Args:
            track_id: 轨迹ID
            
        Returns:
            Optional[Track]: Track对象（如果存在），否则返回None
            
        Example:
            >>> track = tracker.get_track(1)
            >>> if track:
            ...     print(f"Track {track_id} has {len(track.points)} points")
        """
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> List[Track]:
        """
        获取所有Track对象。
        
        Returns:
            List[Track]: 所有Track对象列表
            
        Example:
            >>> all_tracks = tracker.get_all_tracks()
        """
        return list(self.tracks.values())
    
    def get_confirmed_tracks(self) -> List[Track]:
        """
        获取所有已确认的Track对象。
        
        Returns:
            List[Track]: 已确认的Track对象列表
            
        Example:
            >>> confirmed = tracker.get_confirmed_tracks()
        """
        return [t for t in self.tracks.values() if t.is_confirmed]
    
    def _cleanup_stale_tracks(self) -> None:
        """
        移除超龄的Track对象。
        
        清理条件：
        - Track.age > max_age * 2
        
        Note:
            使用max_age * 2作为阈值，给予更多缓冲。
        """
        stale_ids = [
            tid for tid, track in self.tracks.items()
            if track.age > self.max_age * 2
        ]
        
        for tid in stale_ids:
            self.logger.debug(f"Removing stale track: ID={tid}, age={self.tracks[tid].age}")
            del self.tracks[tid]
        
        if stale_ids:
            self.logger.info(f"Cleaned up {len(stale_ids)} stale tracks")
    
    def reset(self) -> None:
        """
        重置跟踪器（清除所有轨迹和计数器）。
        
        Example:
            >>> tracker.reset()
        """
        self.tracks.clear()
        self.frame_id = 0
        self.sort.reset()
        self.logger.info("Tracker reset")
    
    def get_statistics(self) -> dict:
        """
        获取跟踪统计信息。
        
        Returns:
            dict: 包含跟踪统计信息的字典
            
        Example:
            >>> stats = tracker.get_statistics()
            >>> print(f"Total tracks: {stats['total_tracks']}")
        """
        confirmed_count = len([t for t in self.tracks.values() if t.is_confirmed])
        unconfirmed_count = len([t for t in self.tracks.values() if not t.is_confirmed])
        
        return {
            'frame_id': self.frame_id,
            'total_tracks': len(self.tracks),
            'confirmed_tracks': confirmed_count,
            'unconfirmed_tracks': unconfirmed_count,
            'max_age': self.max_age,
            'min_hits': self.min_hits,
            'iou_threshold': self.iou_threshold
        }
    
    def __repr__(self) -> str:
        """返回Tracker的字符串表示。"""
        return (
            f"Tracker("
            f"max_age={self.max_age}, "
            f"min_hits={self.min_hits}, "
            f"iou_threshold={self.iou_threshold}, "
            f"tracks={len(self.tracks)})"
        )
