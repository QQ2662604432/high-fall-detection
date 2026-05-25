"""
轨迹分析模块

实现高空抛物轨迹分析，包括：
1. 三点抛物线拟合（y = ax² + bx + c）→ 判断 a > 0
2. X/Y轴位移分离判断 → 垂直位移主导
3. 速度/加速度阈值过滤
4. 综合判定：满足条件越多，置信度越高

Author: 寇豆码 (Alex)
Date: 2024
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from src.utils import setup_logger
from src.tracker import Track


@dataclass
class TrajectoryResult:
    """
    轨迹分析结果数据结构。
    
    Attributes:
        track_id: 关联轨迹ID
        is_parabola: 是否为抛物线
        parabola_a: 二次项系数（a > 0 → 下落）
        parabola_b: 一次项系数
        parabola_c: 常数项
        vertical_speed: 垂直速度（像素/秒，向下为正）
        horizontal_speed: 水平速度（像素/秒）
        acceleration: 加速度
        confidence: 置信度 [0, 1]
        direction: 运动方向: "down" / "up" / "diagonal" / "horizontal"
    """
    
    track_id: int           # 关联轨迹ID
    is_parabola: bool       # 是否为抛物线
    parabola_a: float       # 二次项系数（a > 0 → 下落）
    parabola_b: float       # 一次项系数
    parabola_c: float       # 常数项
    vertical_speed: float   # 垂直速度（像素/秒，向下为正）
    horizontal_speed: float # 水平速度（像素/秒）
    acceleration: float     # 加速度
    confidence: float       # 置信度 [0, 1]
    direction: str          # 运动方向: "down" / "up" / "diagonal" / "horizontal"


class TrajectoryAnalyzer:
    """
    轨迹分析器 - 判断轨迹是否为高空抛物。
    
    通过分析轨迹点的空间分布和运动特征，综合判断是否为抛物轨迹：
    1. 抛物线拟合：使用numpy.polyfit进行二次拟合
    2. 位移分析：分离X/Y轴位移，判断垂直方向主导
    3. 速度分析：计算垂直/水平速度及加速度
    4. 综合判定：多条件加权，输出置信度
    
    Attributes:
        min_points: 最小拟合点数
        parabola_a_threshold: 抛物线系数a的阈值
        min_vertical_speed: 最小垂直速度（像素/帧）
        max_horizontal_ratio: 水平/垂直速度比最大值
        min_track_length: 最小轨迹长度
        min_acceleration: 最小加速度阈值
        fit_recent_n: 使用最近N个点进行拟合
        logger: 日志器实例
        
    Example:
        >>> config = {"trajectory_analyzer": {"min_points_for_fit": 5}}
        >>> analyzer = TrajectoryAnalyzer(config)
        >>> result = analyzer.analyze(track)
        >>> print(f"Is falling: {analyzer.is_falling_object(result)}")
    """
    
    def __init__(self, config: dict) -> None:
        """
        初始化轨迹分析器。
        
        Args:
            config: 配置字典，应包含 "trajectory_analyzer" 节
            
        Example:
            >>> config = {
            ...     "trajectory_analyzer": {
            ...         "min_points_for_fit": 5,
            ...         "parabola_a_threshold": 0.01,
            ...         "min_vertical_speed": 2.0,
            ...         "max_horizontal_ratio": 2.0,
            ...         "min_track_length": 5,
            ...         "min_acceleration": 0.1,
            ...         "fit_recent_n": 8
            ...     }
            ... }
            >>> analyzer = TrajectoryAnalyzer(config)
        """
        ta_config = config.get("trajectory_analyzer", {})
        self.min_points = ta_config.get("min_points_for_fit", 5)
        self.parabola_a_threshold = ta_config.get("parabola_a_threshold", 0.01)
        self.min_vertical_speed = ta_config.get("min_vertical_speed", 2.0)    # 像素/帧
        self.max_horizontal_ratio = ta_config.get("max_horizontal_ratio", 2.0) # 水平/垂直速度比
        self.min_track_length = ta_config.get("min_track_length", 5)
        self.min_acceleration = ta_config.get("min_acceleration", 0.1)
        self.fit_recent_n = ta_config.get("fit_recent_n", 8)  # 用最近N个点拟合
        
        # 初始化日志器
        self.logger = setup_logger(__name__, config.get("logging", {}))
        self.logger.info(
            f"TrajectoryAnalyzer initialized: "
            f"min_points={self.min_points}, "
            f"parabola_a_threshold={self.parabola_a_threshold}, "
            f"min_vertical_speed={self.min_vertical_speed}"
        )
    
    def analyze(self, track: Track) -> TrajectoryResult:
        """
        分析轨迹，判断是否为抛物。
        
        分析流程：
        1. 获取最近N个轨迹点
        2. 抛物线拟合（y = ax² + bx + c）
        3. X/Y轴位移分离判断
        4. 速度/加速度分析
        5. 综合判定，计算置信度
        
        Args:
            track: Track对象（包含轨迹点列、时间戳、速度历史）
            
        Returns:
            TrajectoryResult: 分析结果
            
        Example:
            >>> result = analyzer.analyze(track)
            >>> if result.is_parabola:
            ...     print(f"Parabola detected with a={result.parabola_a:.3f}")
        """
        # 获取最近N个轨迹点
        points = track.get_recent_points(self.fit_recent_n)
        
        # 点数不足，无法分析
        if len(points) < self.min_points:
            self.logger.debug(
                f"Track {track.track_id}: insufficient points "
                f"({len(points)} < {self.min_points})"
            )
            return TrajectoryResult(
                track_id=track.track_id,
                is_parabola=False,
                parabola_a=0.0,
                parabola_b=0.0,
                parabola_c=0.0,
                vertical_speed=0.0,
                horizontal_speed=0.0,
                acceleration=0.0,
                confidence=0.0,
                direction="unknown"
            )
        
        # 提取x, y序列（注意：图像坐标系y向下为正）
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        
        # ========== 算法1: 三点抛物线拟合 ==========
        # 高空抛物轨迹（物理模型）：
        #   实际世界: z = g*t²/2 + v0*t + z0 (抛物线)
        #   图像投影: y = a*x² + b*x + c
        #   抛物下落: a > 0 (开口向上，因为y向下为正)
        # 飞鸟轨迹: 近似直线或正弦曲线，不满足a > 0
        try:
            # numpy.polyfit返回 [a, b, c] 对应 y = ax² + bx + c
            coeffs = np.polyfit(xs, ys, 2)
            a, b, c = coeffs
        except (np.linalg.LinAlgError, ValueError) as e:
            self.logger.warning(
                f"Track {track.track_id}: polyfit failed - {e}"
            )
            a, b, c = 0.0, 0.0, 0.0
        
        # a > 0 表示抛物线开口向下（图像坐标系y向下为正）
        is_parabola = a > self.parabola_a_threshold
        
        # ========== 算法2: X/Y轴位移分离判断 ==========
        total_dx = abs(xs[-1] - xs[0])  # X轴总位移（绝对值）
        total_dy = ys[-1] - ys[0]       # Y轴总位移（正值=向下，图像坐标）
        
        # 计算平均速度（像素/帧间隔）
        n_frames = len(points) - 1
        vertical_speed = total_dy / max(n_frames, 1)
        horizontal_speed = total_dx / max(n_frames, 1)
        
        # 垂直方向主导？
        # 条件1: 垂直速度超过阈值
        # 条件2: 水平/垂直速度比小于阈值
        is_vertical_dominant = (
            vertical_speed > self.min_vertical_speed and
            (horizontal_speed / max(vertical_speed, 0.001)) < self.max_horizontal_ratio
        )
        
        # ========== 算法3: 速度/加速度分析 ==========
        if len(track.velocities) >= 3:
            # 使用间隔2帧的速度差来估计加速度
            acceleration = track.velocities[-1] - track.velocities[-3]
        else:
            acceleration = 0.0
        
        is_accelerating = acceleration > self.min_acceleration
        
        # ========== 综合判定 ==========
        # 满足条件越多，置信度越高
        conditions_met = sum([is_parabola, is_vertical_dominant, is_accelerating])
        confidence = conditions_met / 3.0
        
        # 判定为抛物: 至少满足2/3条件 且 垂直速度向下
        is_falling = conditions_met >= 2 and vertical_speed > 0
        
        # ========== 方向判断 ==========
        if total_dy > 0 and total_dx < total_dy * 0.5:
            direction = "down"
        elif total_dy > 0:
            direction = "diagonal"
        elif total_dy < 0:
            direction = "up"
        else:
            direction = "horizontal"
        
        # 构建结果
        result = TrajectoryResult(
            track_id=track.track_id,
            is_parabola=is_parabola,
            parabola_a=float(a),
            parabola_b=float(b),
            parabola_c=float(c),
            vertical_speed=float(vertical_speed),
            horizontal_speed=float(horizontal_speed),
            acceleration=float(acceleration),
            confidence=float(confidence),
            direction=direction
        )
        
        self.logger.debug(
            f"Track {track.track_id} analyzed: "
            f"parabola={is_parabola}, "
            f"vertical_dominant={is_vertical_dominant}, "
            f"accelerating={is_accelerating}, "
            f"confidence={confidence:.2f}, "
            f"direction={direction}"
        )
        
        return result
    
    def is_falling_object(self, result: TrajectoryResult) -> bool:
        """
        最终判定: 是否为高空抛物。
        
        综合判断条件：
        1. 置信度 >= 0.6
        2. 运动方向为 "down" 或 "diagonal"
        3. 垂直速度超过阈值
        
        Args:
            result: TrajectoryResult对象
            
        Returns:
            bool: 是否为抛物
            
        Example:
            >>> if analyzer.is_falling_object(result):
            ...     print("高空抛物 detected!")
        """
        is_falling = (
            result.confidence >= 0.6 and
            result.direction in ("down", "diagonal") and
            result.vertical_speed > self.min_vertical_speed
        )
        
        if is_falling:
            self.logger.info(
                f"Falling object detected: "
                f"track_id={result.track_id}, "
                f"confidence={result.confidence:.2f}, "
                f"direction={result.direction}, "
                f"vertical_speed={result.vertical_speed:.2f}"
            )
        
        return is_falling
    
    def batch_analyze(self, tracks: List[Track]) -> List[TrajectoryResult]:
        """
        批量分析多个轨迹。
        
        Args:
            tracks: Track对象列表
            
        Returns:
            List[TrajectoryResult]: 分析结果列表
            
        Example:
            >>> results = analyzer.batch_analyze(tracks)
            >>> falling_tracks = [r for r in results if analyzer.is_falling_object(r)]
        """
        results = []
        for track in tracks:
            result = self.analyze(track)
            results.append(result)
        return results
