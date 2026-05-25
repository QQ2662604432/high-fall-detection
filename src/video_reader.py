"""
Video Reader Module

This module implements video input functionality with support for both RTSP streams
and local video files. It features automatic source type detection, reconnection logic
for RTSP streams with exponential backoff, and a circular frame buffer for
capturing recent frames (useful for alarm video playback).

Author: 寇豆码 (Alex)
Date: 2024
"""

import time
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from collections import deque
import numpy as np
import cv2

from src.utils import setup_logger, generate_timestamp


class VideoReader:
    """
    Video reader with RTSP/file support, auto-reconnection, and frame buffering.

    This class provides a unified interface for reading video from RTSP streams
    or local files. It handles RTSP disconnection with exponential backoff
    reconnection and maintains a circular buffer of recent frames for
    post-event analysis.

    Attributes:
        source (str): Video source (RTSP URL or file path).
        config (dict): Configuration dictionary.
        logger (logging.Logger): Logger instance.
        cap (cv2.VideoCapture): OpenCV video capture object.
        is_rtsp (bool): Whether the source is an RTSP stream.
        fps (float): Frames per second of the video source.
        frame_buffer (deque): Circular buffer for recent frames.
        buffer_seconds (float): Buffer duration in seconds.
        max_reconnect_attempts (int): Maximum RTSP reconnection attempts.
        reconnect_interval (int): Base reconnection interval in seconds.
        _reconnect_count (int): Current reconnection attempt count.
    """

    def __init__(self, source: str, config: dict) -> None:
        """
        Initialize VideoReader with video source and configuration.

        Args:
            source (str): RTSP URL or local file path.
            config (dict): Configuration dictionary containing video settings.

        Raises:
            ValueError: If the video source cannot be opened after retries.
        """
        self.source = source
        self.config = config
        self.logger = setup_logger(__name__, config)

        # Determine source type
        self.is_rtsp = source.lower().startswith('rtsp://')
        self.logger.info(f"Video source type: {'RTSP' if self.is_rtsp else 'File'}")

        # Video capture configuration
        video_config = config.get('video', {})
        self.buffer_seconds = video_config.get('buffer_seconds', 6.0)
        self.max_reconnect_attempts = video_config.get('max_reconnect_attempts', 10)
        self.reconnect_interval = video_config.get('reconnect_interval', 1)

        # Initialize video capture
        self.cap = None
        self.fps = 30.0  # Default FPS
        self.frame_buffer: deque = deque()
        self._reconnect_count = 0

        self._initialize_capture()

        self.logger.info(f"VideoReader initialized: source={self.source}, fps={self.fps:.2f}")

    def _initialize_capture(self) -> None:
        """
        Initialize OpenCV VideoCapture with appropriate settings.

        For RTSP sources, applies low-latency settings (buffer size = 1).
        For file sources, opens with default settings.

        Raises:
            ValueError: If the video source cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.source)

        if self.is_rtsp:
            # Apply RTSP-specific low-latency settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.logger.info("Applied RTSP low-latency settings (buffer_size=1)")

        if not self.cap.isOpened():
            if self.is_rtsp:
                self.logger.warning(f"Initial RTSP connection failed, attempting reconnection...")
                if not self._reconnect():
                    raise ValueError(f"Cannot open video source: {self.source}")
            else:
                raise ValueError(f"Cannot open video file: {self.source}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.logger.warning(f"Invalid FPS ({self.fps}), using default 30.0")
            self.fps = 30.0

        # Initialize circular buffer
        buffer_size = int(self.buffer_seconds * self.fps)
        self.frame_buffer = deque(maxlen=buffer_size)
        self.logger.debug(f"Frame buffer initialized: size={buffer_size} frames ({self.buffer_seconds}s)")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the video source.

        For RTSP sources, automatically attempts reconnection on failure.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - ret (bool): True if frame was successfully read.
                - frame (np.ndarray or None): The captured frame (BGR), or None on failure.

        Note:
            Failed RTSP reads trigger reconnection logic with exponential backoff.
            After max_reconnect_attempts, the method returns (False, None).
        """
        if self.cap is None:
            self.logger.error("VideoCapture is not initialized")
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            if self.is_rtsp:
                self.logger.warning("Frame read failed, attempting RTSP reconnection...")
                if self._reconnect():
                    ret, frame = self.cap.read()
                else:
                    self.logger.error("RTSP reconnection failed, cannot read frame")
                    return False, None
            else:
                self.logger.warning("End of video file reached or read error")
                return False, None

        if ret and frame is not None:
            # Add frame to circular buffer with timestamp
            timestamp = generate_timestamp()
            self.frame_buffer.append((timestamp, frame.copy()))
            return True, frame

        return False, None

    def get_buffered_frames(self, seconds: float) -> List[Tuple[float, np.ndarray]]:
        """
        Retrieve recent frames from the circular buffer.

        Args:
            seconds (float): Number of seconds of recent frames to retrieve.
                            Must be <= buffer_seconds.

        Returns:
            List[Tuple[float, np.ndarray]]:
                List of (timestamp, frame) tuples, ordered oldest to newest.

        Example:
            >>> reader = VideoReader('video.mp4', config)
            >>> frames = reader.get_buffered_frames(5.0)  # Last 5 seconds
            >>> len(frames)
            150  # Assuming 30 FPS

        Note:
            If 'seconds' exceeds buffer_seconds, returns all available frames.
        """
        if seconds <= 0:
            self.logger.warning("Requested seconds <= 0, returning empty list")
            return []

        frame_count = int(seconds * self.fps)
        available_count = len(self.frame_buffer)

        if frame_count > available_count:
            self.logger.warning(
                f"Requested {frame_count} frames, only {available_count} available"
            )
            frame_count = available_count

        # Return the most recent 'frame_count' frames
        result = list(self.frame_buffer)[-frame_count:]
        self.logger.debug(f"Retrieved {len(result)} buffered frames ({seconds}s)")
        return result

    def get_current_timestamp(self) -> float:
        """
        Get the timestamp of the most recently read frame.

        Returns:
            float: Timestamp in seconds, or 0.0 if no frames have been read.
        """
        if len(self.frame_buffer) > 0:
            return self.frame_buffer[-1][0]
        return 0.0

    def release(self) -> None:
        """
        Release the video capture resource.

        This method safely releases the OpenCV VideoCapture object
        and clears the frame buffer to free memory.

        Note:
            This method is idempotent and can be called multiple times.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.logger.info("VideoCapture released")

        if self.frame_buffer is not None:
            self.frame_buffer.clear()
            self.logger.debug("Frame buffer cleared")

    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to RTSP stream with exponential backoff.

        Implements exponential backoff with a maximum wait time of 30 seconds.
        Resets the reconnection count on successful connection.

        Returns:
            bool: True if reconnection successful, False if max attempts exceeded.

        Algorithm:
            1. Calculate wait_time = min(reconnect_interval * 2^attempt, 30)
            2. Wait for wait_time seconds
            3. Attempt to reopen the capture
            4. Repeat up to max_reconnect_attempts times

        Note:
            Only applicable for RTSP sources. Has no effect on file sources.
        """
        if not self.is_rtsp:
            self.logger.warning("_reconnect called for non-RTSP source, ignoring")
            return False

        self.logger.info(f"Starting RTSP reconnection (max attempts: {self.max_reconnect_attempts})")

        for attempt in range(self.max_reconnect_attempts):
            # Calculate exponential backoff wait time (capped at 30 seconds)
            wait_time = min(self.reconnect_interval * (2 ** attempt), 30)
            self.logger.warning(
                f"RTSP reconnection attempt {attempt + 1}/{self.max_reconnect_attempts}, "
                f"waiting {wait_time}s before retry"
            )
            time.sleep(wait_time)

            try:
                # Release existing capture if open
                if self.cap is not None:
                    self.cap.release()

                # Attempt to reopen
                self.cap = cv2.VideoCapture(self.source)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if self.cap.isOpened():
                    self.logger.info(f"RTSP reconnection successful on attempt {attempt + 1}")
                    self._reconnect_count = 0  # Reset counter on success
                    return True

            except Exception as e:
                self.logger.error(f"Exception during RTSP reconnection attempt {attempt + 1}: {e}")

        self.logger.error(
            f"RTSP reconnection failed after {self.max_reconnect_attempts} attempts"
        )
        self._reconnect_count = self.max_reconnect_attempts
        return False

    def is_opened(self) -> bool:
        """
        Check if the video capture is currently open.

        Returns:
            bool: True if the video capture is open and readable.
        """
        return self.cap is not None and self.cap.isOpened()

    def get_fps(self) -> float:
        """
        Get the FPS of the video source.

        Returns:
            float: Frames per second.
        """
        return self.fps

    def get_frame_count(self) -> int:
        """
        Get the total number of frames in the video (for file sources).

        Returns:
            int: Total frame count, or -1 if unavailable (e.g., RTSP streams).
        """
        if self.cap is None:
            return -1
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_buffer_info(self) -> dict:
        """
        Get information about the current state of the frame buffer.

        Returns:
            dict: Dictionary containing buffer statistics.
        """
        return {
            'buffer_seconds': self.buffer_seconds,
            'max_buffer_size': self.frame_buffer.maxlen,
            'current_size': len(self.frame_buffer),
            'fps': self.fps,
            'is_rtsp': self.is_rtsp,
            'reconnect_count': self._reconnect_count
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resource cleanup."""
        self.release()

    def __del__(self):
        """Destructor - ensures resource cleanup on garbage collection."""
        self.release()
