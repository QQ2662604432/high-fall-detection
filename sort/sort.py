"""
SORT (Simple Online and Realtime Tracking) Algorithm Implementation

This module implements the SORT algorithm for multi-object tracking.
It uses Kalman Filter for motion prediction and Hungarian Algorithm for
data association.

Original Paper:
    Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016).
    Simple online and realtime tracking.
    In 2016 IEEE International Conference on Image Processing (ICIP).

Original Implementation:
    https://github.com/abewley/sort

License:
    GPL-3.0 (GNU General Public License v3.0)

Author:
    Original: Alex Bewley (abewley)
    Adapted: 寇豆码 (Alex)

Modifications:
    - Adapted for Python 3.7+
    - Integrated with the High-Altitude Parabolic Detection System
    - Added type hints and docstrings
    - Optimized for real-time performance
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import filterpy for Kalman Filter
try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
    logger.info("Using filterpy for Kalman Filter")
except ImportError:
    FILTERPY_AVAILABLE = False
    logger.warning("filterpy not installed. Using simplified Kalman Filter.")


class KalmanBoxTracker:
    """
    Kalman Filter-based box tracker for 2D bounding boxes.

    This class tracks a single object using a Kalman Filter.
    The state space is 7-dimensional:
        [x, y, s, r, dx, dy, ds]
    where:
        - x, y: center coordinates
        - s: scale (area)
        - r: aspect ratio (constant)
        - dx, dy, ds: velocity components

    The measurement space is 4-dimensional:
        [x, y, s, r]

    Attributes:
        id (int): Unique tracker ID.
        history (List[np.ndarray]): Track history of bounding boxes.
        hits (int): Number of successful detections.
        hit_streak (int): Number of consecutive successful detections.
        age (int): Number of frames since tracker creation.
        time_since_update (int): Number of frames since last detection.
    """

    # Class-level counter for unique ID generation
    _count = 0

    def __init__(self, bbox: np.ndarray):
        """
        Initialize a tracker with the first bounding box detection.

        Args:
            bbox (np.ndarray): Initial bounding box [x1, y1, x2, y2, score].
        """
        # Convert bbox to center format [x, y, s, r]
        x1, y1, x2, y2 = bbox[:4]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        s = (x2 - x1) * (y2 - y1)  # area
        r = (x2 - x1) / (y2 - y1) if (y2 - y1) != 0 else 1.0  # aspect ratio

        # Initialize Kalman Filter
        if FILTERPY_AVAILABLE:
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self._init_kalman_filter(center_x, center_y, s, r)
        else:
            self._init_simplified_kalman(center_x, center_y, s, r)

        # Assign unique ID
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        # Tracking history
        self.history: List[np.ndarray] = []

        # Tracking statistics
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0

    def _init_kalman_filter(self, cx: float, cy: float, s: float, r: float) -> None:
        """Initialize Kalman Filter with standard parameters."""
        # State transition matrix (7x7)
        # [x, y, s, r, dx, dy, ds]
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 0, 1, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 0, 1],  # s = s + ds
            [0, 0, 0, 1, 0, 0, 0],  # r = r (constant)
            [0, 0, 0, 0, 1, 0, 0],  # dx = dx
            [0, 0, 0, 0, 0, 1, 0],  # dy = dy
            [0, 0, 0, 0, 0, 0, 1]   # ds = ds
        ])

        # Measurement matrix (4x7)
        # We only measure [x, y, s, r]
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        # Measurement noise covariance (4x4)
        self.kf.R = np.eye(4) * 10.0

        # Process noise covariance (7x7)
        self.kf.Q = np.eye(7) * 0.01

        # Initial state covariance (7x7)
        self.kf.P = np.eye(7) * 100.0

        # Initial state [x, y, s, r, dx, dy, ds]
        self.kf.x = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float32)

    def _init_simplified_kalman(self, cx: float, cy: float, s: float, r: float) -> None:
        """Initialize simplified Kalman Filter (when filterpy is not available)."""
        # Store state directly [x, y, s, r, dx, dy, ds]
        self.state = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float32)
        self.covariance = np.eye(7) * 100.0

    def update(self, bbox: np.ndarray) -> None:
        """
        Update the Kalman Filter with a new measurement.

        Args:
            bbox (np.ndarray): Bounding box [x1, y1, x2, y2, score].
        """
        # Convert bbox to center format [x, y, s, r]
        x1, y1, x2, y2 = bbox[:4]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1) if (y2 - y1) != 0 else 1.0

        measurement = np.array([center_x, center_y, s, r], dtype=np.float32)

        if FILTERPY_AVAILABLE:
            self.kf.update(measurement)
        else:
            # Simplified update (just overwrite state)
            self.state[:4] = measurement
            self.time_since_update = 0

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def predict(self) -> np.ndarray:
        """
        Predict the next state using the Kalman Filter.

        Returns:
            np.ndarray: Predicted bounding box in [x1, y1, x2, y2] format (4,).
        """
        self.age += 1
        self.time_since_update += 1

        if FILTERPY_AVAILABLE:
            self.kf.predict()
            state = self.kf.x.copy()
        else:
            # Simplified prediction
            self.state[0] += self.state[4]  # x += dx
            self.state[1] += self.state[5]  # y += dy
            self.state[2] += self.state[6]  # s += ds
            state = self.state.copy()

        # Convert state to bbox format [x1, y1, x2, y2]
        cx, cy, s, r = state[0], state[1], state[2], state[3]

        # Avoid negative scale
        s = max(s, 1.0)

        # Compute width and height from scale and aspect ratio
        # s = w * h, r = w / h
        # => w = sqrt(s * r), h = sqrt(s / r)
        if r != 0 and s > 0:
            w = np.sqrt(s * abs(r))
            h = s / w if w != 0 else np.sqrt(s)
        else:
            w = np.sqrt(s)
            h = np.sqrt(s)

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def get_state(self) -> np.ndarray:
        """
        Get the current state as a bounding box.

        Returns:
            np.ndarray: Current bounding box [x1, y1, x2, y2] (4,).
        """
        if FILTERPY_AVAILABLE:
            state = self.kf.x.copy()
        else:
            state = self.state.copy()

        cx, cy, s, r = state[0], state[1], state[2], state[3]

        # Avoid negative scale
        s = max(s, 1.0)

        # Compute width and height
        if r != 0 and s > 0:
            w = np.sqrt(s * abs(r))
            h = s / w if w != 0 else np.sqrt(s)
        else:
            w = np.sqrt(s)
            h = np.sqrt(s)

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def reset_counter() -> None:
        """Reset the global tracker ID counter."""
        KalmanBoxTracker._count = 0


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU (Intersection over Union) between two sets of bounding boxes.

    Vectorized implementation for batch processing.

    Args:
        bboxes1 (np.ndarray): First set of bboxes, shape (N, 4) or (N, 5).
        bboxes2 (np.ndarray): Second set of bboxes, shape (M, 4) or (M, 5).

    Returns:
        np.ndarray: IoU matrix of shape (N, M).

    Example:
        >>> bboxes1 = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
        >>> bboxes2 = np.array([[5, 5, 15, 15]])
        >>> iou = iou_batch(bboxes1, bboxes2)
        >>> print(iou.shape)
        (2, 1)
    """
    # Ensure we only use the first 4 columns (x1, y1, x2, y2)
    bboxes1 = bboxes1[:, :4] if bboxes1.shape[1] > 4 else bboxes1
    bboxes2 = bboxes2[:, :4] if bboxes2.shape[1] > 4 else bboxes2

    N = bboxes1.shape[0]
    M = bboxes2.shape[0]

    # Expand dimensions for broadcasting: (N, 1, 4) and (1, M, 4)
    bboxes1_exp = bboxes1.reshape(N, 1, 4)
    bboxes2_exp = bboxes2.reshape(1, M, 4)

    # Compute intersection coordinates
    x1 = np.maximum(bboxes1_exp[..., 0], bboxes2_exp[..., 0])
    y1 = np.maximum(bboxes1_exp[..., 1], bboxes2_exp[..., 1])
    x2 = np.minimum(bboxes1_exp[..., 2], bboxes2_exp[..., 2])
    y2 = np.minimum(bboxes1_exp[..., 3], bboxes2_exp[..., 3])

    # Compute intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Compute areas
    area1 = (bboxes1_exp[..., 2] - bboxes1_exp[..., 0]) * \
            (bboxes1_exp[..., 3] - bboxes1_exp[..., 1])
    area2 = (bboxes2_exp[..., 2] - bboxes2_exp[..., 0]) * \
            (bboxes2_exp[..., 3] - bboxes2_exp[..., 1])

    # Compute union area
    union = area1 + area2 - intersection

    # Compute IoU (add small epsilon to avoid division by zero)
    iou = intersection / (union + 1e-8)

    return iou


def associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float = 0.3
) -> Tuple[List[int], List[int], List[int]]:
    """
    Associate detections to trackers using Hungarian Algorithm.

    This function uses the Hungarian Algorithm (via scipy) to find the optimal
    assignment of detections to trackers based on IoU.

    Args:
        detections (np.ndarray): Detection bboxes, shape (N, 5) [x1, y1, x2, y2, score].
        trackers (np.ndarray): Tracker bboxes, shape (M, 4) [x1, y1, x2, y2].
        iou_threshold (float): IoU threshold for matching.

    Returns:
        Tuple[List[int], List[int], List[int]]:
            - matched_indices: List of (detection_idx, tracker_idx) pairs.
            - unmatched_detections: List of unmatched detection indices.
            - unmatched_trackers: List of unmatched tracker indices.

    Example:
        >>> dets = np.array([[0, 0, 10, 10, 0.9], [20, 20, 30, 30, 0.8]])
        >>> trks = np.array([[5, 5, 15, 15], [18, 18, 28, 28]])
        >>> matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)
    """
    if len(trackers) == 0:
        # No trackers, all detections are unmatched
        return [], list(range(len(detections))), []

    if len(detections) == 0:
        # No detections, all trackers are unmatched
        return [], [], list(range(len(trackers)))

    # Compute IoU matrix
    iou_matrix = iou_batch(detections, trackers)

    # Convert to cost matrix (1 - IoU)
    cost_matrix = 1.0 - iou_matrix

    # Apply Hungarian Algorithm
    try:
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
    except ImportError:
        # Fallback: greedy matching if scipy is not available
        logger.warning("scipy not installed. Using greedy matching.")
        row_indices, col_indices = _greedy_matching(cost_matrix, iou_matrix, iou_threshold)

    # Filter matches by IoU threshold
    matched_indices = []
    unmatched_detections = list(range(len(detections)))
    unmatched_trackers = list(range(len(trackers)))

    for row, col in zip(row_indices, col_indices):
        if iou_matrix[row, col] >= iou_threshold:
            matched_indices.append((row, col))
            unmatched_detections.remove(row)
            unmatched_trackers.remove(col)

    return matched_indices, unmatched_detections, unmatched_trackers


def _greedy_matching(
    cost_matrix: np.ndarray,
    iou_matrix: np.ndarray,
    iou_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy matching as fallback when Hungarian Algorithm is not available.

    Args:
        cost_matrix (np.ndarray): Cost matrix (1 - IoU).
        iou_matrix (np.ndarray): IoU matrix.
        iou_threshold (float): IoU threshold.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Row indices and column indices of matches.
    """
    matches = []
    num_rows, num_cols = cost_matrix.shape

    # Create a copy of IoU matrix for modification
    iou_temp = iou_matrix.copy()

    while True:
        # Find the best match (highest IoU)
        max_iou = np.max(iou_temp)
        if max_iou < iou_threshold:
            break

        max_loc = np.unravel_index(np.argmax(iou_temp), iou_temp.shape)
        row, col = max_loc

        matches.append((row, col))

        # Remove matched row and column
        iou_temp[row, :] = -1
        iou_temp[:, col] = -1

    if matches:
        rows, cols = zip(*matches)
        return np.array(rows), np.array(cols)
    else:
        return np.array([]), np.array([])


class Sort:
    """
    SORT (Simple Online and Realtime Tracking) Algorithm.

    This class implements the SORT algorithm for multi-object tracking.
    It uses Kalman Filter for motion prediction and Hungarian Algorithm
    for data association.

    Algorithm Steps:
        1. Predict new locations of existing trackers
        2. Associate detections to trackers (IoU + Hungarian)
        3. Update matched trackers
        4. Create new trackers for unmatched detections
        5. Remove dead trackers (not updated for max_age frames)

    Attributes:
        max_age (int): Maximum frames to keep a tracker without updates.
        min_hits (int): Minimum hits before a tracker is considered valid.
        iou_threshold (float): IoU threshold for data association.
        trackers (List[KalmanBoxTracker]): List of active trackers.
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 5,
        iou_threshold: float = 0.1
    ) -> None:
        """
        Initialize SORT tracker.

        Args:
            max_age (int): Maximum number of frames to keep a tracker alive
                          without being matched to a detection. Default: 3.
            min_hits (int): Minimum number of detection matches before a
                          tracker is considered valid (returned in update()).
                          Default: 5.
            iou_threshold (float): IoU threshold for matching detections to
                                  trackers. Default: 0.1.

        Example:
            >>> tracker = Sort(max_age=3, min_hits=5, iou_threshold=0.1)
            >>> detections = np.array([[100, 100, 200, 200, 0.9]])
            >>> trackers = tracker.update(detections)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

        logger.info(
            f"SORT tracker initialized: "
            f"max_age={max_age}, min_hits={min_hits}, "
            f"iou_threshold={iou_threshold}"
        )

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Update the tracker with new detections.

        Args:
            dets (np.ndarray): Detection array of shape (N, 5).
                              Each row is [x1, y1, x2, y2, score].

        Returns:
            np.ndarray: Tracker array of shape (M, 5).
                       Each row is [x1, y1, x2, y2, track_id].
                       Only returns trackers that have been matched
                       to detections at least min_hits times.

        Note:
            - If no detections, returns empty array with shape (0, 5).
            - Trackers not updated for max_age frames are automatically removed.
        """
        self.frame_count += 1

        # Get predictions from all trackers
        trk_preds = []
        to_del = []

        for t, tracker in enumerate(self.trackers):
            pred_bbox = tracker.predict()
            trk_preds.append(pred_bbox)
            if tracker.time_since_update > self.max_age:
                to_del.append(t)

        # If no trackers left, create new trackers for all detections
        if len(trk_preds) == 0:
            return self._create_new_trackers(dets)

        # Convert predictions to numpy array
        trk_preds = np.array(trk_preds)

        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trk_preds, self.iou_threshold
        )

        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(dets[det_idx, :])

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(dets[det_idx, :])
            self.trackers.append(trk)

        # Remove dead trackers (AFTER association and adding new trackers)
        # Note: to_del contains indices into the ORIGINAL self.trackers list
        # Since we appended new trackers at the end, the indices in to_del are still valid
        for t in sorted(to_del, reverse=True):
            if t < len(self.trackers):
                logger.debug(f"Removing dead tracker {self.trackers[t].id}")
                self.trackers.pop(t)

        # Prepare output: only return trackers with enough hits
        output = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = tracker.get_state()
                output.append([bbox[0], bbox[1], bbox[2], bbox[3], tracker.id])

        if len(output) == 0:
            return np.empty((0, 5))

        return np.array(output)

    def _create_new_trackers(self, dets: np.ndarray) -> np.ndarray:
        """
        Create new trackers for all detections (when no trackers exist).

        Args:
            dets (np.ndarray): Detection array.

        Returns:
            np.ndarray: Empty array (new trackers need min_hits before output).
        """
        for det in dets:
            trk = KalmanBoxTracker(det)
            self.trackers.append(trk)

        return np.empty((0, 5))

    def get_trackers(self) -> List[KalmanBoxTracker]:
        """
        Get all active trackers.

        Returns:
            List[KalmanBoxTracker]: List of active trackers.
        """
        return self.trackers

    def reset(self) -> None:
        """
        Reset the tracker (clear all trackers and counters).

        Useful when switching scenes or restarting tracking.
        """
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.reset_counter()
        logger.info("SORT tracker reset")

    def __repr__(self) -> str:
        """Return string representation of the SORT tracker."""
        return (
            f"Sort("
            f"max_age={self.max_age}, "
            f"min_hits={self.min_hits}, "
            f"iou_threshold={self.iou_threshold}, "
            f"active_trackers={len(self.trackers)})"
        )


# Example usage and testing
if __name__ == '__main__':
    # Simple test
    import numpy as np

    # Initialize tracker
    tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.1)

    # Simulate detections over frames
    print("Testing SORT tracker...")
    for frame in range(10):
        # Simulate a moving object
        x1 = 100 + frame * 5
        y1 = 100 + frame * 5
        x2 = x1 + 50
        y2 = y1 + 50

        dets = np.array([[x1, y1, x2, y2, 0.9]])

        # Update tracker
        tracks = tracker.update(dets)

        if len(tracks) > 0:
            print(f"Frame {frame}: Track ID = {int(tracks[0][4])}")
        else:
            print(f"Frame {frame}: No valid tracks yet (need {tracker.min_hits} hits)")

    print("\nTest completed!")
