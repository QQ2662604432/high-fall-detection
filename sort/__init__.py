"""
SORT (Simple Online and Realtime Tracking) Algorithm Package

This package implements the SORT algorithm for multi-object tracking.

Original Paper:
    Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016).
    Simple online and realtime tracking.
    In 2016 IEEE International Conference on Image Processing (ICIP).

License: GPL-3.0

Author: 寇豆码 (Alex) - Adapted from original SORT implementation
Source: https://github.com/abewley/sort
"""

from sort.sort import Sort, KalmanBoxTracker, associate_detections_to_trackers

__version__ = '1.0.0'
__author__ = 'Alex (adapted from abewley/sort)'
__license__ = 'GPL-3.0'

__all__ = ['Sort', 'KalmanBoxTracker', 'associate_detections_to_trackers']
