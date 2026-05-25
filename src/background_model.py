"""
Background Modeling Module

This module implements background subtraction using the KNN (K-Nearest Neighbors)
algorithm from OpenCV. It includes morphological post-processing to clean up
the foreground mask for better object detection.

The processing pipeline is:
1. KNN background subtraction
2. Binarization (thresholding)
3. Morphological opening (noise removal)
4. Morphological dilation (hole filling)

Author: 寇豆码 (Alex)
Date: 2024
"""

import logging
from typing import Optional
import numpy as np
import cv2

from src.utils import setup_logger


class BackgroundModel:
    """
    Background subtraction using KNN algorithm with morphological post-processing.

    This class wraps OpenCV's KNN background subtractor and applies a series of
    morphological operations to produce clean foreground masks suitable for
    object detection and tracking.

    Processing Pipeline:
        1. KNN Background Subtraction
        2. Thresholding (binary mask)
        3. Morphological Opening (remove noise)
        4. Morphological Dilation (fill holes)

    Attributes:
        config (dict): Configuration dictionary.
        logger (logging.Logger): Logger instance.
        subtractor (cv2.BackgroundSubtractorKNN): KNN background subtractor.
        kernel_open (np.ndarray): Structuring element for morphological opening.
        kernel_dilate (np.ndarray): Structuring element for dilation.
        open_iterations (int): Number of iterations for opening operation.
        dilate_iterations (int): Number of iterations for dilation operation.
        history (int): Number of frames used to initialize the background model.
        dist2_threshold (float): Threshold on squared distance.
        detect_shadows (bool): Whether to detect shadows (not recommended).
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize BackgroundModel with configuration parameters.

        Args:
            config (dict): Configuration dictionary. Should contain a
                          'background_model' section with the following keys:
                - history (int): History frames for KNN (default: 7)
                - dist2_threshold (float): Distance threshold (default: 800.0)
                - detect_shadows (bool): Detect shadows (default: False)
                - morph_open_size (int): Size of opening kernel (default: 3)
                - morph_dilate_size (int): Size of dilation kernel (default: 2)
                - open_iterations (int): Opening iterations (default: 1)
                - dilate_iterations (int): Dilation iterations (default: 1)

        Example:
            >>> config = {
            ...     'background_model': {
            ...         'history': 7,
            ...         'dist2_threshold': 800.0,
            ...         'detect_shadows': False
            ...     }
            ... }
            >>> model = BackgroundModel(config)
        """
        self.config = config
        self.logger = setup_logger(__name__, config)

        # Extract background model configuration
        bg_config = config.get('background_model', {})

        # KNN Background Subtractor parameters
        self.history = bg_config.get('history', 7)
        self.dist2_threshold = bg_config.get('dist2_threshold', 800.0)
        self.detect_shadows = bg_config.get('detect_shadows', False)

        # Initialize KNN background subtractor
        self.subtractor = cv2.createBackgroundSubtractorKNN(
            history=self.history,
            dist2Threshold=self.dist2_threshold,
            detectShadows=self.detect_shadows
        )
        self.logger.info(
            f"KNN Background Subtractor initialized: "
            f"history={self.history}, dist2Threshold={self.dist2_threshold}, "
            f"detectShadows={self.detect_shadows}"
        )

        # Morphological operation parameters
        open_size = bg_config.get('morph_open_size', 3)
        dilate_size = bg_config.get('morph_dilate_size', 2)
        self.open_iterations = bg_config.get('open_iterations', 1)
        self.dilate_iterations = bg_config.get('dilate_iterations', 1)

        # Create morphological kernels (elliptical shape works best for objects)
        self.kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size)
        )
        self.kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_size, dilate_size)
        )

        self.logger.debug(
            f"Morphological kernels created: "
            f"open=({open_size}x{open_size}, {self.open_iterations} iter), "
            f"dilate=({dilate_size}x{dilate_size}, {self.dilate_iterations} iter)"
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply background subtraction and morphological processing to a frame.

        Processing Steps:
            1. Apply KNN background subtraction to get raw foreground mask
            2. Threshold the mask to binary (0 or 255)
            3. Apply morphological opening to remove small noise
            4. Apply morphological dilation to fill holes in detected objects

        Args:
            frame (np.ndarray): Input frame in BGR format (H, W, 3).

        Returns:
            np.ndarray: Binary foreground mask (H, W) with values 0 or 255.

        Example:
            >>> model = BackgroundModel(config)
            >>> frame = cv2.imread('frame.jpg')
            >>> fg_mask = model.apply(frame)
            >>> print(fg_mask.shape, np.unique(fg_mask))
            (480, 640) [0 255]

        Note:
            - Input frame should be BGR (OpenCV default)
            - Output mask is binary: 0 (background) or 255 (foreground)
            - Shadows are not detected (detect_shadows=False)
        """
        if frame is None or frame.size == 0:
            self.logger.error("Invalid input frame: empty or None")
            return np.zeros((0, 0), dtype=np.uint8)

        # Step 1: Apply KNN background subtraction
        # Output is a grayscale mask where:
        #   - 0 = background
        #   - 127 = shadow (if detectShadows=True)
        #   - 255 = foreground
        fg_mask = self.subtractor.apply(frame)

        # Step 2: Binarization
        # Threshold at 200 to convert any foreground pixels to 255
        # This eliminates any shadow residuals (even though detectShadows=False)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Step 3: Morphological Opening
        # Removes small noise (insects, pixel noise) by:
        #   - Erosion (shrinks objects)
        #   - Dilation (restores object size)
        # Effective for removing small isolated pixels
        if self.open_iterations > 0:
            fg_mask = cv2.morphologyEx(
                fg_mask,
                cv2.MORPH_OPEN,
                self.kernel_open,
                iterations=self.open_iterations
            )

        # Step 4: Morphological Dilation
        # Fills small holes inside detected objects and
        # connects broken contours
        # Important for ensuring complete object detection
        if self.dilate_iterations > 0:
            fg_mask = cv2.dilate(
                fg_mask,
                self.kernel_dilate,
                iterations=self.dilate_iterations
            )

        return fg_mask

    def apply_batch(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply background subtraction to a batch of frames.

        Args:
            frames (np.ndarray): Batch of frames, shape (N, H, W, 3).

        Returns:
            np.ndarray: Batch of foreground masks, shape (N, H, W).

        Note:
            This is useful for processing multiple frames at once,
            but does not provide significant speedup over sequential
            processing due to OpenCV's internal implementation.
        """
        if frames is None or frames.size == 0:
            self.logger.error("Invalid input frames: empty or None")
            return np.array([], dtype=np.uint8)

        masks = []
        for frame in frames:
            mask = self.apply(frame)
            masks.append(mask)

        return np.stack(masks, axis=0)

    def reset(self) -> None:
        """
        Reset the background model to initial state.

        This is useful when switching scenes or when the background
        model becomes outdated. Creates a fresh KNN subtractor with
        the same parameters.

        Note:
            After reset, the model needs to re-learn the background
            from subsequent frames.
        """
        self.logger.info("Resetting background model...")

        self.subtractor = cv2.createBackgroundSubtractorKNN(
            history=self.history,
            dist2Threshold=self.dist2_threshold,
            detectShadows=self.detect_shadows
        )

        self.logger.info("Background model reset complete")

    def get_background_image(self) -> Optional[np.ndarray]:
        """
        Get the current background image estimated by the KNN model.

        Returns:
            Optional[np.ndarray]: Background image (BGR), or None if not available.

        Note:
            The background image is only available after processing
            at least 'history' frames.
        """
        try:
            bg_image = self.subtractor.getBackgroundImage()
            return bg_image
        except Exception as e:
            self.logger.warning(f"Could not get background image: {e}")
            return None

    def set_history(self, history: int) -> None:
        """
        Update the history parameter of the KNN subtractor.

        Args:
            history (int): New history value (number of frames).

        Note:
            This does not reset the model; it only affects future frames.
        """
        self.history = history
        # Note: OpenCV KNN does not support dynamic history update
        # Need to recreate the subtractor
        self.logger.warning("History parameter changed, resetting background model...")
        self.reset()

    def set_dist2_threshold(self, threshold: float) -> None:
        """
        Update the distance threshold parameter.

        Args:
            threshold (float): New distance threshold.

        Note:
            Lower values make the model more sensitive to changes.
            Higher values make it more robust to noise but may miss slow objects.
        """
        self.dist2_threshold = threshold
        self.subtractor.setDist2Threshold(threshold)
        self.logger.debug(f"Distance threshold updated: {threshold}")

    def get_config(self) -> dict:
        """
        Get the current configuration of the background model.

        Returns:
            dict: Current configuration parameters.
        """
        return {
            'history': self.history,
            'dist2_threshold': self.dist2_threshold,
            'detect_shadows': self.detect_shadows,
            'morph_open_size': self.kernel_open.shape[0],
            'morph_dilate_size': self.kernel_dilate.shape[0],
            'open_iterations': self.open_iterations,
            'dilate_iterations': self.dilate_iterations
        }

    def __repr__(self) -> str:
        """Return string representation of the BackgroundModel."""
        return (
            f"BackgroundModel("
            f"history={self.history}, "
            f"dist2_threshold={self.dist2_threshold}, "
            f"detect_shadows={self.detect_shadows})"
        )
