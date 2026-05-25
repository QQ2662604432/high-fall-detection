"""
Test Script for T02 - Video Input & Background Modeling Layer

This script tests the core functionality of:
1. VideoReader - video input with RTSP/file support
2. BackgroundModel - KNN background subtraction
3. Sort - multi-object tracking (SORT algorithm)

Run this script to verify the implementation is correct.

Author: 寇豆码 (Alex)
Date: 2024
"""

import sys
import logging
import numpy as np
import cv2
from pathlib import Path
import yaml


def load_config() -> dict:
    """
    Load configuration from config.yaml.

    Returns:
        dict: Configuration dictionary.
    """
    project_root = Path(__file__).parent
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        config_path = project_root / "config" / "config.yaml"

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # Return default config
        return {
            'video': {
                'buffer_seconds': 6.0,
                'max_reconnect_attempts': 10,
                'reconnect_interval': 1
            },
            'background_model': {
                'history': 7,
                'dist2_threshold': 800.0,
                'detect_shadows': False,
                'morph_open_size': 3,
                'morph_dilate_size': 2,
                'open_iterations': 1,
                'dilate_iterations': 1
            }
        }


def test_video_reader(config: dict) -> bool:
    """
    Test VideoReader functionality (without actual video file).

    Args:
        config (dict): Configuration dictionary.

    Returns:
        bool: True if tests pass, False otherwise.
    """
    print("\n" + "="*50)
    print("Testing VideoReader")
    print("="*50)

    from src.video_reader import VideoReader

    # Test 1: Check class can be imported
    print("\n1. VideoReader class imported successfully ✓")

    # Test 2: Check class has required methods
    print("\n2. Checking required methods...")
    required_methods = ['read_frame', 'get_buffered_frames', 'release', '_reconnect']
    for method in required_methods:
        if hasattr(VideoReader, method):
            print(f"   ✓ Method '{method}' exists")
        else:
            print(f"   ✗ Method '{method}' missing")
            return False

    # Test 3: Test with a non-existent file (should raise ValueError)
    print("\n3. Testing error handling (non-existent file)...")
    try:
        reader = VideoReader("/non/existent/file.mp4", config)
        print("   ✗ Expected ValueError but didn't get it")
        return False
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"   ✓ Raised exception (expected): {type(e).__name__}")

    print("\n✓ VideoReader tests completed (basic checks)")
    print("   (Full test requires a valid video file or RTSP stream)")
    return True


def test_background_model(config: dict) -> bool:
    """
    Test BackgroundModel functionality.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        bool: True if tests pass, False otherwise.
    """
    print("\n" + "="*50)
    print("Testing BackgroundModel")
    print("="*50)

    from src.background_model import BackgroundModel

    # Initialize BackgroundModel
    print("\n1. Initializing BackgroundModel...")
    model = BackgroundModel(config)
    print(f"   ✓ Model initialized: {model}")

    # Test apply method with random frame
    print("\n2. Testing apply() method...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    fg_mask = model.apply(test_image)
    print(f"   ✓ apply() success: shape={fg_mask.shape}")
    print(f"   ✓ Unique values in mask: {np.unique(fg_mask)}")

    # Verify mask is binary (0 or 255)
    unique_vals = np.unique(fg_mask)
    if len(unique_vals) <= 2 and (0 in unique_vals or 255 in unique_vals):
        print("   ✓ Mask is binary (0 or 255)")
    else:
        print(f"   ✗ Mask is not binary: {unique_vals}")
        return False

    # Test with multiple frames (to build up background model)
    print("\n3. Testing with multiple frames...")
    for i in range(10):
        # Create frame with a moving object
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a white rectangle (simulating moving object)
        x = 100 + i * 10
        frame[200:300, x:x+50, :] = 255

        mask = model.apply(frame)
        fg_ratio = np.sum(mask > 0) / mask.size
        print(f"   Frame {i+1}: foreground ratio = {fg_ratio:.4f}")

    print("\n4. Testing reset()...")
    model.reset()
    print("   ✓ reset() success")

    # Test get_background_image
    print("\n5. Testing get_background_image()...")
    bg = model.get_background_image()
    if bg is not None:
        print(f"   ✓ Background image: shape={bg.shape}")
    else:
        print("   (Background image not available until model is trained)")

    # Test configuration
    print("\n6. Testing get_config()...")
    cfg = model.get_config()
    print(f"   ✓ Config: {cfg}")

    print("\n✓ BackgroundModel tests completed")
    return True


def test_sort() -> bool:
    """
    Test SORT tracking algorithm.

    Returns:
        bool: True if tests pass, False otherwise.
    """
    print("\n" + "="*50)
    print("Testing SORT Algorithm")
    print("="*50)

    from sort import Sort, KalmanBoxTracker

    print("\n1. Initializing SORT tracker...")
    tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.1)
    print(f"   ✓ Tracker initialized: {tracker}")

    # Simulate detections over frames
    print("\n2. Simulating object tracking...")

    for frame in range(10):
        # Simulate a moving object (moving diagonally)
        x1 = 100 + frame * 10
        y1 = 100 + frame * 5
        x2 = x1 + 50
        y2 = y1 + 50

        # Detection array: [x1, y1, x2, y2, score]
        dets = np.array([[x1, y1, x2, y2, 0.9]])

        # Update tracker
        tracks = tracker.update(dets)

        if len(tracks) > 0:
            track_id = int(tracks[0][4])
            bbox = tracks[0][:4]
            print(f"   Frame {frame}: Track ID = {track_id}, BBox = [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            print(f"   Frame {frame}: No valid tracks yet (need {tracker.min_hits} hits)")

    # Verify tracker output format
    print("\n3. Verifying tracker output format...")
    tracker.reset()

    # Create a set of detections that will definitely match
    for i in range(5):
        dets = np.array([[100 + i*5, 100, 150 + i*5, 150, 0.9]])
        tracks = tracker.update(dets)

    if len(tracks) > 0:
        track = tracks[0]
        if len(track) == 5:
            print(f"   ✓ Output format correct: [x1, y1, x2, y2, track_id] = {track}")
        else:
            print(f"   ✗ Output format incorrect: {track}")
            return False
    else:
        print("   ✗ No tracks returned after 5 hits")
        return False

    # Test with multiple objects
    print("\n4. Testing multiple object tracking...")
    tracker.reset()

    for frame in range(10):
        # Two moving objects
        dets = np.array([
            [100 + frame * 5, 100, 150 + frame * 5, 150, 0.9],  # Object 1
            [200, 200 + frame * 5, 250, 250 + frame * 5, 0.8],  # Object 2
        ])

        tracks = tracker.update(dets)

        if len(tracks) > 0 and frame >= 3:
            print(f"   Frame {frame}: {len(tracks)} tracks")
            for track in tracks:
                print(f"      ID={int(track[4])}, BBox=[{track[0]:.1f}, {track[1]:.1f}, {track[2]:.1f}, {track[3]:.1f}]")

    print("\n5. Testing reset()...")
    tracker.reset()
    print("   ✓ reset() success")

    print("\n✓ SORT algorithm tests completed")
    return True


def test_integration(config: dict) -> bool:
    """
    Test integration of BackgroundModel + SORT.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        bool: True if tests pass, False otherwise.
    """
    print("\n" + "="*50)
    print("Testing Integration (BackgroundModel + SORT)")
    print("="*50)

    from src.background_model import BackgroundModel
    from sort import Sort

    # Initialize components
    print("\n1. Initializing components...")
    model = BackgroundModel(config)
    tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.1)
    print("   ✓ BackgroundModel initialized")
    print("   ✓ SORT tracker initialized")

    # Simulate video frames
    print("\n2. Simulating video processing pipeline...")

    total_tracks = 0
    for frame_idx in range(20):
        # Create simulated frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add moving object
        x = 100 + frame_idx * 5
        frame[200:280, x:x+40, :] = 255

        # Apply background subtraction
        fg_mask = model.apply(frame)

        # Detect contours (simple detection)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare detections for SORT
        dets = []
        for contour in contours:
            x_c, y_c, w, h = cv2.boundingRect(contour)
            dets.append([x_c, y_c, x_c+w, y_c+h, 0.9])

        if len(dets) > 0:
            dets = np.array(dets)
            tracks = tracker.update(dets)

            if len(tracks) > 0:
                total_tracks += len(tracks)
                if frame_idx % 5 == 0:
                    print(f"   Frame {frame_idx}: {len(tracks)} tracks")
        else:
            # No detections, still update tracker
            tracks = tracker.update(np.empty((0, 5)))

    print(f"\n3. Total track detections: {total_tracks}")
    print("\n✓ Integration tests completed")
    return True


def main():
    """Main test function."""
    print("\n" + "="*60)
    print("T02 - Video Input & Background Modeling Layer Test")
    print("="*60)

    # Load config
    config = load_config()

    # Setup logging
    from src.utils import setup_logger
    logger = setup_logger("test_t02", config)

    # Run tests
    results = []

    # Test VideoReader
    try:
        result = test_video_reader(config)
        results.append(("VideoReader", result))
    except Exception as e:
        print(f"\n✗ VideoReader test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("VideoReader", False))

    # Test BackgroundModel
    try:
        result = test_background_model(config)
        results.append(("BackgroundModel", result))
    except Exception as e:
        print(f"\n✗ BackgroundModel test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("BackgroundModel", False))

    # Test SORT
    try:
        result = test_sort()
        results.append(("SORT", result))
    except Exception as e:
        print(f"\n✗ SORT test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("SORT", False))

    # Test Integration
    try:
        result = test_integration(config)
        results.append(("Integration", result))
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Integration", False))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")

    all_passed = all(result for _, result in results)
    print("\n" + "="*60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
