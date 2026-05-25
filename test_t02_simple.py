"""
Simple Test for T02 - Video Input & Background Modeling Layer

This script performs basic verification of:
1. VideoReader - class import and structure
2. BackgroundModel - KNN background subtraction
3. Sort - multi-object tracking (SORT algorithm)

Author: 寇豆码 (Alex)
Date: 2024
"""

import sys
import numpy as np
import cv2
from pathlib import Path
import yaml


def load_config():
    """Load configuration from config.yaml."""
    project_root = Path(__file__).parent
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        config_path = project_root / "config" / "config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        return {
            'background_model': {
                'history': 7,
                'dist2_threshold': 800.0,
                'detect_shadows': False,
            }
        }


def test_imports():
    """Test all module imports."""
    print("\n" + "="*60)
    print("Testing Imports")
    print("="*60)
    
    try:
        from src.video_reader import VideoReader
        print("✓ VideoReader imported")
    except Exception as e:
        print(f"✗ VideoReader import failed: {e}")
        return False
    
    try:
        from src.background_model import BackgroundModel
        print("✓ BackgroundModel imported")
    except Exception as e:
        print(f"✗ BackgroundModel import failed: {e}")
        return False
    
    try:
        from sort import Sort, KalmanBoxTracker
        print("✓ SORT imported")
    except Exception as e:
        print(f"✗ SORT import failed: {e}")
        return False
    
    return True


def test_background_model(config):
    """Test BackgroundModel functionality."""
    print("\n" + "="*60)
    print("Testing BackgroundModel")
    print("="*60)
    
    from src.background_model import BackgroundModel
    
    # Initialize
    print("\n1. Initializing BackgroundModel...")
    model = BackgroundModel(config)
    print(f"   ✓ Model initialized")
    
    # Test apply
    print("\n2. Testing apply()...")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mask = model.apply(frame)
    print(f"   ✓ apply() success: shape={mask.shape}")
    print(f"   ✓ Unique values: {np.unique(mask)}")
    
    # Test reset
    print("\n3. Testing reset()...")
    model.reset()
    print("   ✓ reset() success")
    
    print("\n✓ BackgroundModel tests PASSED")
    return True


def test_sort():
    """Test SORT tracker functionality."""
    print("\n" + "="*60)
    print("Testing SORT Tracker")
    print("="*60)
    
    from sort import Sort
    
    # Initialize
    print("\n1. Initializing SORT tracker...")
    tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.1)
    print(f"   ✓ Tracker initialized")
    
    # Test tracking
    print("\n2. Simulating tracking...")
    for frame in range(10):
        x1 = 100 + frame * 10
        y1 = 100 + frame * 5
        x2 = x1 + 50
        y2 = y1 + 50
        
        dets = np.array([[x1, y1, x2, y2, 0.9]])
        tracks = tracker.update(dets)
        
        if len(tracks) > 0:
            print(f"   Frame {frame}: Track ID = {int(tracks[0][4])}")
    
    # Test reset
    print("\n3. Testing reset()...")
    tracker.reset()
    print("   ✓ reset() success")
    
    print("\n✓ SORT tracker tests PASSED")
    return True


def test_integration(config):
    """Test integration of BackgroundModel + SORT."""
    print("\n" + "="*60)
    print("Testing Integration (BackgroundModel + SORT)")
    print("="*60)
    
    from src.background_model import BackgroundModel
    from sort import Sort
    
    # Initialize
    print("\n1. Initializing components...")
    model = BackgroundModel(config)
    tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.1)
    print("   ✓ Components initialized")
    
    # Simulate processing
    print("\n2. Simulating video processing...")
    for frame_idx in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = 100 + frame_idx * 5
        frame[200:280, x:x+40, :] = 255
        
        # Background subtraction
        fg_mask = model.apply(frame)
        
        # Detect contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Prepare detections
        dets = []
        for contour in contours:
            x_c, y_c, w, h = cv2.boundingRect(contour)
            dets.append([x_c, y_c, x_c+w, y_c+h, 0.9])
        
        if len(dets) > 0:
            dets = np.array(dets)
            tracks = tracker.update(dets)
            if len(tracks) > 0:
                print(f"   Frame {frame_idx}: {len(tracks)} track(s)")
        else:
            tracker.update(np.empty((0, 5)))
    
    print("\n✓ Integration tests PASSED")
    return True


def main():
    """Main test function."""
    print("\n" + "="*60)
    print("T02 - Video Input & Background Modeling Layer Test (Simple)")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Run tests
    results = []
    
    try:
        result = test_imports()
        results.append(("Imports", result))
    except Exception as e:
        print(f"\n✗ Import test failed: {e}")
        results.append(("Imports", False))
    
    try:
        result = test_background_model(config)
        results.append(("BackgroundModel", result))
    except Exception as e:
        print(f"\n✗ BackgroundModel test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("BackgroundModel", False))
    
    try:
        result = test_sort()
        results.append(("SORT", result))
    except Exception as e:
        print(f"\n✗ SORT test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("SORT", False))
    
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
        print(f"  {name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "="*60)
    if all_passed:
        print("Overall: ALL TESTS PASSED!")
    else:
        print("Overall: SOME TESTS FAILED!")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
