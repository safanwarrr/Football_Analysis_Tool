#!/usr/bin/env python3
"""
Test script to verify the football analysis system is properly set up.
"""

import sys

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Ultralytics: {e}")
        return False
    
    try:
        import supervision as sv
        print("✓ Supervision imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Supervision: {e}")
        return False
    
    try:
        from sklearn.cluster import KMeans
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Scikit-learn: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Pandas: {e}")
        return False
    
    return True

def test_modules():
    """Test if custom modules can be imported."""
    print("\nTesting custom modules...")
    
    try:
        from utils.video_utils import read_video, save_video
        print("✓ Video utils imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import video utils: {e}")
        return False
    
    try:
        from trackers.tracker import Tracker
        print("✓ Tracker imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Tracker: {e}")
        return False
    
    try:
        from team_assignment.team_assigner import TeamAssigner
        print("✓ Team assigner imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TeamAssigner: {e}")
        return False
    
    try:
        from camera_movement.camera_movement_estimator import CameraMovementEstimator
        print("✓ Camera movement estimator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CameraMovementEstimator: {e}")
        return False
    
    try:
        from perspective.view_transformer import ViewTransformer
        print("✓ View transformer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ViewTransformer: {e}")
        return False
    
    return True

def test_video_file():
    """Test if the input video file exists."""
    print("\nTesting video file...")
    import os
    
    video_path = '/Users/safanwar/Documents/Untitled.mp4'
    if os.path.exists(video_path):
        print(f"✓ Video file found: {video_path}")
        # Get file size
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ Video file not found: {video_path}")
        return False

def test_directories():
    """Test if required directories exist."""
    print("\nTesting directories...")
    import os
    
    dirs = ['models', 'output', 'stubs']
    all_exist = True
    
    for d in dirs:
        if os.path.exists(d):
            print(f"✓ Directory exists: {d}/")
        else:
            print(f"✗ Directory missing: {d}/")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("=" * 60)
    print("Football Analysis System - Setup Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Custom Modules", test_modules()))
    results.append(("Video File", test_video_file()))
    results.append(("Directories", test_directories()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nRun the analysis with: python main.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
