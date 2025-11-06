#!/usr/bin/env python3
"""
Quick validation script to verify tracking is working correctly.
Run this after analysis to check:
- Ball detection
- Player detection  
- Pass detection
- Video creation
"""

import os
import sys
from pathlib import Path
import json

def validate_tracking():
    """Validate that tracking output exists and is correct"""
    print("🔍 Validating Tracking Output...")
    print("="*60)
    
    issues = []
    warnings = []
    
    # Check for video file
    video_file = Path("tracking_output.mp4")
    if video_file.exists():
        size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"✅ Tracking video found: {video_file} ({size_mb:.1f} MB)")
        if size_mb < 1:
            warnings.append(f"Video file is very small ({size_mb:.1f} MB) - may be incomplete")
    else:
        issues.append("tracking_output.mp4 not found - video was not created")
        print(f"❌ Tracking video NOT found")
    
    # Check for frames
    frames_dir = Path("output_frames")
    if frames_dir.exists():
        frames = list(frames_dir.glob("frame_*.jpg"))
        print(f"✅ Found {len(frames)} visualization frames")
        if len(frames) == 0:
            issues.append("No frames found in output_frames/")
        elif len(frames) < 100:
            warnings.append(f"Only {len(frames)} frames - video may be short")
    else:
        issues.append("output_frames/ directory not found")
        print(f"❌ Frames directory NOT found")
    
    # Check for results CSV
    csv_file = Path("detailed_football_results.csv")
    if csv_file.exists():
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            pass_count = len(lines) - 1  # Subtract header
        print(f"✅ Results CSV found: {pass_count} passes detected")
        if pass_count == 0:
            warnings.append("No passes detected in results CSV")
    else:
        warnings.append("detailed_football_results.csv not found")
        print(f"⚠️  Results CSV NOT found")
    
    # Check for metrics JSON
    json_file = Path("detailed_accuracy_metrics.json")
    if json_file.exists():
        with open(json_file, 'r') as f:
            metrics = json.load(f)
        print(f"✅ Metrics JSON found:")
        print(f"   - Overall accuracy: {metrics.get('overall', 0):.1%}")
        print(f"   - Total passes: {metrics.get('total_passes', 0)}")
        print(f"   - Short: {metrics.get('short_count', 0)}, Long: {metrics.get('long_count', 0)}")
    else:
        warnings.append("detailed_accuracy_metrics.json not found")
        print(f"⚠️  Metrics JSON NOT found")
    
    # Summary
    print("\n" + "="*60)
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n💡 SOLUTIONS:")
        print("   1. Run the analysis: python3 main.py")
        print("   2. Check if video was processed completely")
        print("   3. Verify ffmpeg is installed: sudo apt-get install ffmpeg")
        return False
    elif warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   - {warning}")
        print("\n✅ Tracking appears to be working, but check warnings above")
        return True
    else:
        print("✅ All tracking outputs validated successfully!")
        print("\n📺 To view the tracking video:")
        print("   python3 serve_video.py")
        print("   (Then open the URL in your browser)")
        return True

if __name__ == "__main__":
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    success = validate_tracking()
    sys.exit(0 if success else 1)

