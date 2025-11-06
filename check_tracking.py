#!/usr/bin/env python3
"""
Quick script to check live tracking progress
Run this in another terminal while analysis is running
"""

import os
import time
import glob
from pathlib import Path

def check_tracking():
    """Check current tracking status"""
    output_dir = Path('output_frames')
    
    if not output_dir.exists():
        print("⏳ Waiting for analysis to start...")
        return
    
    # Count frames
    frames = sorted(glob.glob(str(output_dir / 'frame_*.jpg')))
    frame_count = len(frames)
    
    # Check preview
    preview = output_dir / 'LIVE_PREVIEW.jpg'
    preview_exists = preview.exists()
    
    if preview_exists:
        mod_time = preview.stat().st_mtime
        age = time.time() - mod_time
        age_str = f"{age:.1f}s ago" if age < 60 else f"{age/60:.1f}min ago"
    else:
        age_str = "not yet"
    
    print("\n" + "="*60)
    print("📊 LIVE TRACKING STATUS")
    print("="*60)
    print(f"📁 Frames processed: {frame_count}")
    print(f"🖼️  Live preview: {'✅ Available' if preview_exists else '⏳ Not ready'} ({age_str})")
    
    if frames:
        latest = frames[-1]
        latest_time = Path(latest).stat().st_mtime
        latest_age = time.time() - latest_time
        print(f"🕐 Latest frame: {Path(latest).name} ({latest_age:.1f}s ago)")
    
    # Show preview info
    if preview_exists:
        print("\n" + "-"*60)
        print("🖼️  TO VIEW LIVE PREVIEW:")
        print("-"*60)
        print(f"   File: {preview}")
        print(f"   Size: {preview.stat().st_size / 1024:.1f} KB")
        print(f"   Updated: {age_str}")
        print("\n   Download it or use:")
        print(f"   eog {preview}")
        print(f"   # Or download via SCP:")
        print(f"   scp user@host:{preview.absolute()} ./")
    
    # Check for results
    if Path('detailed_football_results.csv').exists():
        print("\n" + "-"*60)
        print("✅ Results file found!")
        print("-"*60)
        with open('detailed_football_results.csv', 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                print(f"   Passes detected: {len(lines) - 1}")
                print(f"   Latest pass: {lines[-1].strip()}")
    
    if Path('tracking_output.mp4').exists():
        print("\n" + "-"*60)
        print("🎬 Tracking video ready!")
        print("-"*60)
        size_mb = Path('tracking_output.mp4').stat().st_size / (1024*1024)
        print(f"   File: tracking_output.mp4")
        print(f"   Size: {size_mb:.1f} MB")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        while True:
            os.system('clear')
            check_tracking()
            print("\n🔄 Refreshing in 3 seconds... (Ctrl+C to stop)")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n\n👋 Stopped watching")

