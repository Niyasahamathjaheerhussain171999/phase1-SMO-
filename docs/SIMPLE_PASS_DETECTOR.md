# Simple Pass Detector - Clean & Minimal

A clean, minimal football pass detection script that focuses on core functionality without overengineering.

## Features

✅ **Player Detection** - YOLO object detection  
✅ **Ball Tracking** - Tracks ball position across frames  
✅ **Team Assignment** - Jersey color-based or left/right split  
✅ **Pass Detection** - Short (<120px) and Long (≥120px) passes  
✅ **Success/Failure** - Determines if receiver gains possession  
✅ **Real-time Visualization** - Shows video with detections  

## Usage

```bash
# Use default video
python simple_pass_detector.py

# Specify video path
python simple_pass_detector.py path/to/video.mp4
```

## How It Works

### 1. Player Detection
- Uses YOLO to detect players (class 0)
- Filters by size (1000-50000 pixels area)
- Confidence threshold: 0.3

### 2. Team Assignment
- **Primary**: Jersey color (HSV hue from upper body)
- **Fallback**: Left/right field split
- Learns team colors from first few frames

### 3. Player Tracking
- Simple proximity-based ID assignment
- Max distance: 50 pixels to match
- Creates new IDs for unmatched players

### 4. Ball Tracking
- Detects ball using YOLO (class 32)
- Maintains history (last 10 frames)
- Keeps last position if not detected

### 5. Pass Detection
- Requires ball movement (≥30 pixels over 5 frames)
- Finds passer (closest to old ball position)
- Finds receiver (closest to current ball position)
- Validates:
  - Same team only
  - Distance 40-350 pixels
  - Cooldown (30 frames between same pair)

### 6. Pass Classification
- **Short pass**: < 120 pixels
- **Long pass**: ≥ 120 pixels
- **Success**: Receiver within 50px of ball
- **Failure**: Receiver not close enough

## Visualization

The script shows a video window with:
- **Players**: Colored bounding boxes (Red=Team A, Blue=Team B)
- **Player IDs**: P1, P2, etc. with team name
- **Ball**: Yellow circle
- **Passes**: Green line (success) or Red line (failure)
- **Info**: Frame number, player count, ball status, total passes

## Output

### Per Frame:
- Players detected with IDs and teams
- Ball position
- Pass events (if any)

### Console:
- Pass detection messages: `⚽ PASS SHORT (Team A): P1 → P2 (95px, ✓)`
- Progress updates every 100 frames
- Final summary with pass counts

## Key Design Decisions

1. **Simple Tracking**: No fancy smoothing - just proximity matching
2. **Extra IDs OK**: If player temporarily disappears, new ID is fine
3. **Frame-by-Frame**: No complex state machines
4. **Minimal Dependencies**: Just YOLO, OpenCV, NumPy

## Requirements

```bash
pip install ultralytics opencv-python numpy torch
```

## Controls

- **Press 'Q'** to quit video visualization
- Video will continue processing if you close the window

## Example Output

```
⚽ Simple Football Pass Detection
==================================================
🔧 Using device: cpu
📹 Video: 4500 frames at 30 FPS
⏱️ Processing...
✅ Teams assigned by color: Team A (hue=120.5), Team B (hue=15.3)
⚽ PASS SHORT (Team A): P1 → P3 (95px, ✓)
⚽ PASS LONG (Team A): P3 → P5 (180px, ✓)
⚽ PASS SHORT (Team B): P7 → P9 (88px, ✗)
⏳ Processed 100/4500 frames (2.2%)
...
✅ Analysis Complete!
📊 Total passes detected: 45

📈 Pass Summary:
  short_success: 28
  short_failed: 5
  long_success: 10
  long_failed: 2
```

---

**That's it!** Clean, simple, and focused on the essentials. 🚀

