# Football Video Analysis - Complete Tracking System

AI-powered football match analysis with player tracking, ball tracking, and pass detection.

---

## 🚀 Quick Start

### 1. Run Full Analysis

```bash
cd /home/essashah10/phase1-SMO-
source venv/bin/activate
python3 main.py
```

This will:
- Analyze the default match video
- Track all players and ball
- Detect all passes (short/long)
- Create `tracking_output.mp4` with visual overlays
- Generate CSV results and statistics

### 2. Watch the Tracking Video

**Option A: Stream to browser**
```bash
python3 serve_video.py
```
Then open the URL shown (e.g., `http://10.128.0.3:8000`)

**Option B: Download**
```bash
scp user@host:/home/essashah10/phase1-SMO-/tracking_output.mp4 ./
```

### 3. Check Results
```bash
cat detailed_football_results.csv          # All passes
cat detailed_accuracy_metrics.json         # Statistics
```

---

## 📁 Project Structure

```
phase1-SMO-/
├── main.py                          # Main script
├── requirements.txt                 # Dependencies
├── src/
│   ├── classes/                     # Analysis classes
│   │   ├── colab_cell_5_class.py   # Fixed class
│   │   ├── colab_cell_5_class_A_PLUS.py
│   │   └── colab_cell_5_class_BALANCED.py
│   └── validation/                  # Validation scripts
├── venv/                            # Virtual environment
└── output_frames/                   # Saved frames (auto-created)
```

---

## 🎯 What You Get

### 1. Tracking Video (`tracking_output.mp4`)
Complete match with overlays:
- **Red boxes** = Team A players
- **Blue boxes** = Team B players
- **Green box** = Ball tracking
- **Purple arrow** = Ball velocity
- **Yellow lines** = Pass connections
- **Labels** = SHORT/LONG passes

### 2. Pass Data (`detailed_football_results.csv`)
Every detected pass:
- Passer → Receiver
- Pass type (short/long)
- Success/failure
- Distance, confidence, frame number

### 3. Statistics (`detailed_accuracy_metrics.json`)
- Overall accuracy
- Short/long pass breakdown
- Success rates

---

## 🎬 Usage Options

### Analyze Your Own Video
```bash
python3 main.py --video /path/to/your/video.mp4
```

### Use Different Analysis Class
```bash
python3 main.py --class aplus      # More accurate (slower)
python3 main.py --class balanced   # Balanced
python3 main.py --class fixed      # Faster (default)
```

### Use GPU (if available)
```bash
python3 main.py --device cuda
```

### Run Without Display
```bash
python3 main.py --no-show
```

---

## 📊 Tracking Features

### Ball Tracking
- YOLO-based detection
- Kalman filter smoothing
- Velocity tracking
- Confidence threshold: 0.35

### Player Tracking
- Multi-player tracking
- Team assignment (color-based)
- Stable bounding boxes
- Ghost player filtering

### Pass Detection
- Trajectory-based detection
- Short vs Long classification (150px threshold)
- Pass velocity: 3.0 px/frame minimum
- Success/failure determination

---

## 🔧 Requirements

Installed automatically via `requirements.txt`:
- Python 3.11+
- OpenCV (cv2)
- PyTorch
- Ultralytics YOLO
- Pandas, NumPy
- Matplotlib
- yt-dlp (for YouTube videos)
- ffmpeg (for video creation)

---

## 📺 Live Monitoring

### Watch Progress in Real-Time
```bash
# Terminal 1
python3 main.py

# Terminal 2
python3 check_tracking.py
```

Shows:
- Frame count updates
- Ball tracking status
- Player counts
- Pass detections
- Live preview updates

---

## ✅ Verify Accuracy

1. **Watch tracking video** - Visual verification of tracking
2. **Check CSV data** - Compare timestamps with video
3. **Review statistics** - Overall accuracy metrics
4. **Spot check frames** - View individual frames in `output_frames/`

---

## 🐛 Troubleshooting

### Virtual Environment Issues
```bash
# Recreate venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### GPU Not Detected
```bash
# Check GPU
nvidia-smi

# Force CPU
python3 main.py --device cpu
```

### Video Not Created
```bash
# Manual video creation
cd output_frames
ffmpeg -framerate 30 -pattern_type glob -i '*.jpg' \
       -c:v libx264 -pix_fmt yuv420p ../tracking_output.mp4
```

---

## 📖 Additional Documentation

See `docs/` folder for:
- Technical implementation details
- Pass detection logic
- Team and player statistics
- Troubleshooting guides

---

## 🎯 Default Video

Currently analyzing:
- **APEA FC vs AEP Polemidia Highlights**
- URL: https://www.youtube.com/watch?v=awdBdYZSD1Q&t=2s

Change in `main.py` line 101: `DEFAULT_YOUTUBE_URL`

---

## 📝 Output Files

After running analysis:
```
tracking_output.mp4                  # Full tracking video
detailed_football_results.csv        # All passes
detailed_accuracy_metrics.json       # Statistics
detailed_analysis_plots.png          # Visualization graphs
output_frames/                       # Individual frames
  ├── frame_000001.jpg
  ├── frame_000002.jpg
  ├── ...
  └── LIVE_PREVIEW.jpg              # Latest frame
```

---

## 🚀 Quick Commands

```bash
# Basic run
python3 main.py

# Your video + GPU + live view
python3 main.py --video match.mp4 --device cuda --show

# YouTube video + A+ class
python3 main.py --youtube "URL" --class aplus

# Fast analysis (no validation/viz)
python3 main.py --no-validation --no-viz
```

---

## 📧 Support

For issues or questions:
1. Check error logs in terminal output
2. Review documentation in `docs/`
3. Check frame output in `output_frames/`

---

## ⚙️ Tuned Parameters

- **Ball confidence**: 0.35 (reduced false positives)
- **Pass velocity**: 3.0 px/frame (clear passes only)
- **Pass distance**: 150px (short vs long threshold)
- **Possession radius**: 70px (tight accuracy)

All configurable in `src/classes/colab_cell_5_class.py`
