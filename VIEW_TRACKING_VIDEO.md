# 🎥 How to Watch the Full Tracking Video

After running the analysis, you'll have `tracking_output.mp4` - the complete match with all tracking overlays.

---

## 🚀 QUICKEST WAY: Stream to Browser

### Step 1: Run the video server
```bash
cd /home/essashah10/phase1-SMO-
python3 serve_video.py
```

### Step 2: Open in your browser
The server will show you the URL, something like:
```
http://10.128.0.3:8000
```

Open that URL in your browser and watch the full tracking video with controls!

---

## 📥 Method 2: Download and Watch Locally

### On your LOCAL computer, run:
```bash
# Download the tracking video
scp user@host:/home/essashah10/phase1-SMO-/tracking_output.mp4 ./

# Watch it
# Windows: double-click the file
# Mac: open tracking_output.mp4
# Linux: vlc tracking_output.mp4
```

---

## 🖥️ Method 3: Watch on Server (if X11 available)

```bash
cd /home/essashah10/phase1-SMO-

# Using VLC
vlc tracking_output.mp4

# Using mpv
mpv tracking_output.mp4

# Using ffplay
ffplay tracking_output.mp4
```

---

## 🎬 What You'll See in the Video

The tracking video shows:

### Player Tracking:
- **Red boxes** = Team A players (with player IDs)
- **Blue boxes** = Team B players (with player IDs)
- Boxes follow players throughout the match

### Ball Tracking:
- **Green box** = Ball position
- **Purple arrow** = Ball velocity/direction
- Shows ball movement and trajectory

### Pass Detection:
- **Yellow lines** = Pass connections (passer → receiver)
- **Text labels** = "SHORT PASS" or "LONG PASS"
- **Success indicators** = Pass completion status

### Stats Overlay:
- Frame counter
- Players detected count
- Ball tracking status
- Pass statistics

---

## ✅ Verify Pass Accuracy

Watch the video and check:

1. **Ball tracking**: Is the green box following the ball correctly?
2. **Player tracking**: Are red/blue boxes stable on players?
3. **Pass detection**: Do yellow lines appear when passes happen?
4. **Pass classification**: Are SHORT vs LONG passes correct?
5. **Pass success**: Do success/failure labels match what happened?

---

## 📊 Cross-Reference with CSV Data

While watching the video:

```bash
# View all detected passes
cat detailed_football_results.csv

# Count passes
tail -n +2 detailed_football_results.csv | wc -l

# See pass breakdown
cat detailed_accuracy_metrics.json
```

Match the video timestamps with the CSV data to verify accuracy.

---

## 🔍 Detailed Review

### Frame-by-Frame Analysis:
If you spot an issue at a specific time:

1. Note the timestamp (e.g., 2:35 = 155 seconds)
2. Calculate frame: `155 * FPS` (usually 30 FPS = frame 4650)
3. Check that frame: `eog output_frames/frame_004650.jpg`

### Compare Multiple Frames:
```bash
# View frames around a specific pass
ls output_frames/frame_00465*.jpg
```

---

## 🎯 Quality Checks

### Good Tracking Signs:
- ✅ Boxes stay locked on players
- ✅ Ball tracking is smooth
- ✅ Pass lines appear at right moments
- ✅ Classification matches reality

### Bad Tracking Signs:
- ❌ Boxes jump between players
- ❌ Ball tracking lost frequently
- ❌ Pass lines appear randomly
- ❌ Short/long classification wrong

---

## 🔧 If Video Quality Issues

### Create Higher Quality Video:
```bash
cd output_frames
ffmpeg -framerate 30 -pattern_type glob -i '*.jpg' \
       -c:v libx264 -preset slow -crf 18 \
       -pix_fmt yuv420p ../tracking_output_hq.mp4
```

### Create Slow-Motion Version:
```bash
ffmpeg -i tracking_output.mp4 -filter:v "setpts=2.0*PTS" tracking_output_slowmo.mp4
```

---

## 📱 Share for Review

The tracking video is perfect for:
- Showing to coaches
- Team analysis meetings
- Validating AI accuracy
- Demonstrating the system

Just share `tracking_output.mp4` - it's a standard MP4 file that plays anywhere.

---

## 💡 Pro Tips

1. **Watch at different speeds** - Use 0.5x to see tracking details
2. **Pause frequently** - Check box positions when paused
3. **Focus on specific players** - Watch individual player tracking
4. **Note timestamps** - Mark any issues you see
5. **Compare with original** - Watch original video side-by-side

---

## 🎮 Video Player Shortcuts

Most video players support:
- **Space** - Play/Pause
- **← →** - Jump 5 seconds
- **↑ ↓** - Volume
- **F** - Fullscreen
- **J K L** - Rewind/Pause/Forward

---

That's it! Watch the full tracking video to verify everything is working correctly.

