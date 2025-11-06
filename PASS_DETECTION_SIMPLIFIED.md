# Simplified Pass Detection System

## What Changed

### Old System (Complex, Many False Negatives)
- Checked every 3rd frame only
- Required ball to move >20px between frames
- Required strict alignment between passer/receiver and ball positions
- Had cooldowns that blocked valid passes
- Only detected 1 pass because of overly strict validation

### New System (Simple, Possession-Based)
- **Tracks possession every frame**: Finds which player is closest to ball (<80px)
- **Detects possession changes**: When ball moves from Player A → Player B = pass
- **Simple distance classification**:
  - Short pass: < 15 meters
  - Long pass: ≥ 15 meters
- **Same-team validation**: Only counts passes between teammates
- **Small cooldown**: 15 frames (0.5 seconds) to avoid duplicate detections
- **Range validation**: Pass distance must be 3m - 50m to be realistic

## How It Works

1. **Every Frame**:
   - Find the player closest to the ball
   - If within 80 pixels → that player has possession

2. **When Possession Changes**:
   - Previous owner = Passer
   - New owner = Receiver
   - Calculate distance between them
   - Validate: same team, reasonable distance, cooldown elapsed

3. **Output**:
   ```
   [PASS DETECTED] Team A | Passer ID: 3 → Receiver ID: 5 | Type: Short | Distance: 10.8m
   ```

## Advantages

✅ **Simple and reliable**: No complex trajectory calculations
✅ **Catches all passes**: Checks every frame, not just every 3rd
✅ **Clear possession tracking**: Always know who has the ball
✅ **Debug-friendly**: Shows possession every 50 frames
✅ **No false positives from movement**: Only counts actual possession changes

## How to Run

```bash
cd "/Users/essashah/Desktop/SWE/SMO analysis/phase1-SMO-"

# With video window (see detections live)
python main.py

# Without video (faster, console output only)
python main.py --no-show

# Use your own video
python main.py --video path/to/video.mp4
```

## What You'll See

- Console output showing:
  - Possession changes every 50 frames
  - Pass detections with team, player IDs, type, and distance
  - Progress updates every 100 frames
  
- Video window showing (if enabled):
  - Players with colored boxes (Red=Team A, Blue=Team B)
  - Ball as yellow circle
  - Pass lines (green=successful)
  - Real-time info overlay

## Debug Output

Every 50 frames:
```
[Frame 150] Possession: Player 3 (dist: 45.2px)
```

When pass detected:
```
[PASS DETECTED] Team A | Passer ID: 3 → Receiver ID: 7 | Type: Short | Distance: 12.3m
```

## Technical Details

- **Possession threshold**: 80 pixels (adjustable)
- **Pass distance range**: 3-50 meters
- **Short/Long threshold**: 15 meters
- **Cooldown**: 15 frames (~0.5 seconds at 30fps)
- **Pixel-to-meter conversion**: ~10 pixels = 1 meter (rough estimate)

## Troubleshooting

If no passes detected:
1. Check if ball is being tracked: Look for "Ball tracked: True" in debug output
2. Check if players are assigned teams: Look for "Teams assigned by color" message
3. Verify possession is changing: Look for possession debug output every 50 frames
4. Check distance threshold: Passes must be 3-50 meters (30-500 pixels)

## Files Modified

- `src/classes/colab_cell_5_class.py`:
  - Simplified `detect_passes()` method
  - Added possession tracking variables
  - Added debug output for possession changes
  - Removed complex ball trajectory and confidence checks

