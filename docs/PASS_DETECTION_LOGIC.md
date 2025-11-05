# Pass Detection Logic - Simple Explanation

## Overview
The system detects passes by tracking the ball and players, then checking if the ball moves from one player to another.

## Method 1: Ball-Based Detection (Primary - Most Accurate)

### Step-by-Step Process:

1. **Check if we have enough data**
   - Need at least 2 players detected
   - Need ball to be tracked
   - Need 12 frames of ball history

2. **Check every 5 frames** (not every frame to avoid spam)

3. **Is the ball moving?**
   - Ball speed must be > 2.5 pixels/frame
   - Ball detection confidence must be > 30%
   - If not → NOT a pass

4. **Look back 12 frames** (0.4 seconds at 30fps)
   - Get ball position 12 frames ago
   - Get ball position now
   - Calculate how far ball moved

5. **Ball must have moved at least 40 pixels**
   - If ball barely moved → NOT a pass (maybe just player dribbling)

6. **Find the PASSER** (who had the ball)
   - Look at all players 12 frames ago
   - Find which player was closest to the OLD ball position
   - That player = PASSER
   - Passer must be within 50 pixels of old ball position

7. **Find the RECEIVER** (who has the ball now)
   - Look at all players NOW
   - Find which player is closest to the CURRENT ball position
   - That player = RECEIVER
   - Receiver must be within 50 pixels of current ball position

8. **Basic validation**
   - Passer and receiver must be DIFFERENT players
   - Check cooldown (don't detect same pass twice within 50 frames)

9. **Calculate pass distance** (player-to-player distance)
   - Must be between 50-300 pixels
   - Too close = just dribbling
   - Too far = not a realistic pass

10. **CRITICAL: Check ball trajectory alignment**
    - Calculate direction ball is moving (from old position to new position)
    - Calculate direction from passer to receiver
    - Check if these directions align (at least 50% same direction)
    - If ball is moving AWAY from receiver → NOT a pass

11. **Additional checks**
    - Ball must be close to passer at start (< 60 pixels)
    - Ball must be close to receiver at end (< 50 pixels)

12. **Calculate confidence score**
    - Ball detection confidence (40%)
    - Trajectory alignment (30%)
    - Ball speed (20%)
    - Player detection confidence (10%)
    - Total confidence must be > 60% to accept

13. **If all checks pass → IT'S A PASS!**
    - Record the pass
    - Mark it as short (< 120px) or long (>= 120px)
    - Mark as successful if ball reached receiver

---

## Method 2: Movement-Based Detection (Fallback - Less Accurate)

### When does it run?
- ONLY when ball tracking is NOT available (ball not detected)
- This is a fallback method

### How does it work?

1. **Check every 15 frames** (less frequent)

2. **Look for player movement patterns**
   - Player 1 was moving fast
   - Player 1 slowed down (deceleration = passed ball)
   - Player 2 is moving toward Player 1
   - Players were far apart, now closer together

3. **Multiple strict checks**
   - Players must be 50-300 pixels apart
   - Players were at least 100 pixels apart before
   - Players got at least 25 pixels closer
   - Both players must be moving (> 1.5 px/frame)
   - Player 1's movement direction must align 40% toward Player 2

4. **If all checks pass → Possible pass**
   - Lower confidence (since no ball tracking)
   - Only accepted if confidence > 65%

---

## Key Concepts Explained

### 1. **Trajectory Alignment**
Imagine a compass:
- Ball is moving NORTH
- Receiver is EAST of passer
- If ball moves NORTH-EAST (45°), that's 50% aligned
- If ball moves NORTH-WEST, that's 0% aligned (wrong direction!)

**Formula**: Dot product of two direction vectors
- 1.0 = Perfectly aligned (same direction)
- 0.5 = 50% aligned (45° angle)
- 0.0 = Perpendicular (90° angle)
- -1.0 = Opposite direction

### 2. **Cooldown System**
- Prevents detecting the same pass multiple times
- Once a pass is detected between Player A → Player B
- Wait 50 frames before detecting another pass between them
- This prevents counting one pass as 5 passes

### 3. **Confidence Scoring**
Multiple factors determine how confident we are:
- **Ball confidence (40%)**: How sure are we we detected the ball?
- **Alignment (30%)**: Does ball move toward receiver?
- **Speed (20%)**: Is ball moving fast enough?
- **Player confidence (10%)**: How sure are we about player detection?

Higher confidence = More likely to be a real pass

---

## Visual Example

```
Frame 1 (12 frames ago):
    P1 (passer)  ●  ← Ball here
    P2 (receiver)
    
Frame 13 (now):
    P1 (passer)
    P2 (receiver)  ●  ← Ball here
    
✅ PASS DETECTED!
- Ball moved from P1 to P2
- Ball trajectory aligned with P1→P2 direction
- Distance: 150 pixels (short pass)
```

---

## Why So Many Checks?

Each check filters out false positives:
- **Ball speed check**: Filters out slow dribbling
- **Distance check**: Filters out players just standing close
- **Trajectory check**: Filters out ball moving in wrong direction
- **Proximity check**: Filters out ball far from players
- **Cooldown check**: Filters out duplicate detections
- **Confidence check**: Only keeps high-quality detections

The goal is: **Only count ACTUAL passes, not noise!**

---

## Parameters (Current - Balanced for Detection)

- `pass_cooldown = 30`: Frames between pass detections (1 second at 30fps)
- `min_pass_distance = 30`: Minimum pass distance (pixels)
- `max_pass_distance = 400`: Maximum pass distance (pixels)
- `ball_speed_threshold = 1.5`: Minimum ball speed for pass (px/frame)
- `ball_conf_threshold = 0.2`: Minimum ball detection confidence
- `alignment_threshold = 0.3`: Minimum trajectory alignment (30%)
- `confidence_threshold = 0.4`: Minimum confidence to accept pass
- `check_interval = 3`: Check every 3 frames
- `history_lookback = 8`: Look back 8 frames (0.27 seconds)
- `proximity_threshold = 80`: Player must be within 80px of ball

**These parameters are now BALANCED** - relaxed enough to detect passes but still filtered to reduce false positives. You can adjust these if passes are being missed or false positives are detected.

### Debug Output
The system now includes extensive debug output:
- Every 100 frames: Shows why passes are rejected (ball speed, ball travel, distance, alignment)
- Every 50 frames: Shows low-confidence passes that were rejected
- Real-time: Shows when passes are detected with full metrics

### Visualization Enhancements
- Ball status shows speed and confidence
- Ball history requirement (8 frames) is displayed
- Pass detection readiness indicator (✓ or ❌)
- Color-coded ball (Yellow=high conf, Orange=medium, Red=predicted)

