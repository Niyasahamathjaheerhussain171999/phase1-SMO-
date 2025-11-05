"""
SIMPLE PASS DETECTION - NO BULLSHIT
Just detects passes (long and short) accurately
"""

# Fix OpenCV import issue
import sys
import types

# Create mock cv2.dnn module with DictValue before cv2 loads
if 'cv2.dnn' not in sys.modules:
    mock_dnn = types.ModuleType('cv2.dnn')
    mock_dnn.DictValue = type('DictValue', (), {})
    sys.modules['cv2.dnn'] = mock_dnn

# Now import cv2
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import yt_dlp

# Load model
print("Loading YOLO model...")
model = YOLO('yolov8x.pt')

# Download video
def download_video(url):
    print(f"Downloading video from YouTube...")
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': 'football_video.mp4',
        'quiet': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return 'football_video.mp4'

# Simple pass detection parameters
MIN_PASS_DISTANCE = 50  # pixels
MAX_PASS_DISTANCE = 400  # pixels
SHORT_PASS_THRESHOLD = 150  # pixels
PASS_COOLDOWN = 30  # frames
BALL_HISTORY_SIZE = 10  # frames to look back

# Tracking
ball_history = deque(maxlen=BALL_HISTORY_SIZE)
player_history = {}
last_pass_frame = {}
passes = []

def detect_ball(frame, results):
    """Detect ball from YOLO results"""
    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Class 32 = sports ball
            if cls == 32 and conf > 0.2:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                
                # Size validation
                if 8 < w < 100 and 8 < h < 100:
                    return {
                        'x': (x1 + x2) / 2,
                        'y': (y1 + y2) / 2,
                        'conf': conf
                    }
    return None

def detect_players(frame, results):
    """Detect players from YOLO results"""
    players = []
    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Class 0 = person
            if cls == 0 and conf > 0.3:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                # Filter for players (not spectators, refs, etc)
                if 1000 < area < 50000:
                    players.append({
                        'x': (x1 + x2) / 2,
                        'y': (y1 + y2) / 2,
                        'conf': conf,
                        'id': None  # Will assign later
                    })
    return players

def assign_player_ids(players, frame_num):
    """Simple player ID assignment"""
    global player_history
    
    # First frame - just assign IDs
    if not player_history:
        for i, player in enumerate(players):
            player['id'] = i + 1
            player_history[player['id']] = deque(maxlen=BALL_HISTORY_SIZE)
            player_history[player['id']].append({
                'frame': frame_num,
                'x': player['x'],
                'y': player['y']
            })
        return
    
    # Match to existing players by proximity
    used_ids = set()
    for player in players:
        min_dist = float('inf')
        best_id = None
        
        for pid, history in player_history.items():
            if history and pid not in used_ids:
                last_pos = history[-1]
                dist = np.sqrt((player['x'] - last_pos['x'])**2 + 
                              (player['y'] - last_pos['y'])**2)
                if dist < min_dist and dist < 100:
                    min_dist = dist
                    best_id = pid
        
        if best_id:
            player['id'] = best_id
            used_ids.add(best_id)
        else:
            # New player
            new_id = max(player_history.keys()) + 1 if player_history else 1
            player['id'] = new_id
        
        # Update history
        if player['id'] not in player_history:
            player_history[player['id']] = deque(maxlen=BALL_HISTORY_SIZE)
        player_history[player['id']].append({
            'frame': frame_num,
            'x': player['x'],
            'y': player['y']
        })

def detect_passes(players, frame_num):
    """Simple, accurate pass detection"""
    global ball_history, last_pass_frame, passes
    
    if len(ball_history) < BALL_HISTORY_SIZE or len(players) < 2:
        return []
    
    # Check every 5 frames
    if frame_num % 5 != 0:
        return []
    
    current_ball = ball_history[-1]
    old_ball = ball_history[0]
    
    # Ball must have moved
    ball_distance = np.sqrt((current_ball['x'] - old_ball['x'])**2 + 
                           (current_ball['y'] - old_ball['y'])**2)
    if ball_distance < 30:
        return []
    
    # Find player closest to old ball position (passer)
    passer = None
    passer_dist = float('inf')
    for player in players:
        if player['id'] and player['id'] in player_history:
            history = player_history[player['id']]
            if len(history) >= BALL_HISTORY_SIZE:
                old_pos = history[0]
                dist = np.sqrt((old_pos['x'] - old_ball['x'])**2 + 
                              (old_pos['y'] - old_ball['y'])**2)
                if dist < passer_dist:
                    passer_dist = dist
                    passer = player
    
    # Find player closest to current ball position (receiver)
    receiver = None
    receiver_dist = float('inf')
    for player in players:
        dist = np.sqrt((player['x'] - current_ball['x'])**2 + 
                      (player['y'] - current_ball['y'])**2)
        if dist < receiver_dist:
            receiver_dist = dist
            receiver = player
    
    # Validation
    if (passer and receiver and 
        passer['id'] != receiver['id'] and
        passer_dist < 80 and receiver_dist < 80):
        
        # Check cooldown
        pair_key = tuple(sorted([passer['id'], receiver['id']]))
        if pair_key in last_pass_frame:
            if frame_num - last_pass_frame[pair_key] < PASS_COOLDOWN:
                return []
        
        # Calculate pass distance
        pass_distance = np.sqrt((receiver['x'] - passer['x'])**2 + 
                               (receiver['y'] - passer['y'])**2)
        
        if MIN_PASS_DISTANCE < pass_distance < MAX_PASS_DISTANCE:
            # Check ball trajectory alignment
            ball_dx = current_ball['x'] - old_ball['x']
            ball_dy = current_ball['y'] - old_ball['y']
            ball_vec = np.array([ball_dx, ball_dy])
            ball_vec_norm = np.linalg.norm(ball_vec)
            
            if ball_vec_norm > 0:
                ball_vec = ball_vec / ball_vec_norm
                
                # Player direction
                player_dx = receiver['x'] - passer['x']
                player_dy = receiver['y'] - passer['y']
                player_vec = np.array([player_dx, player_dy])
                player_vec_norm = np.linalg.norm(player_vec)
                
                if player_vec_norm > 0:
                    player_vec = player_vec / player_vec_norm
                    alignment = np.dot(ball_vec, player_vec)
                    
                    # Must be somewhat aligned (30% min)
                    if alignment > 0.3:
                        pass_type = 'LONG' if pass_distance > SHORT_PASS_THRESHOLD else 'SHORT'
                        
                        pass_data = {
                            'frame': frame_num,
                            'passer_id': passer['id'],
                            'receiver_id': receiver['id'],
                            'distance': pass_distance,
                            'type': pass_type,
                            'alignment': alignment
                        }
                        
                        last_pass_frame[pair_key] = frame_num
                        passes.append(pass_data)
                        
                        print(f"✅ {pass_type} PASS: P{passer['id']} → P{receiver['id']} ({pass_distance:.0f}px, align={alignment:.2f})")
                        return [pass_data]
    
    return []

# Main
print("="*60)
print("SIMPLE PASS DETECTION - ACCURATE")
print("="*60)

# Download video
video_url = "https://www.youtube.com/watch?v=awdBdYZSD1Q"
video_path = download_video(video_url)

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n📹 Video: {total_frames} frames at {fps:.0f} FPS")
print(f"⏱️ Length: {total_frames/fps/60:.1f} minutes\n")
print("Processing...")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO detection
    results = model(frame, device='cpu', verbose=False)
    
    # Detect ball and players
    ball = detect_ball(frame, results)
    players = detect_players(frame, results)
    
    # Track ball
    if ball:
        ball['frame'] = frame_count
        ball_history.append(ball)
    
    # Track players
    if players:
        assign_player_ids(players, frame_count)
        
        # Detect passes
        detect_passes(players, frame_count)
    
    frame_count += 1
    
    if frame_count % 100 == 0:
        progress = frame_count / total_frames * 100
        print(f"⏳ {frame_count}/{total_frames} ({progress:.1f}%) - {len(passes)} passes detected")

cap.release()

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

short_passes = [p for p in passes if p['type'] == 'SHORT']
long_passes = [p for p in passes if p['type'] == 'LONG']

print(f"\n📊 Total Passes: {len(passes)}")
print(f"   🔹 Short Passes: {len(short_passes)}")
print(f"   🔹 Long Passes: {len(long_passes)}")

if passes:
    print(f"\n💯 Pass Breakdown:")
    for p in passes[:10]:  # Show first 10
        print(f"   Frame {p['frame']}: P{p['passer_id']} → P{p['receiver_id']} ({p['type']}, {p['distance']:.0f}px)")
    if len(passes) > 10:
        print(f"   ... and {len(passes) - 10} more passes")

print("\n" + "="*60)
print("✅ DONE!")
print("="*60)

