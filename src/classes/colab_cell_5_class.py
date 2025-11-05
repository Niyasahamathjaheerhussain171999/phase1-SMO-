# CELL 5: Fixed Football Analysis Class - Professional Ball & Player Tracking

# Workaround for OpenCV typing module issue (cv2.dnn.DictValue AttributeError)
# This is a known bug in OpenCV typing stubs - cv2.dnn.DictValue doesn't exist in some versions
# We need to ensure DictValue exists before cv2.typing tries to access it during bootstrap
import sys
import types
import importlib

# Pre-create cv2.dnn module with DictValue in sys.modules BEFORE cv2 imports
# This ensures it's available when cv2.typing tries to access it
if 'cv2.dnn' not in sys.modules:
    cv2_dnn_mock = types.ModuleType('cv2.dnn')
    
    class DictValue:
        """Mock DictValue class for OpenCV typing stubs compatibility"""
        pass
    
    cv2_dnn_mock.DictValue = DictValue
    sys.modules['cv2.dnn'] = cv2_dnn_mock

# Monkey-patch importlib.import_module to ensure DictValue exists when cv2.typing loads
_original_import_module = importlib.import_module

def _patched_import_module(name, package=None):
    """Patched import_module to handle cv2.typing DictValue issue"""
    # Before importing cv2.typing, ensure cv2.dnn.DictValue exists
    if name == 'cv2.typing' or (package and 'cv2' in str(package)):
        # Ensure cv2.dnn exists with DictValue
        if 'cv2' in sys.modules:
            cv2_module = sys.modules['cv2']
            if hasattr(cv2_module, 'dnn'):
                if not hasattr(cv2_module.dnn, 'DictValue'):
                    class DictValue:
                        pass
                    cv2_module.dnn.DictValue = DictValue
            else:
                # If cv2 doesn't have dnn yet, create it
                if 'cv2.dnn' in sys.modules:
                    cv2_module.dnn = sys.modules['cv2.dnn']
    
    result = _original_import_module(name, package)
    
    # After import, ensure DictValue exists
    if name == 'cv2' and hasattr(result, 'dnn'):
        if not hasattr(result.dnn, 'DictValue'):
            class DictValue:
                pass
            result.dnn.DictValue = DictValue
    
    return result

# Replace importlib.import_module with our patched version
importlib.import_module = _patched_import_module

# Also patch __import__ to catch cv2 imports
_original_import = __import__

def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Patched __import__ to handle cv2 bootstrap"""
    module = _original_import(name, globals, locals, fromlist, level)
    
    # If cv2 is being loaded, ensure dnn.DictValue exists
    if name == 'cv2':
        if hasattr(module, 'dnn'):
            if not hasattr(module.dnn, 'DictValue'):
                class DictValue:
                    pass
                module.dnn.DictValue = DictValue
    
    return module

# Replace __import__ with our patched version (use builtins module for compatibility)
import builtins
builtins.__import__ = _patched_import

# Now safe to import cv2
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import json
from collections import deque
from .kalman_tracker import KalmanTracker
from .pass_detector import PassDetector

class FixedFootballAnalysis:
    def __init__(self, show_video=True):
        # Try to load model, if fails use default
        try:
            self.model = YOLO('best.pt')
            print("✅ Loaded custom model: best.pt")
        except:
            # Use default YOLOv8 model if custom model fails
            self.model = YOLO('yolov8n.pt')
            print("⚠️ Using default YOLOv8 model (yolov8n.pt)")
        
        self.results = {
            'passes': [],
            'players': [],
            'accuracy_metrics': {}
        }
        
        self.show_video = show_video
        self.recent_passes = []  # Store recent passes for visualization
        
        # Initialize robust player tracker (Kalman-based) with RELAXED ghost filtering for debugging
        self.player_tracker = KalmanTracker(
            max_age=30,              # Keep tracks alive for 30 frames without detection
            min_hits=2,              # RELAXED: Require only 2 detections (was 3)
            iou_threshold=0.3,       # IoU threshold for matching
            min_bbox_area=500,       # RELAXED: Lower minimum (was 800)
            max_bbox_area=80000,     # RELAXED: Higher maximum (was 60000)
            min_track_length=15,     # RELAXED: Longer before validation (was 10)
            max_velocity_jump=100    # RELAXED: Higher tolerance (was 50)
        )
        
        # Initialize robust pass detector
        self.pass_detector = PassDetector(
            min_pass_distance=40,
            max_pass_distance=350,
            short_pass_threshold=120,
            min_ball_speed=1.5,      # RELAXED: Lower from 2.0
            min_ball_travel=25,      # RELAXED: Lower from 30
            min_alignment=0.35,      # RELAXED: Lower from 0.4
            min_confidence=0.45,    # RELAXED: Lower from 0.5
            pass_cooldown=30,
            ball_history_size=6,    # RELAXED: Lower from 10 (works with shorter history)
            max_ball_to_player_dist=80  # RELAXED: Increased from 70
        )
        
        # Legacy structures (kept for compatibility)
        self.player_history = {}  # Extended history for each player
        self.player_smooth_pos = {}  # Smoothed positions using moving average
        self.player_smooth_vel = {}  # Smoothed velocities
        self.velocity_history = {}  # Track velocity patterns
        
        # Ball tracking structures - CRITICAL for pass detection
        self.ball_history = deque(maxlen=50)  # Ball position history
        self.ball_tracking = None  # Current ball state (position, velocity, confidence)
        self.ball_smooth_pos = None  # Smoothed ball position (Kalman-like)
        self.ball_smooth_vel = None  # Smoothed ball velocity
        self.ball_predicted_pos = None  # Predicted position when not detected
        
        # Pass detection structures
        self.ball_possession = {}  # Track which player has ball: {player_id: {'frame': frame_num, 'confidence': conf}}
        self.ball_possession_history = deque(maxlen=30)  # History of possession changes
        self.pass_candidates = []  # Potential passes being validated
        
        # Ball detection parameters
        self.ball_detection_conf_threshold = 0.2  # Higher threshold to reduce false positives
        self.ball_size_min = 8  # Minimum ball size (pixels)
        self.ball_size_max = 120  # Maximum ball size (pixels)
        self.ball_tracking_timeout = 10  # Frames to keep tracking when not detected
        self.ball_max_jump_distance = 100  # Maximum distance ball can jump between frames (outlier rejection)
        
        # Pass detection parameters - BALANCED for accuracy and detection
        self.last_pass_frame = {}  # Track last pass detection per player pair
        self.pass_cooldown = 30  # Minimum frames between pass detections (1 second at 30fps)
        self.history_length = 30  # Keep 30 frames of history for analysis
        self.min_pass_distance = 30  # Minimum distance for a pass (pixels) - relaxed
        self.max_pass_distance = 400  # Maximum distance for a pass (pixels) - relaxed
        self.short_pass_threshold = 120  # Distance threshold between short and long passes
        self.ball_possession_threshold = 80  # Distance threshold for ball possession (pixels) - relaxed
        self.min_pass_duration = 5  # Minimum frames for a pass to be valid
        self.max_pass_duration = 30  # Maximum frames for a pass (ball moving between players)
        
        # Kalman filter parameters for ball smoothing
        self.ball_kalman_state = None  # State: [x, y, vx, vy]
        self.ball_kalman_cov = None  # Covariance matrix
        self.kalman_process_noise = 0.5  # Increased for more stability
        self.kalman_measurement_noise = 10.0  # Increased to trust predictions more
        
        # Team detection and player stats
        self.team_colors = {}  # player_id -> team_id mapping
        self.team_stats = {'Team A': {}, 'Team B': {}}  # team stats
        self.player_stats = {}  # player_id -> individual stats
    
    def analyze_video(self, video_path, device='auto'):
        """Complete analysis with GPU acceleration"""
        print("🎯 Starting Professional Football Analysis...")
        print("⚽ Ball tracking: ENABLED (critical for pass detection)")
        print("👥 Player tracking: ENABLED (smooth Kalman filtering)")
        
        # Auto-detect device if needed
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🔧 Using device: {device}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Could not open video file")
            return [], self.get_empty_accuracy()
        
        # Get video properties with fallback defaults
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0  # Default to 30 FPS if unable to detect
            print("⚠️ Warning: Could not detect FPS, defaulting to 30 FPS")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video: {total_frames} frames at {fps:.1f} FPS ({frame_width}x{frame_height})")
        if fps > 0:
            print(f"⏱️ Video length: {total_frames/fps/60:.1f} minutes")
        else:
            print(f"⏱️ Video length: Unknown")
        
        frame_count = 0
        players_tracking = {}
        pass_events = []
        detection_count = 0
        
        # Initialize tracking (reset trackers) with RELAXED ghost filtering for debugging
        self.player_tracker = KalmanTracker(
            max_age=30, min_hits=2, iou_threshold=0.3,
            min_bbox_area=500, max_bbox_area=80000,
            min_track_length=15, max_velocity_jump=100
        )
        self.pass_detector = PassDetector(
            min_pass_distance=40, max_pass_distance=350, short_pass_threshold=120,
            min_ball_speed=1.5, min_ball_travel=25, min_alignment=0.35,
            min_confidence=0.45, pass_cooldown=30, ball_history_size=6,
            max_ball_to_player_dist=80
        )
        
        # Legacy structures (kept for compatibility)
        self.player_history = {}
        self.player_smooth_pos = {}
        self.player_smooth_vel = {}
        self.velocity_history = {}
        self.last_pass_frame = {}
        self.ball_history.clear()
        self.ball_tracking = None
        self.ball_smooth_pos = None
        self.ball_smooth_vel = None
        self.ball_predicted_pos = None
        self.ball_possession = {}
        self.ball_possession_history.clear()
        self.pass_candidates = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO detection with specified device
            try:
                results = self.model(frame, device=device, verbose=False)
                
                # Process detections (players AND ball)
                player_detections, ball_detections = self.process_detections(results[0], frame_count, frame)
                detection_count += len(player_detections)
                
                # Track ball with smooth Kalman filtering
                self.track_ball_smooth(ball_detections, frame_count, frame_width, frame_height)
                
                # NEW: Use robust Kalman tracker for players (with interpolation)
                if player_detections:
                    # Convert detections to format expected by tracker
                    tracker_detections = []
                    for det in player_detections:
                        # Detect team and jersey color
                        team_result = self.assign_team(det, {}, frame)
                        if isinstance(team_result, tuple):
                            team, jersey_color = team_result
                        else:
                            team = team_result
                            jersey_color = None
                        
                        tracker_detections.append({
                            'x': det['center_x'],
                            'y': det['center_y'],
                            'width': det['width'],
                            'height': det['height'],
                            'confidence': det['confidence'],
                            'team': team,
                            'jersey_color': jersey_color
                        })
                    
                    # Update tracker (includes interpolation for missing detections)
                    players_tracking = self.player_tracker.update(tracker_detections, frame_count)
                    
                    # Update team assignments in tracking
                    for player_id, player in players_tracking.items():
                        if player['team'] is None:
                            # Try to assign team from history
                            if player_id in self.player_tracker.tracks:
                                track = self.player_tracker.tracks[player_id]
                                if track.get('team'):
                                    player['team'] = track['team']
                else:
                    # No detections - tracker will interpolate
                    players_tracking = self.player_tracker.update([], frame_count)
                
                # DEBUG: Print frame info (every 50 frames or first 20 frames)
                if frame_count % 50 == 0 or frame_count < 20:
                    print(f"\n[DEBUG Frame {frame_count}]")
                    print(f"  Players detected: {len(player_detections)}")
                    print(f"  Players tracked: {len(players_tracking)}")
                    if len(players_tracking) > 0:
                        player_ids = sorted(players_tracking.keys())
                        print(f"  Tracked IDs: {player_ids[:10]}{'...' if len(player_ids) > 10 else ''}")
                    print(f"  Ball tracking: {'YES' if self.ball_tracking else 'NO'}")
                    if self.ball_tracking:
                        print(f"  Ball pos: ({self.ball_tracking['x']:.0f}, {self.ball_tracking['y']:.0f}), "
                              f"speed: {self.ball_tracking.get('speed', 0):.1f}, "
                              f"conf: {self.ball_tracking.get('confidence', 0):.2f}")
                    print(f"  Ball history length: {len(self.ball_history)} (need 6 for ball-based detection)")
                
                # Update ball possession tracking (for stats)
                self.update_ball_possession(players_tracking, frame_count)
                
                # DEBUG: Print possession info
                if frame_count % 50 == 0 or frame_count < 20:
                    print(f"  Ball possession: {len(self.ball_possession)} player(s)")
                    if len(self.ball_possession) > 0:
                        for pid, poss in self.ball_possession.items():
                            print(f"    P{pid} has ball (conf: {poss.get('confidence', 0):.2f}, "
                                  f"held for {frame_count - poss.get('start_frame', frame_count)} frames)")
                    else:
                        print(f"    ⚠️  No player has ball (ball threshold: {self.ball_possession_threshold}px)")
                    print(f"  Possession history: {len(self.ball_possession_history)} events")
                
                # NEW: Use robust pass detector (tightly integrated with ball tracking)
                # Also use possession-change-based detection as fallback
                passes = []
                
                # Method 1: Ball-based pass detection (requires 6+ frames of history)
                if self.ball_tracking and len(self.ball_history) >= 6:
                    # Convert ball_history to list format expected by pass detector
                    ball_history_list = list(self.ball_history)
                    
                    # Detect passes using ball tracking
                    passes_ball = self.pass_detector.detect_passes(
                        players_tracking,
                        self.ball_tracking,
                        ball_history_list,
                        frame_count
                    )
                    passes.extend(passes_ball)
                
                # Method 2: Possession-change-based detection (STRICT: requires valid sequence)
                if len(self.ball_possession_history) >= 3:
                    # Check for possession changes (ball moved from one player to another)
                    recent_possession = list(self.ball_possession_history)[-15:]  # Last 15 possession events
                    
                    # STRICT: Look for valid pass sequence: gained -> lost -> gained (by different player)
                    for i in range(len(recent_possession) - 2):
                        event1 = recent_possession[i]  # First player gains
                        event2 = recent_possession[i + 1]  # First player loses
                        event3 = recent_possession[i + 2]  # Second player gains
                        
                        # Validate sequence: same player gains then loses, then different player gains
                        if (event1.get('action') == 'gained' and 
                            event2.get('action') == 'lost' and
                            event3.get('action') == 'gained' and
                            event1.get('player_id') == event2.get('player_id') and  # Same player loses it
                            event3.get('player_id') != event1.get('player_id') and  # Different player gains
                            event3.get('frame') - event1.get('frame', 0) < 25 and  # Within 25 frames
                            event3.get('frame') - event1.get('frame', 0) > 3):  # At least 3 frames apart
                            
                            passer_id = event1.get('player_id')
                            receiver_id = event3.get('player_id')
                            
                            # Validate: both players must still exist and be on same team
                            if (passer_id in players_tracking and 
                                receiver_id in players_tracking):
                                
                                passer = players_tracking[passer_id]
                                receiver = players_tracking[receiver_id]
                                
                                # Check same team
                                if (passer.get('team') and receiver.get('team') and
                                    passer.get('team') == receiver.get('team')):
                                    
                                    # Calculate distance
                                    pass_distance = np.sqrt(
                                        (receiver['x'] - passer['x'])**2 + 
                                        (receiver['y'] - passer['y'])**2
                                    )
                                    
                                    # Validate distance (STRICT)
                                    if 50 <= pass_distance <= 300:
                                        # Check cooldown
                                        pair_key = tuple(sorted([passer_id, receiver_id]))
                                        if pair_key not in self.last_pass_frame or \
                                           frame_count - self.last_pass_frame[pair_key] >= 40:
                                            
                                            pass_type = 'long' if pass_distance > 120 else 'short'
                                            
                                            # Determine success (receiver still has ball)
                                            success = receiver_id in self.ball_possession
                                            
                                            # Create pass event
                                            pass_event = {
                                                'frame': event3.get('frame', frame_count),
                                                'passer_id': passer_id,
                                                'receiver_id': receiver_id,
                                                'team': passer.get('team'),
                                                'distance': float(pass_distance),
                                                'type': pass_type,
                                                'success': success,
                                                'confidence': 0.75,  # Higher confidence for strict validation
                                                'ball_speed': self.ball_tracking.get('speed', 0) if self.ball_tracking else 0,
                                                'alignment': 0.6,  # Estimated
                                                'ball_travel': pass_distance * 0.9,  # Estimated
                                                'method': 'possession_change'
                                            }
                                            
                                            passes.append(pass_event)
                                            self.last_pass_frame[pair_key] = frame_count
                                            print(f"  ✅ PASS (possession): P{passer_id} → P{receiver_id} ({pass_type}, {pass_distance:.0f}px, {'✓' if success else '✗'})")
                                            break  # Found pass, move on
                
                pass_events.extend(passes)
                
                # DEBUG: Print pass detection info
                if passes:
                    print(f"  ✅ Detected {len(passes)} pass(es) at frame {frame_count}")
                elif frame_count % 50 == 0 or frame_count < 10:
                    # Only print debug if no passes detected and it's a debug frame
                    print(f"  ⚠️  No passes detected (players: {len(players_tracking)}, "
                          f"ball_history: {len(self.ball_history)}, "
                          f"possession_events: {len(self.ball_possession_history)})")
                
                # Store recent passes for visualization and update player stats
                if passes:
                    for p in passes:
                        p['frame_detected'] = frame_count
                        
                        # Update player statistics
                        passer_id = p['passer_id']
                        receiver_id = p['receiver_id']
                        team = p.get('team', 'Unknown')
                        
                        # Initialize stats if needed
                        if passer_id not in self.player_stats:
                            self.player_stats[passer_id] = {
                                'passes': 0, 'successful_passes': 0,
                                'received': 0, 'short_passes': 0, 'long_passes': 0,
                                'team': team
                            }
                        if receiver_id not in self.player_stats:
                            self.player_stats[receiver_id] = {
                                'passes': 0, 'successful_passes': 0,
                                'received': 0, 'short_passes': 0, 'long_passes': 0,
                                'team': team
                            }
                        
                        # Update passer stats
                        self.player_stats[passer_id]['passes'] += 1
                        self.player_stats[passer_id]['team'] = team
                        if p.get('success', False):
                            self.player_stats[passer_id]['successful_passes'] += 1
                        pass_type = p.get('type', '').lower()
                        if pass_type == 'short':
                            self.player_stats[passer_id]['short_passes'] += 1
                        elif pass_type == 'long':
                            self.player_stats[passer_id]['long_passes'] += 1
                        
                        # Update receiver stats
                        self.player_stats[receiver_id]['received'] += 1
                        self.player_stats[receiver_id]['team'] = team
                        
                        # Print pass detection
                        print(f"✅ {p['type'].upper()} PASS ({team}): P{passer_id} → P{receiver_id} "
                              f"({p['distance']:.0f}px, conf={p['confidence']:.2f}, "
                              f"align={p['alignment']:.2f}, {'✓' if p['success'] else '✗'})")
                    
                    self.recent_passes.extend(passes)
                    # Keep only recent passes (last 60 frames worth)
                    self.recent_passes = [p for p in self.recent_passes if frame_count - p['frame_detected'] < 60]
                
                # Visualize if enabled
                if self.show_video:
                    try:
                        frame_vis = self.visualize_frame(frame, player_detections, players_tracking, passes, frame_count, fps)
                        if frame_vis is not None and frame_vis.size > 0:
                            cv2.imshow('Football Analysis - Press Q to quit, Space to pause', frame_vis)
                            
                            # Handle keyboard input
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q') or key == ord('Q'):
                                print("\n⏸️  Visualization stopped by user")
                                self.show_video = False
                            elif key == ord(' '):
                                # Pause until space is pressed again
                                print("⏸️  Paused - Press Space to continue")
                                while True:
                                    key2 = cv2.waitKey(0) & 0xFF
                                    if key2 == ord(' '):
                                        print("▶️  Resumed")
                                        break
                                    elif key2 == ord('q') or key2 == ord('Q'):
                                        print("\n⏸️  Visualization stopped by user")
                                        self.show_video = False
                                        break
                    except Exception as viz_error:
                        # Don't let visualization errors stop the analysis
                        if frame_count % 100 == 0:
                            print(f"⚠️ Visualization error (frame {frame_count}): {viz_error}")
                
            except Exception as e:
                import traceback
                # Print full error for first few occurrences only
                if frame_count < 10 or frame_count % 100 == 0:
                    print(f"⚠️ Detection error at frame {frame_count}: {e}")
                    if frame_count < 5:  # Only print traceback for first few errors
                        traceback.print_exc()
                # For other frames, just count silently to avoid spam
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                ball_status = "Detected" if self.ball_tracking else "Not detected"
                possession_count = len(self.ball_possession)
                print(f"⏳ Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
                print(f"   Players: {len(players_tracking)}, Ball: {ball_status}, Possession: {possession_count}")
                print(f"   🎯 Total Passes Detected: {len(pass_events)}")
        
        # Clean up
        if self.show_video:
            cv2.destroyAllWindows()
        
        cap.release()
        
        print(f"📊 Total detections: {detection_count}")
        print(f"📊 Total passes detected: {len(pass_events)}")
        
        # Calculate accuracy
        accuracy = self.calculate_accuracy(pass_events)
        
        print(f"\n{'='*60}")
        print(f"✅ Analysis Complete!")
        print(f"{'='*60}")
        print(f"📊 Total Passes Detected: {len(pass_events)}")
        print(f"   - Short passes: {accuracy['short_count']}")
        print(f"   - Long passes: {accuracy['long_count']}")
        print(f"   - Successful: {accuracy['successful_count']}")
        print(f"   - Failed: {len(pass_events) - accuracy['successful_count']}")
        print(f"🎯 Overall Pass Accuracy: {accuracy['overall']:.1%}")
        print(f"{'='*60}\n")
        
        # Display team and player-specific statistics
        self.print_player_statistics()
        
        if len(pass_events) == 0:
            print("⚠️  WARNING: No passes detected!")
            print("   Possible reasons:")
            print("   1. Ball not being tracked properly (check video visualization)")
            print("   2. Players not close enough to ball")
            print("   3. Ball speed too low")
            print("   4. Ball trajectory not aligned with players")
            print("   5. Players on different teams (passes only count within same team)")
            print("   Run with visualization enabled to debug further.")
        
        return pass_events, accuracy
    
    def print_player_statistics(self):
        """Print detailed player and team statistics"""
        if not self.player_stats:
            print("⚠️  No player statistics available (no passes detected)")
            return
        
        print(f"\n{'='*60}")
        print(f"👥 PLAYER STATISTICS (by Team)")
        print(f"{'='*60}\n")
        
        # Group by team
        teams = {}
        for player_id, stats in self.player_stats.items():
            team = stats.get('team', 'Unknown')
            if team not in teams:
                teams[team] = []
            teams[team].append((player_id, stats))
        
        # Print each team
        for team_name in sorted(teams.keys()):
            players = teams[team_name]
            print(f"🔵 {team_name}")
            print(f"{'-'*60}")
            
            # Calculate team totals
            team_passes = sum(s['passes'] for _, s in players)
            team_successful = sum(s['successful_passes'] for _, s in players)
            team_short = sum(s['short_passes'] for _, s in players)
            team_long = sum(s['long_passes'] for _, s in players)
            
            print(f"Team Total: {team_passes} passes ({team_successful} successful, {team_short} short, {team_long} long)")
            print()
            
            # Sort players by number of passes (descending)
            players_sorted = sorted(players, key=lambda x: x[1]['passes'], reverse=True)
            
            # Print each player
            for player_id, stats in players_sorted:
                accuracy = (stats['successful_passes'] / stats['passes'] * 100) if stats['passes'] > 0 else 0
                print(f"  Player {player_id}:")
                print(f"    Passes Made: {stats['passes']} ({stats['successful_passes']} successful, {accuracy:.0f}% accuracy)")
                print(f"    Passes Received: {stats['received']}")
                print(f"    Short/Long: {stats['short_passes']}/{stats['long_passes']}")
                print()
        
        print(f"{'='*60}\n")
    
    def process_detections(self, results, frame_num, frame=None):
        """Process YOLO detections - separate players and ball"""
        player_detections = []
        ball_detections = []
        
        if results.boxes is None or len(results.boxes) == 0:
            return player_detections, ball_detections
        
        for box in results.boxes:
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Class 0 = person, Class 32 = sports ball (COCO dataset)
            if cls == 0 and conf > 0.25:  # Lower threshold to detect ALL players
                if 800 < area < 60000:  # Wider size range to catch all players
                    player_detections.append({
                        'frame': frame_num,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'confidence': conf,
                        'class': cls,
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2,
                        'width': width,
                        'height': height
                    })
            elif cls == 32 and conf > self.ball_detection_conf_threshold:  # Sports ball
                # Ball size range
                if self.ball_size_min**2 < area < self.ball_size_max**2:
                    ball_detections.append({
                        'frame': frame_num,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'confidence': conf,
                        'class': cls,
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2,
                        'width': width,
                        'height': height,
                        'area': area
                    })
        
        # If no ball detected by YOLO, try color-based detection
        if len(ball_detections) == 0 and frame is not None:
            color_ball_detections = self.detect_ball_by_color(frame, frame_num)
            if color_ball_detections:
                # Use the best one
                best_ball = max(color_ball_detections, key=lambda x: x['confidence'])
                ball_detections.append(best_ball)
        
        return player_detections, ball_detections
    
    def detect_ball_by_color(self, frame, frame_num):
        """Detect ball using color-based detection (white/colored ball)"""
        ball_detections = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multiple color ranges for different ball types
        # White range
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Yellow range
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        # Orange range
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([20, 255, 255])
        
        # Create masks
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask = cv2.bitwise_or(cv2.bitwise_or(mask_white, mask_yellow), mask_orange)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 3000:  # Ball size range
                x, y, w, h = cv2.boundingRect(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if circularity > 0.4 and 0.6 < aspect_ratio < 1.4:
                        ball_detections.append({
                            'frame': frame_num,
                            'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h,
                            'confidence': min(0.6, circularity * 0.8),
                            'class': 32,
                            'center_x': x + w / 2,
                            'center_y': y + h / 2,
                            'width': w,
                            'height': h,
                            'area': area
                        })
        
        return ball_detections
    
    def track_ball_smooth(self, ball_detections, frame_num, frame_width, frame_height):
        """Track ball with Kalman filter-like smoothing with outlier rejection"""
        # Initialize Kalman state if needed
        if self.ball_kalman_state is None:
            self.ball_kalman_state = np.array([frame_width/2, frame_height/2, 0.0, 0.0])  # [x, y, vx, vy]
            self.ball_kalman_cov = np.eye(4) * 50.0  # Lower initial uncertainty
        
        # Prediction step (always predict)
        dt = 1.0  # Assume 1 frame = 1 time unit
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 0.95, 0],  # Velocity decay (ball slows down)
            [0, 0, 0, 0.95]
        ])
        
        # Predict state
        predicted_state = F @ self.ball_kalman_state
        predicted_cov = F @ self.ball_kalman_cov @ F.T + np.eye(4) * self.kalman_process_noise
        
        # Get predicted position for validation
        predicted_x = predicted_state[0]
        predicted_y = predicted_state[1]
        
        # Measurement step (if ball detected)
        if len(ball_detections) > 0:
            # Filter detections: only accept those close to predicted position (outlier rejection)
            valid_detections = []
            for ball in ball_detections:
                if self.ball_tracking is not None:
                    # Check distance from predicted position
                    dist_from_predicted = np.sqrt(
                        (ball['center_x'] - predicted_x)**2 + 
                        (ball['center_y'] - predicted_y)**2
                    )
                    # Only accept if close to prediction OR if we have no tracking (first detection)
                    if dist_from_predicted < self.ball_max_jump_distance:
                        valid_detections.append(ball)
                else:
                    # First detection - accept it
                    valid_detections.append(ball)
            
            if len(valid_detections) > 0:
                # Use highest confidence valid detection
                best_ball = max(valid_detections, key=lambda x: x['confidence'])
                measurement = np.array([best_ball['center_x'], best_ball['center_y']])
                
                # Check distance from prediction one more time
                innovation_magnitude = np.sqrt(
                    (measurement[0] - predicted_x)**2 + 
                    (measurement[1] - predicted_y)**2
                )
                
                # Only update if measurement is reasonable
                if innovation_magnitude < self.ball_max_jump_distance or self.ball_tracking is None:
                    # Measurement matrix (we only observe position)
                    H = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]
                    ])
                    
                    # Innovation (difference between prediction and measurement)
                    innovation = measurement - H @ predicted_state
                    innovation_cov = H @ predicted_cov @ H.T + np.eye(2) * self.kalman_measurement_noise
                    
                    # Kalman gain
                    try:
                        K = predicted_cov @ H.T @ np.linalg.inv(innovation_cov)
                    except:
                        K = predicted_cov @ H.T @ np.linalg.pinv(innovation_cov)
                    
                    # Update state with weighted average (less aggressive)
                    alpha = 0.6  # How much to trust measurement (lower = smoother)
                    self.ball_kalman_state = predicted_state + alpha * K @ innovation
                    self.ball_kalman_cov = (np.eye(4) - alpha * K @ H) @ predicted_cov
                    
                    # Extract smoothed position and velocity
                    smooth_x = self.ball_kalman_state[0]
                    smooth_y = self.ball_kalman_state[1]
                    smooth_vx = self.ball_kalman_state[2]
                    smooth_vy = self.ball_kalman_state[3]
                    smooth_speed = np.sqrt(smooth_vx**2 + smooth_vy**2)
                    
                    # Store ball tracking
                    self.ball_tracking = {
                        'x': smooth_x,
                        'y': smooth_y,
                        'raw_x': best_ball['center_x'],
                        'raw_y': best_ball['center_y'],
                        'velocity_x': smooth_vx,
                        'velocity_y': smooth_vy,
                        'speed': smooth_speed,
                        'confidence': best_ball['confidence'],
                        'frame': frame_num
                    }
                    
                    self.ball_smooth_pos = (smooth_x, smooth_y)
                    self.ball_smooth_vel = (smooth_vx, smooth_vy)
                    self.ball_predicted_pos = None
                    
                    # Add to history
                    self.ball_history.append({
                        'frame': frame_num,
                        'x': smooth_x,
                        'y': smooth_y,
                        'velocity_x': smooth_vx,
                        'velocity_y': smooth_vy,
                        'speed': smooth_speed,
                        'confidence': best_ball['confidence']
                    })
                else:
                    # Measurement too far from prediction - reject it, use prediction
                    if self.ball_tracking is not None:
                        frames_missing = frame_num - self.ball_tracking['frame']
                        if frames_missing < self.ball_tracking_timeout:
                            smooth_x = predicted_state[0]
                            smooth_y = predicted_state[1]
                            smooth_vx = predicted_state[2]
                            smooth_vy = predicted_state[3]
                            
                            self.ball_tracking = {
                                'x': smooth_x,
                                'y': smooth_y,
                                'raw_x': smooth_x,
                                'raw_y': smooth_y,
                                'velocity_x': smooth_vx,
                                'velocity_y': smooth_vy,
                                'speed': np.sqrt(smooth_vx**2 + smooth_vy**2),
                                'confidence': max(0.1, self.ball_tracking['confidence'] * 0.85),
                                'frame': frame_num,
                                'predicted': True
                            }
                            self.ball_kalman_state = predicted_state
                            self.ball_kalman_cov = predicted_cov
                            
                            # Add predicted position to history (CRITICAL: keeps history building)
                            self.ball_history.append({
                                'frame': frame_num,
                                'x': smooth_x,
                                'y': smooth_y,
                                'velocity_x': smooth_vx,
                                'velocity_y': smooth_vy,
                                'speed': np.sqrt(smooth_vx**2 + smooth_vy**2),
                                'confidence': max(0.1, self.ball_tracking['confidence'] * 0.85),
                                'predicted': True
                            })
            else:
                # No valid detections - use prediction
                if self.ball_tracking is not None:
                    frames_missing = frame_num - self.ball_tracking['frame']
                    if frames_missing < self.ball_tracking_timeout:
                        smooth_x = predicted_state[0]
                        smooth_y = predicted_state[1]
                        smooth_vx = predicted_state[2]
                        smooth_vy = predicted_state[3]
                        
                        self.ball_tracking = {
                            'x': smooth_x,
                            'y': smooth_y,
                            'raw_x': smooth_x,
                            'raw_y': smooth_y,
                            'velocity_x': smooth_vx,
                            'velocity_y': smooth_vy,
                            'speed': np.sqrt(smooth_vx**2 + smooth_vy**2),
                            'confidence': max(0.1, self.ball_tracking['confidence'] * 0.85),
                            'frame': frame_num,
                            'predicted': True
                        }
                        self.ball_kalman_state = predicted_state
                        self.ball_kalman_cov = predicted_cov
                        
                        # Add predicted position to history (CRITICAL: keeps history building)
                        self.ball_history.append({
                            'frame': frame_num,
                            'x': smooth_x,
                            'y': smooth_y,
                            'velocity_x': smooth_vx,
                            'velocity_y': smooth_vy,
                            'speed': np.sqrt(smooth_vx**2 + smooth_vy**2),
                            'confidence': max(0.1, self.ball_tracking['confidence'] * 0.85),
                            'predicted': True
                        })
                    else:
                        # Lost ball
                        self.ball_tracking = None
                        self.ball_smooth_pos = None
                        self.ball_smooth_vel = None
        else:
            # No detection - use prediction
            if self.ball_tracking is not None:
                frames_missing = frame_num - self.ball_tracking['frame']
                
                if frames_missing < self.ball_tracking_timeout:
                    smooth_x = predicted_state[0]
                    smooth_y = predicted_state[1]
                    smooth_vx = predicted_state[2]
                    smooth_vy = predicted_state[3]
                    smooth_speed = np.sqrt(smooth_vx**2 + smooth_vy**2)
                    
                    # Keep tracking with predicted position
                    self.ball_tracking = {
                        'x': smooth_x,
                        'y': smooth_y,
                        'raw_x': smooth_x,
                        'raw_y': smooth_y,
                        'velocity_x': smooth_vx,
                        'velocity_y': smooth_vy,
                        'speed': smooth_speed,
                        'confidence': max(0.1, self.ball_tracking['confidence'] * 0.85),  # Decay confidence
                        'frame': frame_num,
                        'predicted': True
                    }
                    
                    self.ball_smooth_pos = (smooth_x, smooth_y)
                    self.ball_smooth_vel = (smooth_vx, smooth_vy)
                    self.ball_predicted_pos = (smooth_x, smooth_y)
                    
                    # Update Kalman state with prediction
                    self.ball_kalman_state = predicted_state
                    self.ball_kalman_cov = predicted_cov
                    
                    # CRITICAL: Add predicted position to history (keeps history building even without detection)
                    self.ball_history.append({
                        'frame': frame_num,
                        'x': smooth_x,
                        'y': smooth_y,
                        'velocity_x': smooth_vx,
                        'velocity_y': smooth_vy,
                        'speed': smooth_speed,
                        'confidence': max(0.1, self.ball_tracking['confidence'] * 0.85),
                        'predicted': True
                    })
                else:
                    # Lost ball for too long
                    self.ball_tracking = None
                    self.ball_smooth_pos = None
                    self.ball_smooth_vel = None
                    self.ball_predicted_pos = None
    
    def track_players_smooth(self, detections, players_tracking, frame_num, frame=None):
        """Track players with smooth position and velocity estimation"""
        current_players = {}
        
        # Smoothing factor for exponential moving average
        alpha_pos = 0.7
        alpha_vel = 0.6
        
        for det in detections:
            # Validate detection has required fields
            if not all(key in det for key in ['center_x', 'center_y', 'confidence', 'width', 'height']):
                continue
            if det['center_x'] is None or det['center_y'] is None:
                continue
            
            player_id = self.assign_player_id(det, players_tracking)
            det['player_id'] = player_id  # Store for team assignment
            
            # Calculate instantaneous velocity
            velocity_x = 0
            velocity_y = 0
            speed = 0
            
            if player_id in self.player_history and len(self.player_history[player_id]) > 0:
                try:
                    last_pos = self.player_history[player_id][-1]
                    last_frame = last_pos.get('frame')
                    
                    # Ensure both frame_num and last_frame are valid numbers
                    if last_frame is not None and frame_num is not None:
                        frames_diff = frame_num - last_frame
                        if frames_diff > 0 and frames_diff <= 5:
                            velocity_x = (det['center_x'] - last_pos['x']) / frames_diff
                            velocity_y = (det['center_y'] - last_pos['y']) / frames_diff
                            speed = np.sqrt(velocity_x**2 + velocity_y**2)
                except (TypeError, KeyError, ZeroDivisionError):
                    # Silently handle any calculation errors
                    pass
            
            # Apply smoothing (with None checks)
            if player_id in self.player_smooth_pos:
                try:
                    prev_x = self.player_smooth_pos[player_id].get('x', det['center_x'])
                    prev_y = self.player_smooth_pos[player_id].get('y', det['center_y'])
                    if prev_x is not None and prev_y is not None:
                        smooth_x = alpha_pos * det['center_x'] + (1 - alpha_pos) * prev_x
                        smooth_y = alpha_pos * det['center_y'] + (1 - alpha_pos) * prev_y
                    else:
                        smooth_x = det['center_x']
                        smooth_y = det['center_y']
                except (TypeError, KeyError):
                    smooth_x = det['center_x']
                    smooth_y = det['center_y']
            else:
                smooth_x = det['center_x']
                smooth_y = det['center_y']
            
            if player_id in self.player_smooth_vel:
                try:
                    prev_vx = self.player_smooth_vel[player_id].get('x', velocity_x)
                    prev_vy = self.player_smooth_vel[player_id].get('y', velocity_y)
                    if prev_vx is not None and prev_vy is not None:
                        smooth_vx = alpha_vel * velocity_x + (1 - alpha_vel) * prev_vx
                        smooth_vy = alpha_vel * velocity_y + (1 - alpha_vel) * prev_vy
                        smooth_speed = np.sqrt(smooth_vx**2 + smooth_vy**2)
                    else:
                        smooth_vx = velocity_x
                        smooth_vy = velocity_y
                        smooth_speed = speed
                except (TypeError, KeyError):
                    smooth_vx = velocity_x
                    smooth_vy = velocity_y
                    smooth_speed = speed
            else:
                smooth_vx = velocity_x
                smooth_vy = velocity_y
                smooth_speed = speed
            
            # Store smoothed values
            self.player_smooth_pos[player_id] = {'x': smooth_x, 'y': smooth_y}
            self.player_smooth_vel[player_id] = {'x': smooth_vx, 'y': smooth_vy, 'speed': smooth_speed}
            
            # Detect team and jersey color
            team_result = self.assign_team(det, players_tracking, frame)
            if isinstance(team_result, tuple):
                team, jersey_color = team_result
            else:
                team = team_result
                jersey_color = None
            
            # Store current player data
            current_players[player_id] = {
                'frame': frame_num,
                'x': smooth_x,
                'y': smooth_y,
                'raw_x': det['center_x'],
                'raw_y': det['center_y'],
                'confidence': det['confidence'],
                'team': team,
                'jersey_color': jersey_color,
                'width': det['width'],
                'height': det['height'],
                'velocity_x': smooth_vx,
                'velocity_y': smooth_vy,
                'speed': smooth_speed
            }
            
            # Update player history
            if player_id not in self.player_history:
                self.player_history[player_id] = deque(maxlen=self.history_length)
            
            self.player_history[player_id].append({
                'frame': frame_num,
                'x': smooth_x,
                'y': smooth_y,
                'velocity_x': smooth_vx,
                'velocity_y': smooth_vy,
                'speed': smooth_speed
            })
        
        return current_players
    
    def update_ball_possession(self, players, frame_num):
        """Update which player has possession of the ball"""
        if self.ball_tracking is None:
            return
        
        ball_x = self.ball_tracking.get('x', 0)
        ball_y = self.ball_tracking.get('y', 0)
        ball_conf = self.ball_tracking.get('confidence', 0)
        
        if ball_x is None or ball_y is None or ball_conf < 0.1:
            return
        
        # Find closest player to ball
        closest_player_id = None
        closest_distance = float('inf')
        
        for player_id, player in players.items():
            # Handle both old format (x, y) and new format (from tracker)
            player_x = player.get('x', player.get('center_x', 0))
            player_y = player.get('y', player.get('center_y', 0))
            
            if player_x is None or player_y is None:
                continue
            
            dist_to_ball = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
            if dist_to_ball < closest_distance:
                closest_distance = dist_to_ball
                closest_player_id = player_id
        
        # Update possession if player is close enough (STRICT: ball must have reasonable confidence)
        if closest_player_id is not None and closest_distance < self.ball_possession_threshold and ball_conf > 0.15:
            # Update or set possession
            if closest_player_id not in self.ball_possession:
                # New possession
                self.ball_possession[closest_player_id] = {
                    'frame': frame_num,
                    'confidence': ball_conf,
                    'start_frame': frame_num
                }
                self.ball_possession_history.append({
                    'frame': frame_num,
                    'player_id': closest_player_id,
                    'action': 'gained'
                })
            else:
                # Update existing possession
                self.ball_possession[closest_player_id]['frame'] = frame_num
                self.ball_possession[closest_player_id]['confidence'] = max(
                    self.ball_possession[closest_player_id]['confidence'],
                    ball_conf
                )
        
        # Remove old possession (if player lost ball)
        players_to_remove = []
        for player_id, possession_info in self.ball_possession.items():
            if player_id != closest_player_id or closest_distance >= self.ball_possession_threshold:
                # Player lost possession
                if frame_num - possession_info['frame'] > 5:  # Lost possession for 5+ frames
                    players_to_remove.append(player_id)
        
        for player_id in players_to_remove:
            if player_id in self.ball_possession:
                possession_info = self.ball_possession[player_id]
                self.ball_possession_history.append({
                    'frame': frame_num,
                    'player_id': player_id,
                    'action': 'lost',
                    'duration': frame_num - possession_info['start_frame']
                })
                del self.ball_possession[player_id]
    
    def detect_passes_with_ball_tracking(self, players, frame_num):
        """ACCURATE pass detection using ball tracking with balanced validation"""
        passes = []
        
        if len(players) < 2 or self.ball_tracking is None or len(self.ball_history) < 8:
            return passes
        
        # Check every 3 frames (more frequent for better detection)
        if frame_num % 3 != 0:
            return passes
        
        # Get current ball state
        ball_x = self.ball_tracking['x']
        ball_y = self.ball_tracking['y']
        ball_speed = self.ball_tracking.get('speed', 0)
        ball_conf = self.ball_tracking.get('confidence', 0)
        
        # Ball must be moving (relaxed threshold)
        if ball_speed < 1.5 or ball_conf < 0.2:
            if frame_num % 100 == 0:  # Debug output
                print(f"[DEBUG] Frame {frame_num}: Ball not moving enough (speed={ball_speed:.1f}, conf={ball_conf:.2f})")
            return passes
        
        # Get ball position 8 frames ago (0.27 seconds at 30fps)
        if len(self.ball_history) >= 8:
            old_ball = self.ball_history[-8]
            old_ball_x = old_ball['x']
            old_ball_y = old_ball['y']
            
            # Calculate ball movement
            ball_travel = np.sqrt((ball_x - old_ball_x)**2 + (ball_y - old_ball_y)**2)
            
            # Ball must have moved (relaxed threshold)
            if ball_travel < 25:
                if frame_num % 100 == 0:  # Debug output
                    print(f"[DEBUG] Frame {frame_num}: Ball didn't travel enough ({ball_travel:.0f}px)")
                return passes
            
            # Find player closest to old ball position (passer)
            passer_id = None
            passer_dist = float('inf')
            passer_old_pos = None
            
            for pid, player in players.items():
                if pid not in self.player_history or len(self.player_history[pid]) < 8:
                    continue
                old_pos = self.player_history[pid][-8]
                dist = np.sqrt((old_pos['x'] - old_ball_x)**2 + (old_pos['y'] - old_ball_y)**2)
                if dist < passer_dist:
                    passer_dist = dist
                    passer_id = pid
                    passer_old_pos = old_pos
            
            # Find player closest to current ball position (receiver)
            receiver_id = None
            receiver_dist = float('inf')
            
            for pid, player in players.items():
                dist = np.sqrt((player['x'] - ball_x)**2 + (player['y'] - ball_y)**2)
                if dist < receiver_dist:
                    receiver_dist = dist
                    receiver_id = pid
            
            # VALIDATION: Players must be reasonably close to ball
            if (passer_id is not None and receiver_id is not None and 
                passer_id != receiver_id and 
                passer_dist < 80 and receiver_dist < 80):  # Relaxed proximity
                
                # Get current player positions
                passer = players[passer_id]
                receiver = players[receiver_id]
                
                # CRITICAL: Only count passes between SAME TEAM players!
                passer_team = passer.get('team', 'Unknown')
                receiver_team = receiver.get('team', 'Unknown')
                
                if passer_team != receiver_team:
                    if frame_num % 100 == 0:  # Debug output
                        print(f"[DEBUG] Frame {frame_num}: Pass rejected - different teams (P{passer_id}={passer_team}, P{receiver_id}={receiver_team})")
                    return passes
                
                # Check cooldown
                pair_key = tuple(sorted([passer_id, receiver_id]))
                if pair_key in self.last_pass_frame:
                    if frame_num - self.last_pass_frame[pair_key] < self.pass_cooldown:
                        return passes
                
                # Calculate pass distance (player-to-player distance, not ball travel)
                pass_distance = np.sqrt(
                    (receiver['x'] - passer['x'])**2 + 
                    (receiver['y'] - passer['y'])**2
                )
                
                # Distance validation (relaxed)
                if pass_distance < self.min_pass_distance or pass_distance > self.max_pass_distance:
                    if frame_num % 100 == 0:  # Debug output
                        print(f"[DEBUG] Frame {frame_num}: Pass distance out of range ({pass_distance:.0f}px)")
                    return passes
                
                # IMPORTANT: Validate ball trajectory - ball should move toward receiver
                ball_dx = ball_x - old_ball_x
                ball_dy = ball_y - old_ball_y
                ball_dir = np.array([ball_dx, ball_dy])
                ball_dir_norm = np.linalg.norm(ball_dir)
                
                if ball_dir_norm > 0:
                    ball_dir = ball_dir / ball_dir_norm
                    
                    # Direction from passer to receiver
                    receiver_dx = receiver['x'] - passer['x']
                    receiver_dy = receiver['y'] - passer['y']
                    receiver_dir = np.array([receiver_dx, receiver_dy])
                    receiver_dir_norm = np.linalg.norm(receiver_dir)
                    
                    if receiver_dir_norm > 0:
                        receiver_dir = receiver_dir / receiver_dir_norm
                        
                        # Calculate alignment (ball should move toward receiver)
                        alignment = np.dot(ball_dir, receiver_dir)
                        
                        # Ball trajectory should generally align with receiver direction (relaxed to 30%)
                        if alignment < 0.3:
                            if frame_num % 100 == 0:  # Debug output
                                print(f"[DEBUG] Frame {frame_num}: Poor alignment ({alignment:.2f})")
                            return passes
                        
                        # Valid pass - basic checks passed
                        pass_type = 'long' if pass_distance > self.short_pass_threshold else 'short'
                        success = receiver_dist < 60 and ball_speed > 2.0
                        
                        # Calculate confidence based on multiple factors
                        confidence = (
                            ball_conf * 0.3 +  # Ball detection confidence
                            min(1.0, alignment) * 0.4 +  # Trajectory alignment (most important)
                            min(1.0, ball_speed / 6.0) * 0.2 +  # Ball speed
                            (passer['confidence'] + receiver['confidence']) / 2 * 0.1  # Player confidence
                        )
                        
                        # Accept passes with reasonable confidence (lowered threshold)
                        if confidence > 0.4:
                            # Record pass event
                            pass_event = {
                                'frame': frame_num,
                                'passer_id': passer_id,
                                'receiver_id': receiver_id,
                                'team': passer_team,
                                'distance': pass_distance,
                                'type': pass_type,
                                'success': success,
                                'confidence': confidence,
                                'method': 'ball_tracking',
                                'alignment': alignment,
                                'ball_speed': ball_speed
                            }
                            passes.append(pass_event)
                            
                            # Update player statistics
                            if passer_id not in self.player_stats:
                                self.player_stats[passer_id] = {
                                    'passes': 0, 'successful_passes': 0,
                                    'received': 0, 'short_passes': 0, 'long_passes': 0,
                                    'team': passer_team
                                }
                            if receiver_id not in self.player_stats:
                                self.player_stats[receiver_id] = {
                                    'passes': 0, 'successful_passes': 0,
                                    'received': 0, 'short_passes': 0, 'long_passes': 0,
                                    'team': receiver_team
                                }
                            
                            self.player_stats[passer_id]['passes'] += 1
                            self.player_stats[receiver_id]['received'] += 1
                            if success:
                                self.player_stats[passer_id]['successful_passes'] += 1
                            if pass_type == 'short':
                                self.player_stats[passer_id]['short_passes'] += 1
                            else:
                                self.player_stats[passer_id]['long_passes'] += 1
                            
                            self.last_pass_frame[pair_key] = frame_num
                            self.last_pass_frame[passer_id] = frame_num
                            print(f"✅ Pass ({passer_team}): P{passer_id} → P{receiver_id} ({pass_type}, {pass_distance:.0f}px, spd={ball_speed:.1f}, align={alignment:.2f}, conf={confidence:.2f})")
                        else:
                            if frame_num % 50 == 0:  # More frequent debug for close calls
                                print(f"[DEBUG] Frame {frame_num}: Low confidence pass rejected (conf={confidence:.2f})")
        
        return passes
    
    def detect_passes_movement_based(self, players, frame_num):
        """STRICT fallback pass detection - only when ball tracking fails"""
        passes = []
        
        # Only use movement-based if ball is NOT being tracked
        if self.ball_tracking is not None:
            return passes  # Prefer ball-based detection
        
        if len(players) < 2:
            return passes
        
        # Check every 15 frames (less frequent)
        if frame_num % 15 != 0:
            return passes
        
        # Look for players with good history
        for p1_id, p1 in players.items():
            if p1_id not in self.player_history or len(self.player_history[p1_id]) < 20:
                continue
            
            p1_history = list(self.player_history[p1_id])
            p1_old = p1_history[-20]
            p1_recent_speeds = [h['speed'] for h in p1_history[-8:]]
            p1_recent_speed = np.mean(p1_recent_speeds)
            p1_old_speed = np.mean([h['speed'] for h in p1_history[-20:-10]])
            
            # Passer should have been moving, then slowed down (passing)
            speed_decrease = p1_old_speed - p1_recent_speed
            if speed_decrease < 1.5:  # Not enough deceleration
                continue
            
            for p2_id, p2 in players.items():
                if p2_id == p1_id or p2_id not in self.player_history or len(self.player_history[p2_id]) < 20:
                    continue
                
                # Check cooldown
                pair_key = tuple(sorted([p1_id, p2_id]))
                if pair_key in self.last_pass_frame:
                    if frame_num - self.last_pass_frame[pair_key] < self.pass_cooldown:
                        continue
                
                p2_history = list(self.player_history[p2_id])
                p2_old = p2_history[-20]
                p2_recent_speeds = [h['speed'] for h in p2_history[-8:]]
                p2_recent_speed = np.mean(p2_recent_speeds)
                
                # Calculate distances
                old_dist = np.sqrt((p1_old['x'] - p2_old['x'])**2 + (p1_old['y'] - p2_old['y'])**2)
                current_dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                dist_change = old_dist - current_dist
                
                # STRICT: Multiple validation checks
                if not (self.min_pass_distance < current_dist < self.max_pass_distance):
                    continue
                if old_dist < 100:  # Were too close
                    continue
                if dist_change < 25:  # Not getting closer enough
                    continue
                if p1_recent_speed < 1.5 or p2_recent_speed < 1.5:  # Not moving enough
                    continue
                
                # Check trajectory alignment - p1 moving toward p2
                p1_vel_x = p1['velocity_x']
                p1_vel_y = p1['velocity_y']
                
                # Direction from p1 to p2
                dx = p2['x'] - p1['x']
                dy = p2['y'] - p1['y']
                norm = np.sqrt(dx**2 + dy**2)
                
                if norm > 0:
                    dx /= norm
                    dy /= norm
                    
                    # Check if p1 is moving toward p2
                    p1_speed_val = np.sqrt(p1_vel_x**2 + p1_vel_y**2)
                    if p1_speed_val > 0:
                        p1_dir_x = p1_vel_x / p1_speed_val
                        p1_dir_y = p1_vel_y / p1_speed_val
                        alignment = p1_dir_x * dx + p1_dir_y * dy
                        
                        # STRICT: Better alignment required
                        if alignment > 0.4:  # At least 40% alignment
                            pass_distance = current_dist
                            pass_type = 'long' if pass_distance > self.short_pass_threshold else 'short'
                            
                            # Lower confidence for movement-based (less reliable)
                            confidence = (
                                (p1['confidence'] + p2['confidence']) / 2 * 0.5 +
                                alignment * 0.3 +
                                min(1.0, speed_decrease / 5.0) * 0.2
                            )
                            
                            # Only accept if ball tracking is unavailable
                            if confidence > 0.65:
                                passes.append({
                                    'frame': frame_num,
                                    'passer_id': p1_id,
                                    'receiver_id': p2_id,
                                    'distance': pass_distance,
                                    'type': pass_type,
                                    'success': True,  # Assume success for movement-based
                                    'confidence': confidence,
                                    'method': 'movement',
                                    'alignment': alignment
                                })
                                
                                self.last_pass_frame[pair_key] = frame_num
                                print(f"⚠️ Movement pass (no ball): P{p1_id} → P{p2_id} ({pass_type}, {pass_distance:.0f}px, align={alignment:.2f})")
        
        return passes
    
    def merge_pass_detections(self, passes_ball, passes_movement, frame_num):
        """Merge pass detections from different methods, avoiding duplicates"""
        all_passes = []
        
        # Add ball-based passes (higher priority)
        for pass_event in passes_ball:
            all_passes.append(pass_event)
        
        # Add movement-based passes if not duplicate
        for pass_event in passes_movement:
            pair_key = tuple(sorted([pass_event['passer_id'], pass_event['receiver_id']]))
            
            # Check if this pair already detected by ball method
            is_duplicate = False
            for existing_pass in all_passes:
                existing_pair = tuple(sorted([existing_pass['passer_id'], existing_pass['receiver_id']]))
                if existing_pair == pair_key and abs(existing_pass['frame'] - pass_event['frame']) < 20:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_passes.append(pass_event)
        
        return all_passes
    
    def calculate_accuracy(self, passes):
        """Calculate accuracy metrics"""
        if not passes:
            return self.get_empty_accuracy()
        
        df = pd.DataFrame(passes)
        
        # Overall accuracy based on confidence and success rate
        avg_confidence = df['confidence'].mean()
        success_rate = df['success'].mean()
        overall_accuracy = (avg_confidence * 0.7 + success_rate * 0.3)
        
        # Short pass metrics
        short_passes = df[df['type'] == 'short']
        short_accuracy = short_passes['confidence'].mean() if len(short_passes) > 0 else 0
        short_success = short_passes['success'].mean() if len(short_passes) > 0 else 0
        
        # Long pass metrics
        long_passes = df[df['type'] == 'long']
        long_accuracy = long_passes['confidence'].mean() if len(long_passes) > 0 else 0
        long_success = long_passes['success'].mean() if len(long_passes) > 0 else 0
        
        # Count successful passes
        successful_passes = [p for p in passes if p.get('success', False)]
        
        return {
            'overall': overall_accuracy,
            'short': short_accuracy,
            'long': long_accuracy,
            'success': success_rate,
            'total_passes': len(passes),
            'short_count': len(short_passes),
            'long_count': len(long_passes),
            'successful_count': len(successful_passes),
            'short_success_rate': short_success,
            'long_success_rate': long_success
        }
    
    def get_empty_accuracy(self):
        """Return empty accuracy metrics"""
        return {
            'overall': 0.0,
            'short': 0.0,
            'long': 0.0,
            'success': 0.0,
            'total_passes': 0,
            'short_count': 0,
            'long_count': 0,
            'successful_count': 0,
            'short_success_rate': 0.0,
            'long_success_rate': 0.0
        }
    
    def assign_player_id(self, detection, existing_players):
        """Assign player ID based on proximity"""
        if not existing_players:
            return 1
        
        min_distance = float('inf')
        closest_id = None
        
        for player_id, player in existing_players.items():
            player_x = player.get('x', player.get('raw_x', 0))
            player_y = player.get('y', player.get('raw_y', 0))
            
            distance = np.sqrt(
                (detection['center_x'] - player_x)**2 + 
                (detection['center_y'] - player_y)**2
            )
            
            if distance < min_distance and distance < 120:
                min_distance = distance
                closest_id = player_id
        
        if closest_id is not None:
            return closest_id
        else:
            return max(existing_players.keys()) + 1 if existing_players else 1
    
    def detect_jersey_color(self, frame, x1, y1, x2, y2):
        """Detect dominant jersey color in bounding box"""
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                return None
            
            # Ensure coordinates are valid
            h, w = frame.shape[:2]
            x1 = max(0, min(int(x1), w-1))
            y1 = max(0, min(int(y1), h-1))
            x2 = max(x1+1, min(int(x2), w))
            y2 = max(y1+1, min(int(y2), h))
            
            # Crop player region
            player_region = frame[y1:y2, x1:x2]
            if player_region.size == 0 or player_region.shape[0] < 2 or player_region.shape[1] < 2:
                return None
            
            # Focus on upper body (jersey area, top 40% of bbox)
            upper_height = max(1, int((y2 - y1) * 0.4))
            jersey_region = player_region[:upper_height, :]
            
            if jersey_region.size == 0 or jersey_region.shape[0] < 1 or jersey_region.shape[1] < 1:
                return None
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
            
            # Calculate mean HSV
            mean_h = np.mean(hsv[:,:,0])
            mean_s = np.mean(hsv[:,:,1])
            mean_v = np.mean(hsv[:,:,2])
            
            # Return HSV tuple for clustering
            return (mean_h, mean_s, mean_v)
        except Exception as e:
            # Silently handle errors (too many frames to log)
            return None
    
    def assign_team(self, detection, existing_players, frame=None):
        """Assign team based on jersey color clustering"""
        player_id = detection.get('player_id')
        
        # If player already has a team, keep it
        if player_id and player_id in self.team_colors:
            return self.team_colors[player_id]
        
        # Try to detect jersey color
        if frame is not None:
            x1, y1 = int(detection['x1']), int(detection['y1'])
            x2, y2 = int(detection['x2']), int(detection['y2'])
            jersey_color = self.detect_jersey_color(frame, x1, y1, x2, y2)
            
            if jersey_color:
                # Simple clustering: if we have existing players, find closest color
                if existing_players:
                    team_colors = {}
                    for pid, player in existing_players.items():
                        if 'jersey_color' in player and 'team' in player:
                            team = player['team']
                            if team not in team_colors:
                                team_colors[team] = []
                            team_colors[team].append(player['jersey_color'])
                    
                    # Find which team's color is closest
                    if team_colors:
                        min_dist = float('inf')
                        best_team = 'Team A'
                        
                        for team, colors in team_colors.items():
                            avg_color = np.mean(colors, axis=0)
                            dist = np.linalg.norm(np.array(jersey_color) - avg_color)
                            if dist < min_dist:
                                min_dist = dist
                                best_team = team
                        
                        if player_id:
                            self.team_colors[player_id] = best_team
                        return best_team, jersey_color
                
                # First players: assign alternately based on color difference
                if len(existing_players) == 0:
                    team = 'Team A'
                else:
                    # Find if any team exists
                    existing_teams = set(p.get('team') for p in existing_players.values() if 'team' in p)
                    if len(existing_teams) < 2:
                        team = 'Team A' if 'Team A' not in existing_teams else 'Team B'
                    else:
                        team = 'Team A'  # Default
                
                if player_id:
                    self.team_colors[player_id] = team
                return team, jersey_color
        
        # Fallback: position-based
        field_center = 640
        team = 'Team A' if detection['center_x'] < field_center else 'Team B'
        if player_id:
            self.team_colors[player_id] = team
        return team, None
    
    def visualize_frame(self, frame, detections, players_tracking, passes, frame_num, fps):
        """Draw bounding boxes, player IDs, ball, and pass lines"""
        # Validate inputs
        if frame is None or frame.size == 0:
            return None
        
        if fps is None or fps <= 0:
            fps = 30.0  # Fallback
        
        vis_frame = frame.copy()
        
        # Draw player bounding boxes
        for det in detections:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw ball with trajectory
        if self.ball_tracking:
            ball_x = int(self.ball_tracking['x'])
            ball_y = int(self.ball_tracking['y'])
            ball_conf = self.ball_tracking.get('confidence', 0)
            ball_speed = self.ball_tracking.get('speed', 0)
            is_predicted = self.ball_tracking.get('predicted', False)
            
            # Color based on confidence and prediction
            if is_predicted:
                ball_color = (0, 0, 255)  # Red for predicted
            elif ball_conf > 0.5:
                ball_color = (0, 255, 255)  # Yellow for high confidence
            else:
                ball_color = (0, 165, 255)  # Orange for medium
            
            # Draw ball
            cv2.circle(vis_frame, (ball_x, ball_y), 12, ball_color, -1)
            cv2.circle(vis_frame, (ball_x, ball_y), 12, (0, 0, 0), 2)
            
            # Draw ball label
            label = "BALL?" if is_predicted else "BALL"
            cv2.putText(vis_frame, label, (ball_x - 20, ball_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2)
            
            # Draw ball trajectory
            if len(self.ball_history) > 1:
                history_list = list(self.ball_history)
                for i in range(max(0, len(history_list) - 15), len(history_list) - 1):
                    pt1 = (int(history_list[i]['x']), int(history_list[i]['y']))
                    pt2 = (int(history_list[i+1]['x']), int(history_list[i+1]['y']))
                    alpha = (i - max(0, len(history_list) - 15)) / 15.0
                    color_intensity = int(255 * alpha)
                    cv2.line(vis_frame, pt1, pt2, (0, color_intensity, color_intensity), 2)
        
        # Draw players (color-coded by team)
        for player_id, player in players_tracking.items():
            x, y = int(player['x']), int(player['y'])
            team = player.get('team', 'Unknown')
            
            # Color by team: Team A = Blue, Team B = Red, Unknown = White
            if team == 'Team A':
                color = (255, 0, 0)  # Blue
            elif team == 'Team B':
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 255)  # White
            
            cv2.circle(vis_frame, (x, y), 8, color, -1)
            cv2.circle(vis_frame, (x, y), 8, (0, 0, 0), 2)  # Black border
            cv2.putText(vis_frame, f'P{player_id}', (x + 12, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw pass lines
        for pass_event in self.recent_passes:
            if 'passer_id' in pass_event and 'receiver_id' in pass_event:
                passer_id = pass_event['passer_id']
                receiver_id = pass_event['receiver_id']
                
                if passer_id in players_tracking and receiver_id in players_tracking:
                    passer = players_tracking[passer_id]
                    receiver = players_tracking[receiver_id]
                    
                    # Color: green for success, red for failure
                    color = (0, 255, 0) if pass_event.get('success', True) else (0, 0, 255)
                    thickness = 3 if pass_event.get('type') == 'long' else 2
                    
                    cv2.line(vis_frame, 
                            (int(passer['x']), int(passer['y'])),
                            (int(receiver['x']), int(receiver['y'])),
                            color, thickness)
                    
                    # Pass type label
                    mid_x = (int(passer['x']) + int(receiver['x'])) // 2
                    mid_y = (int(passer['y']) + int(receiver['y'])) // 2
                    pass_type = pass_event.get('type', 'unknown')
                    cv2.putText(vis_frame, pass_type.upper(), (mid_x, mid_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw info
        timestamp = frame_num / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        
        # Enhanced ball status with diagnostic info
        if self.ball_tracking:
            ball_speed = self.ball_tracking.get('speed', 0)
            ball_conf = self.ball_tracking.get('confidence', 0)
            is_predicted = self.ball_tracking.get('predicted', False)
            status_icon = "?" if is_predicted else "✓"
            ball_status = f"Ball {status_icon}: {ball_speed:.1f} px/f (conf:{ball_conf:.2f})"
        else:
            ball_status = "Ball: ❌ Not detected"
        
        # Check pass detection readiness
        ball_history_ok = len(self.ball_history) >= 8
        ready_icon = "✓" if (self.ball_tracking and ball_history_ok and len(players_tracking) >= 2) else "❌"
        
        info_text = [
            f"Frame: {frame_num} | Time: {minutes:02d}:{seconds:02d}",
            f"Players: {len(players_tracking)} | {ball_status}",
            f"Ball History: {len(self.ball_history)}/8",
            f"Pass Detection: {ready_icon} | Passes: {len(passes)}",
            f"Total Passes Detected: {len(self.recent_passes)}"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(vis_frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        height = vis_frame.shape[0]
        instructions = ["Press Q to quit", "Press Space to pause"]
        for i, text in enumerate(instructions):
            cv2.putText(vis_frame, text, (10, height - 40 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return vis_frame

# Initialize analyzer
analyzer = FixedFootballAnalysis()
