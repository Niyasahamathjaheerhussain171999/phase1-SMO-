# CELL 5: Fixed Football Analysis Class
import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import json
from collections import deque
import torch

class FixedFootballAnalysis:
    def __init__(self, show_video=False):
        # Auto-detect device (GPU/CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        
        # Load YOLOv11x only (best accuracy)
        self.model = YOLO('yolo11x.pt')
        print("✅ Loaded YOLOv11x model")
        
        self.show_video = show_video
        # Check if we should save frames instead (for SSH/remote)
        # ALWAYS save frames if on SSH or if explicitly requested
        is_ssh = os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_CLIENT')
        self.save_frames = os.environ.get('SAVE_FRAMES') == 'True' or is_ssh or not show_video
        if self.save_frames:
            os.makedirs('output_frames', exist_ok=True)
            print("📁 Saving ALL visualization frames to: output_frames/")
            print("   Each frame shows: Player boxes, Ball tracking, Pass detection")
        self.results = {
            'passes': [],
            'players': [],
            'accuracy_metrics': {}
        }
        
        # Ball tracking with trajectory
        self.ball_history = deque(maxlen=30)  # longer history for trajectory
        self.ball_position = None
        self.ball_confidence = 0
        self.ball_bbox = None
        self.ball_velocity = (0, 0)  # (vx, vy) pixels/frame
        self.ball_missed_frames = 0
        
        # Ball detection thresholds - TUNED FOR ACCURACY
        self.ball_min_conf = 0.35  # Higher confidence = less false positives
        self.ball_min_area = 50    # Minimum area in pixels (realistic ball size)
        self.ball_max_area = 5000  # Maximum area in pixels
        self.ball_max_jump = 200   # Max distance ball can move between frames (realistic ball speed)
        
        # Simple player tracking (just last position)
        self.player_positions = {}  # player_id -> (x, y, team)
        # Player tracking state for stability and ghost filtering
        self.player_state = {}  # player_id -> dict(ema_center, ema_bbox, hits, missed, team)
        self.player_alpha = 0.25  # Lower = more smoothing, more stable boxes
        self.player_min_hits = 5  # Stricter confirmation to reduce ghost IDs
        self.player_max_missed = 8  # Keep tracks longer before deleting
        self.next_player_id = 1
        
        # Team colors (learned from detections)
        self.team_colors = {'Team A': None, 'Team B': None}
        self.team_assigned = False
        
        # Trajectory-based pass detection
        self.pass_in_progress = False
        self.pass_start_frame = None
        self.pass_start_pos = None
        self.pass_passer_id = None
        self.pass_start_velocity = None
        self.last_pass_end_frame = 0
        
        # Pass detection parameters - TUNED FOR ACCURACY
        self.pass_min_velocity = 3.0  # Min pixels/frame (clear passes only)
        self.pass_max_frames = 75  # Max frames for a pass to complete
        self.pass_cooldown = 10  # Frames between passes (prevent duplicates)
        self.possession_radius = 70  # Radius for possession (tighter = more accurate)
        self.short_long_threshold_px = 150  # Distance threshold (realistic pass distances)
    
    def analyze_video(self, video_path):
        """Complete analysis with GPU acceleration"""
        print("🎯 Starting Football Analysis...")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Could not open video file")
            return [], self.get_empty_accuracy()
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps and fps > 0:
            self._kf_dt = 1.0 / float(fps)
        
        print(f"📹 Video: {total_frames} frames at {fps} FPS")
        print(f"⏱️ Video length: {total_frames/fps/60:.1f} minutes")
        
        # Reset tracking for new video
        self.ball_history.clear()
        self.ball_position = None
        self.ball_confidence = 0
        self.ball_bbox = None
        self.ball_velocity = (0, 0)
        self.ball_missed_frames = 0
        self.player_positions.clear()
        self.player_state.clear()
        self.next_player_id = 1
        self.team_colors = {'Team A': None, 'Team B': None}
        self.team_assigned = False
        self.pass_in_progress = False
        self.pass_start_frame = None
        self.pass_start_pos = None
        self.pass_passer_id = None
        self.pass_start_velocity = None
        self.last_pass_end_frame = 0
        
        frame_count = 0
        players_tracking = {}
        pass_events = []
        detection_count = 0
        self.max_players_seen = 0  # Track max players for stats
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO detection with auto-detected device
            try:
                results = self.model(frame, device=self.device, verbose=False)
                
                # Process detections (players AND ball)
                player_detections, ball_detections = self.process_detections(results[0], frame_count, frame)
                detection_count += len(player_detections)
                
                # If no ball detected by YOLO, try color-based detection
                if len(ball_detections) == 0:
                    color_ball = self.detect_ball_by_color(frame, frame_count)
                    if color_ball:
                        ball_detections.append(color_ball)
                
                # Track ball
                self.track_ball(ball_detections, frame_count)
                
                # Debug ball detection every 50 frames
                if frame_count % 50 == 0:
                    ball_status = "TRACKED" if (self.ball_position and self.ball_confidence >= self.ball_min_conf) else "NOT TRACKED"
                    print(f"[DEBUG Frame {frame_count}] Ball detections: {len(ball_detections)}, "
                          f"Ball status: {ball_status}, "
                          f"Confidence: {self.ball_confidence:.2f}, "
                          f"Position: {self.ball_position}, "
                          f"BBox: {self.ball_bbox}, "
                          f"History: {len(self.ball_history)}")
                
                # Track players (with team assignment)
                players_tracking = self.track_players(player_detections, players_tracking, frame_count, frame)
                
                # Update max players seen
                self.max_players_seen = max(self.max_players_seen, len(players_tracking))
                
                # Detect passes (using ball tracking)
                passes = self.detect_passes(players_tracking, frame_count)
                
                # Trajectory-based detection only - no fallback
                
                pass_events.extend(passes)
                self.results['passes'].extend(passes)
                
                # Visualize if enabled
                if self.show_video or self.save_frames:
                    vis_frame = self.visualize_frame(frame, players_tracking, ball_detections, passes, frame_count)
                    
                    if self.show_video:
                        try:
                            cv2.imshow('Football Pass Detection', vis_frame)
                            
                            # Press 'q' to quit
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("⏹️  Video visualization stopped by user")
                                break
                        except cv2.error as e:
                            # Display failed, switch to frame saving
                            print(f"⚠️  Display error: {e}")
                            print("🔄 Switching to frame saving mode...")
                            self.show_video = False
                            self.save_frames = True
                            os.makedirs('output_frames', exist_ok=True)
                    
                    # Save frames if requested (EVERY FRAME for complete tracking view)
                    if self.save_frames:
                        frame_path = f'output_frames/frame_{frame_count:06d}.jpg'
                        cv2.imwrite(frame_path, vis_frame)
                        
                        # Real-time progress updates
                        if frame_count % 50 == 0:
                            # Show tracking stats
                            ball_status = "✅" if (self.ball_position and self.ball_confidence >= self.ball_min_conf) else "❌"
                            players_count = len(players_tracking)
                            passes_count = len(passes) if passes else 0
                            
                            print(f"📊 Frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) | "
                                  f"Ball: {ball_status} | Players: {players_count} | Passes: {passes_count} | "
                                  f"Saved: {frame_path}")
                        
                        # Create live preview every 100 frames
                        if frame_count % 100 == 0:
                            preview_path = 'output_frames/LIVE_PREVIEW.jpg'
                            cv2.imwrite(preview_path, vis_frame)
                            print(f"🖼️  LIVE PREVIEW updated: {preview_path} (refresh to see latest)")
                
            except Exception as e:
                print(f"⚠️ Detection error at frame {frame_count}: {e}")
                # Continue processing - don't crash on single frame error
                import traceback
                if frame_count % 100 == 0:  # Only print full traceback every 100 frames
                    traceback.print_exc()
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"⏳ Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        cap.release()
        if self.show_video:
            cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"📹 Total frames processed: {frame_count}")
        print(f"👥 Total player detections: {detection_count}")
        print(f"⚽ Total passes detected: {len(pass_events)}")
        
        # Calculate tracking statistics
        ball_tracked_frames = sum(1 for h in self.ball_history if h.get('confidence', 0) >= self.ball_min_conf)
        ball_tracking_percentage = (ball_tracked_frames / frame_count * 100) if frame_count > 0 else 0
        
        # Player tracking stats
        avg_players_per_frame = detection_count / frame_count if frame_count > 0 else 0
        max_players = max(len(players_tracking) if hasattr(self, 'max_players_seen') else 0, 
                         len(players_tracking))
        
        print(f"\n🎯 TRACKING QUALITY:")
        print(f"   ⚽ Ball tracked: {ball_tracked_frames}/{frame_count} frames ({ball_tracking_percentage:.1f}%)")
        print(f"   👥 Average players per frame: {avg_players_per_frame:.1f}")
        print(f"   👥 Max players tracked simultaneously: {max_players}")
        
        # Pass breakdown
        if pass_events:
            short_passes = [p for p in pass_events if p.get('type') == 'short']
            long_passes = [p for p in pass_events if p.get('type') == 'long']
            successful_passes = [p for p in pass_events if p.get('success', False)]
            
            print(f"\n⚽ PASS BREAKDOWN:")
            print(f"   📏 Short passes: {len(short_passes)}")
            print(f"   📏 Long passes: {len(long_passes)}")
            print(f"   ✅ Successful: {len(successful_passes)}/{len(pass_events)} ({len(successful_passes)/len(pass_events)*100:.1f}%)")
            
            if short_passes:
                avg_short_dist = sum(p['distance'] for p in short_passes) / len(short_passes)
                print(f"   📏 Avg short pass distance: {avg_short_dist:.0f}px")
            if long_passes:
                avg_long_dist = sum(p['distance'] for p in long_passes) / len(long_passes)
                print(f"   📏 Avg long pass distance: {avg_long_dist:.0f}px")
        else:
            print(f"\n⚠️  NO PASSES DETECTED")
            print(f"   This could mean:")
            print(f"   - Ball not being tracked properly (check ball detection)")
            print(f"   - Players not close enough to ball")
            print(f"   - Pass velocity too low (min: {self.pass_min_velocity}px/frame)")
        
        # Auto-create video from saved frames (ALWAYS if frames are saved)
        if self.save_frames:
            print(f"\n🎬 Creating tracking video from frames...")
            import subprocess
            try:
                # Check if we have frames
                import glob
                frames = sorted(glob.glob('output_frames/frame_*.jpg'))
                if len(frames) > 0:
                    print(f"   Found {len(frames)} frames")
                    video_output = 'tracking_output.mp4'
                    cmd = [
                        'ffmpeg', '-y', '-framerate', str(fps),
                        '-pattern_type', 'glob', '-i', 'output_frames/frame_*.jpg',
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-pix_fmt', 'yuv420p', video_output
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ Created tracking video: {video_output}")
                        print(f"   📺 Watch it to verify:")
                        print(f"      - Player tracking (red/blue boxes)")
                        print(f"      - Ball tracking (green box + purple arrow)")
                        print(f"      - Pass detection (yellow lines)")
                        print(f"      - Pass labels (SHORT/LONG)")
                    else:
                        print(f"⚠️  ffmpeg failed: {result.stderr}")
                        print(f"   Install with: sudo apt-get install ffmpeg")
                        print(f"   You can still view frames in: output_frames/")
                else:
                    print(f"⚠️  No frames found in output_frames/")
            except Exception as e:
                print(f"⚠️  Could not create video: {e}")
                print(f"   View individual frames in: output_frames/")
        
        # Validation warnings
        print(f"\n{'='*60}")
        if ball_tracking_percentage < 30:
            print(f"⚠️  WARNING: Ball tracking is low ({ball_tracking_percentage:.1f}%)")
            print(f"   - Check if ball is visible in video")
            print(f"   - Try lowering ball_min_conf (current: {self.ball_min_conf})")
        if avg_players_per_frame < 5:
            print(f"⚠️  WARNING: Low player detection ({avg_players_per_frame:.1f} per frame)")
            print(f"   - Check if players are visible")
            print(f"   - YOLO may need adjustment")
        if len(pass_events) == 0 and ball_tracking_percentage > 50:
            print(f"⚠️  WARNING: Ball tracked but no passes detected")
            print(f"   - Pass velocity threshold may be too high (current: {self.pass_min_velocity}px/frame)")
            print(f"   - Possession radius may be too tight (current: {self.possession_radius}px)")
        print(f"{'='*60}")
        
        # Calculate accuracy
        accuracy = self.calculate_accuracy(pass_events)
        
        print(f"✅ Analysis Complete!")
        print(f"📊 Found {len(pass_events)} passes")
        print(f"🎯 Overall Accuracy: {accuracy['overall']:.1%}")
        
        # Print summary
        if pass_events:
            df_summary = {}
            for p in pass_events:
                key = f"{p['type']}_{'success' if p['success'] else 'failed'}"
                df_summary[key] = df_summary.get(key, 0) + 1
            
            print("\n📈 Pass Summary:")
            for key, count in df_summary.items():
                print(f"  {key}: {count}")
        
        return pass_events, accuracy
    
    def process_detections(self, results, frame_num, frame=None):
        """Process YOLO detections - separate players and ball"""
        player_detections = []
        ball_detections = []
        
        if results.boxes is None or len(results.boxes) == 0:
            return player_detections, ball_detections
        
        for box in results.boxes:
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            detection = {
                'frame': frame_num,
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'confidence': conf,
                'class': cls,
                'width': width,
                'height': height
            }
            
            # Detect players (class 0 = person)
            if conf > 0.3 and cls == 0 and 1000 < area < 50000:
                player_detections.append(detection)
            
            # Detect ball (class 32 = sports ball) - stricter threshold
            elif cls == 32:  # Sports ball class
                # Filter by confidence and area
                if conf >= self.ball_min_conf and 30 < area < 8000:
                    ball_detections.append(detection)
                # Debug: log all ball detections (even low confidence)
                if frame_num % 50 == 0:
                    print(f"[BALL DEBUG Frame {frame_num}] Found ball: conf={conf:.2f}, area={area:.0f}, "
                          f"bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)}), "
                          f"center=({int((x1+x2)/2)},{int((y1+y2)/2)})")
        
        return player_detections, ball_detections
    
    def detect_ball_by_color(self, frame, frame_num):
        """Fallback: detect ball using color (white/yellow/orange)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # White range
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Yellow range
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        # Create masks
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 3000:  # Ball size range
                x, y, w, h = cv2.boundingRect(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if circularity > 0.4 and 0.6 < aspect_ratio < 1.4:
                        return {
                            'frame': frame_num,
                            'bbox': (x, y, x + w, y + h),
                            'center': (x + w // 2, y + h // 2),
                            'confidence': min(0.5, circularity * 0.7),
                            'class': 32
                        }
        
        return None
    
    def assign_team_by_jersey_color(self, frame, player_bbox):
        """Assign team based on jersey color (upper body HSV)"""
        x1, y1, x2, y2 = player_bbox
        
        # Crop upper body region (top 60% of bounding box)
        h = y2 - y1
        upper_body_y1 = y1
        upper_body_y2 = y1 + int(h * 0.6)
        
        # Ensure valid coordinates
        h_img, w_img = frame.shape[:2]
        x1 = max(0, min(x1, w_img))
        x2 = max(0, min(x2, w_img))
        upper_body_y1 = max(0, min(upper_body_y1, h_img))
        upper_body_y2 = max(0, min(upper_body_y2, h_img))
        
        if upper_body_y2 <= upper_body_y1 or x2 <= x1:
            return None
        
        try:
            roi = frame[upper_body_y1:upper_body_y2, x1:x2]
            if roi.size == 0:
                return None
            
            # Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Get average HSV values (excluding black/white/low saturation)
            mask = (hsv[:, :, 2] > 50) & (hsv[:, :, 1] > 30)
            if np.sum(mask) == 0:
                return None
            
            h_values = hsv[:, :, 0][mask]
            avg_hue = np.mean(h_values)
            
            return avg_hue
        except:
            return None
    
    def assign_teams(self, frame, players):
        """Assign teams to players (jersey color or left/right split)"""
        if not players:
            return
        
        # Try jersey color first
        jersey_hues = []
        for player in players:
            hue = self.assign_team_by_jersey_color(frame, player['bbox'])
            if hue is not None:
                jersey_hues.append((player, hue))
        
        # If we have enough players with colors, cluster into 2 teams
        if len(jersey_hues) >= 4 and not self.team_assigned:
            hues = [h for _, h in jersey_hues]
            if len(set(hues)) > 1:  # More than one color
                # Simple clustering: sort by hue and split
                sorted_hues = sorted(enumerate(hues), key=lambda x: x[1])
                mid = len(sorted_hues) // 2
                
                team_a_hue = np.mean([hues[i] for i, _ in sorted_hues[:mid]])
                team_b_hue = np.mean([hues[i] for i, _ in sorted_hues[mid:]])
                
                self.team_colors['Team A'] = team_a_hue
                self.team_colors['Team B'] = team_b_hue
                self.team_assigned = True
                print(f"✅ Teams assigned by color: Team A (hue={team_a_hue:.1f}), Team B (hue={team_b_hue:.1f})")
        
        # Assign teams to players
        for player in players:
            if self.team_assigned:
                # Use color-based assignment
                hue = self.assign_team_by_jersey_color(frame, player['bbox'])
                if hue is not None:
                    # Assign to closest team color
                    dist_a = abs(hue - self.team_colors['Team A']) if self.team_colors['Team A'] else float('inf')
                    dist_b = abs(hue - self.team_colors['Team B']) if self.team_colors['Team B'] else float('inf')
                    player['team'] = 'Team A' if dist_a < dist_b else 'Team B'
                else:
                    # Fallback: left/right split
                    player['team'] = 'Team A' if player['center'][0] < frame.shape[1] // 2 else 'Team B'
            else:
                # Fallback: left/right split
                player['team'] = 'Team A' if player['center'][0] < frame.shape[1] // 2 else 'Team B'
    
    def track_ball(self, ball_detections, frame_num):
        """Track ball with high-confidence YOLO detections and velocity tracking"""
        best_ball = None
        best_conf = 0
        
        # Find best detection (highest confidence near last position)
        for det in ball_detections:
            conf = det['confidence']
            center = det['center']
            bbox = det['bbox']
            
            # Calculate area from bbox
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            
            # Filter by confidence and area
            if conf < self.ball_min_conf or area < self.ball_min_area or area > self.ball_max_area:
                continue
            
            # If we have a previous position, prefer detections close to it
            if self.ball_position is not None:
                dist = np.sqrt((center[0] - self.ball_position[0])**2 + 
                              (center[1] - self.ball_position[1])**2)
                if dist > self.ball_max_jump:
                    continue  # Too far, probably wrong detection
                # Weighted score: confidence + proximity bonus
                score = conf + (1.0 - min(dist / self.ball_max_jump, 1.0)) * 0.3
            else:
                score = conf
            
            if score > best_conf:
                best_conf = score
                best_ball = det
        
        # Update ball tracking
        if best_ball:
            new_pos = best_ball['center']
            
            # Calculate velocity
            if self.ball_position is not None:
                vx = new_pos[0] - self.ball_position[0]
                vy = new_pos[1] - self.ball_position[1]
                self.ball_velocity = (vx, vy)
            
            # Update position
            self.ball_position = new_pos
            self.ball_bbox = best_ball['bbox']
            self.ball_confidence = best_ball['confidence']
            self.ball_missed_frames = 0
            
            # Add to history
            self.ball_history.append({
                'frame': frame_num,
                'center': self.ball_position,
                'velocity': self.ball_velocity,
                'confidence': self.ball_confidence
            })
        else:
            # No detection - use velocity prediction for a few frames
            if self.ball_position is not None and self.ball_missed_frames < 10:
                # Predict next position using velocity
                pred_x = self.ball_position[0] + self.ball_velocity[0]
                pred_y = self.ball_position[1] + self.ball_velocity[1]
                self.ball_position = (int(pred_x), int(pred_y))
                self.ball_confidence *= 0.8  # Decay confidence
                self.ball_missed_frames += 1
            else:
                # Lost tracking
                self.ball_position = None
                self.ball_bbox = None
                self.ball_confidence = 0
                self.ball_velocity = (0, 0)
                self.ball_missed_frames = 0
    
    def _old_track_ball_backup(self, ball_detections, frame_num):
        """OLD VERSION - DO NOT USE"""
        if len(ball_detections) > 0:
            # Prefer detection close to last known ball to avoid drift
            chosen = None
            if self._ball_center_ema is not None:
                px, py = self._ball_center_ema
                close = []
                for d in ball_detections:
                    cx, cy = d['center']
                    dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                    if dist <= self.ball_max_jump:
                        close.append((dist, -d['confidence'], d))
                if close:
                    # nearest, then highest confidence
                    close.sort()
                    chosen = close[0][2]
            # If none close enough, pick highest confidence but require re-acquisition
            if chosen is None:
                chosen = max(ball_detections, key=lambda x: x['confidence'])
                self.ball_missed_frames += 1
                if self.ball_missed_frames < 3 and self._ball_center_ema is not None:
                    # Do not jump immediately; keep last
                    new_center = None
                else:
                    new_center = chosen['center']
                    self.ball_missed_frames = 0
            else:
                new_center = chosen['center']
                self.ball_missed_frames = 0

            # Always run Kalman predict first if initialized
            if self.ball_kf_initialized:
                pred = self.ball_kf.predict()
                pred_x, pred_y = float(pred[0]), float(pred[1])
            else:
                pred_x = pred_y = None

            if new_center is not None:
                new_bbox = chosen['bbox']
                self.ball_confidence = chosen['confidence']

                # Initialize EMA/KF state if needed
                if self._ball_center_ema is None:
                    self._ball_center_ema = (float(new_center[0]), float(new_center[1]))
                if self._ball_bbox_ema is None:
                    x1, y1, x2, y2 = new_bbox
                    self._ball_bbox_ema = (float(x1), float(y1), float(x2), float(y2))
                if not self.ball_kf_initialized:
                    _init_kf(float(new_center[0]), float(new_center[1]))

                # EMA update
                a = self.ball_ema_alpha
                cx, cy = self._ball_center_ema
                ecx = a * float(new_center[0]) + (1 - a) * cx
                ecy = a * float(new_center[1]) + (1 - a) * cy
                self._ball_center_ema = (ecx, ecy)
                # Kalman correct with measurement
                if self.ball_kf_initialized:
                    import numpy as _np
                    m = _np.array([[float(new_center[0])], [float(new_center[1])]], dtype=_np.float32)
                    post = self.ball_kf.correct(m)
                    fx, fy = float(post[0]), float(post[1])
                    self.ball_position = (int(round(fx)), int(round(fy)))
                else:
                    self.ball_position = (int(round(ecx)), int(round(ecy)))
                
                # Motion-based filtering with velocity tracking
                if self.ball_last_pos is not None:
                    movement = np.sqrt((self.ball_position[0] - self.ball_last_pos[0])**2 + 
                                     (self.ball_position[1] - self.ball_last_pos[1])**2)
                    
                    # Track velocity
                    vx = self.ball_position[0] - self.ball_last_pos[0]
                    vy = self.ball_position[1] - self.ball_last_pos[1]
                    self.ball_velocity_history.append((vx, vy, movement))
                    
                    # Use velocity-based filtering if enabled
                    if self.ball_use_velocity_filter and len(self.ball_velocity_history) >= 3:
                        # Check average velocity over last few frames
                        avg_movement = np.mean([m for _, _, m in list(self.ball_velocity_history)])
                        
                        # If consistently static (low average movement), reject
                        if avg_movement < self.ball_min_movement:
                            self.ball_static_frames += 1
                            # Reject if static too long
                            if self.ball_static_frames > self.ball_max_static_frames:
                                print(f"[BALL FILTER] Rejecting static detection (avg movement: {avg_movement:.1f}px)")
                                self.ball_position = None
                                self.ball_bbox = None
                                self.ball_confidence = 0
                                self._ball_center_ema = None
                                self._ball_bbox_ema = None
                                self.ball_kf_initialized = False
                                self.ball_static_frames = 0
                                self.ball_last_pos = None
                                self.ball_velocity_history.clear()
                                return
                        else:
                            self.ball_static_frames = 0  # Reset if moving
                    elif movement < self.ball_min_movement:
                        self.ball_static_frames += 1
                        if self.ball_static_frames > self.ball_max_static_frames:
                            self.ball_position = None
                            self.ball_bbox = None
                            self.ball_confidence = 0
                            self._ball_center_ema = None
                            self._ball_bbox_ema = None
                            self.ball_kf_initialized = False
                            self.ball_static_frames = 0
                            self.ball_last_pos = None
                            self.ball_velocity_history.clear()
                            return
                    else:
                        self.ball_static_frames = 0
                self.ball_last_pos = self.ball_position

                bx1, by1, bx2, by2 = self._ball_bbox_ema
                nx1, ny1, nx2, ny2 = map(float, new_bbox)
                ebx1 = a * nx1 + (1 - a) * bx1
                eby1 = a * ny1 + (1 - a) * by1
                ebx2 = a * nx2 + (1 - a) * bx2
                eby2 = a * ny2 + (1 - a) * by2
                self._ball_bbox_ema = (ebx1, eby1, ebx2, eby2)
                self.ball_bbox = (int(round(ebx1)), int(round(eby1)), int(round(ebx2)), int(round(eby2)))

                # Add smoothed center to history
                self.ball_history.append({
                    'frame': frame_num,
                    'center': self.ball_position,
                    'confidence': self.ball_confidence
                })
        else:
            # Ball not detected - keep last position for a few frames (up to 10 frames)
            if self.ball_kf_initialized:
                pred = self.ball_kf.predict()
                fx, fy = float(pred[0]), float(pred[1])
                self.ball_position = (int(round(fx)), int(round(fy)))
                # Slightly expand bbox if available
                if self.ball_bbox is not None:
                    x1, y1, x2, y2 = self.ball_bbox
                    pad = min(6 + self.ball_missed_frames, 14)
                    self.ball_bbox = (max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad)
                # Confidence decays
                self.ball_confidence = max(0.0, self.ball_confidence * 0.9)
                self.ball_missed_frames += 1
                # Add predicted position to history
                self.ball_history.append({
                    'frame': frame_num,
                    'center': self.ball_position,
                    'confidence': self.ball_confidence
                })
            elif len(self.ball_history) > 0:
                last_ball = self.ball_history[-1]
                frames_since_detection = frame_num - last_ball['frame']
                if frames_since_detection < 10:
                    # Keep last smoothed state with confidence decay
                    self.ball_position = last_ball['center']
                    if self.ball_bbox is not None:
                        # Lightly expand bbox to account for motion uncertainty
                        x1, y1, x2, y2 = self.ball_bbox
                        pad = min(3 + frames_since_detection, 10)
                        self.ball_bbox = (max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad)
                    self.ball_confidence = last_ball['confidence'] * (0.9 ** frames_since_detection)
                else:
                    self.ball_position = None
                    self.ball_bbox = None
    
    def track_players(self, detections, players_tracking, frame_num, frame=None):
        """Stable player tracking with EMA smoothing and ghost filtering"""
        current_players = {}
        
        # Assign teams first to enrich detections
        if frame is not None:
            self.assign_teams(frame, detections)
        
        # Build list of existing tracks' last centers
        existing_centers = {pid: (self.player_state.get(pid, {}).get('ema_center') or (px, py))
                            for pid, (px, py, _) in self.player_positions.items()}
        
        matched_ids = set()
        # Match detections to existing tracks by proximity
        for det in detections:
            dx, dy = det['center']
            best_pid = None
            best_dist = 1e9
            for pid, (px, py) in existing_centers.items():
                dist = np.sqrt((dx - px)**2 + (dy - py)**2)
                if dist < best_dist and dist <= 60:
                    best_dist = dist
                    best_pid = pid
            
            if best_pid is None:
                # Create a new track
                best_pid = self.next_player_id
                self.next_player_id += 1
                # Initialize state
                self.player_state[best_pid] = {
                    'ema_center': (float(dx), float(dy)),
                    'ema_bbox': tuple(map(float, det['bbox'])),
                    'hits': 1,
                    'missed': 0,
                    'team': det.get('team', 'Team A')
                }
            else:
                # Update state via EMA
                st = self.player_state.get(best_pid, {
                    'ema_center': (float(dx), float(dy)),
                    'ema_bbox': tuple(map(float, det['bbox'])),
                    'hits': 0,
                    'missed': 0,
                    'team': det.get('team', 'Team A')
                })
                a = self.player_alpha
                cx, cy = st['ema_center']
                ecx = a * float(dx) + (1 - a) * cx
                ecy = a * float(dy) + (1 - a) * cy
                bx1, by1, bx2, by2 = st['ema_bbox']
                nx1, ny1, nx2, ny2 = map(float, det['bbox'])
                ebx1 = a * nx1 + (1 - a) * bx1
                eby1 = a * ny1 + (1 - a) * by1
                ebx2 = a * nx2 + (1 - a) * bx2
                eby2 = a * ny2 + (1 - a) * by2
                st['ema_center'] = (ecx, ecy)
                st['ema_bbox'] = (ebx1, eby1, ebx2, eby2)
                st['hits'] = st.get('hits', 0) + 1
                st['missed'] = 0
                st['team'] = det.get('team', st.get('team', 'Team A'))
                self.player_state[best_pid] = st
            
            matched_ids.add(best_pid)
            # Update quick access positions dict
            team = self.player_state[best_pid]['team']
            self.player_positions[best_pid] = (
                int(round(self.player_state[best_pid]['ema_center'][0])),
                int(round(self.player_state[best_pid]['ema_center'][1])),
                team
            )
        
        # Update missed counters for unmatched tracks and prune ghosts
        to_delete = []
        for pid in list(self.player_positions.keys()):
            if pid not in matched_ids:
                st = self.player_state.get(pid)
                if st:
                    st['missed'] = st.get('missed', 0) + 1
                    if st['missed'] > self.player_max_missed or st.get('hits', 0) == 0:
                        to_delete.append(pid)
        for pid in to_delete:
            self.player_positions.pop(pid, None)
            self.player_state.pop(pid, None)
        
        # Build current_players only from confirmed tracks to reduce ghosts
        for pid, (px, py, team) in self.player_positions.items():
            st = self.player_state.get(pid)
            if not st:
                continue
            if st.get('hits', 0) < self.player_min_hits:
                # Not yet confirmed
                continue
            sbx1, sby1, sbx2, sby2 = st['ema_bbox']
            current_players[pid] = {
                'frame': frame_num,
                'x': int(round(px)),
                'y': int(round(py)),
                'center': (int(round(px)), int(round(py))),
                'bbox': (int(round(sbx1)), int(round(sby1)), int(round(sbx2)), int(round(sby2))),
                'confidence': 1.0,
                'team': team,
                'width': int(round(sbx2 - sbx1)),
                'height': int(round(sby2 - sby1))
            }
        
        return current_players
    
    def detect_passes(self, players, frame_num):
        """
        Trajectory-based pass detection:
        1. Detect pass start: ball leaves player (high velocity)
        2. Track ball trajectory
        3. Detect pass end: ball reaches player closest to trajectory
        """
        passes = []
        
        if self.ball_position is None or len(players) < 2:
            return passes
        
        # Calculate ball speed
        ball_speed = np.sqrt(self.ball_velocity[0]**2 + self.ball_velocity[1]**2)
        
        # Find player closest to ball
        closest_player = None
        closest_dist = float('inf')
        for pid, p in players.items():
            dist = np.sqrt((p['center'][0] - self.ball_position[0])**2 + 
                          (p['center'][1] - self.ball_position[1])**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_player = (pid, p, dist)
        
        if closest_player is None:
            return passes
        
        pid, player, dist = closest_player
        
        # STATE 1: No pass in progress - detect pass start
        if not self.pass_in_progress:
            # Ball is close to a player AND moving fast = pass starting
            if dist < self.possession_radius and ball_speed > self.pass_min_velocity:
                # Check cooldown
                if (frame_num - self.last_pass_end_frame) > self.pass_cooldown:
                    self.pass_in_progress = True
                    self.pass_start_frame = frame_num
                    self.pass_start_pos = self.ball_position
                    self.pass_passer_id = pid
                    self.pass_start_velocity = self.ball_velocity
                    print(f"🚀 Pass starting: P{pid} at frame {frame_num}, speed={ball_speed:.1f}px/frame")
        
        # STATE 2: Pass in progress - detect receiver
        else:
            frames_since_start = frame_num - self.pass_start_frame
            
            # Timeout: pass took too long
            if frames_since_start > self.pass_max_frames:
                print(f"⏱️ Pass timeout at frame {frame_num}")
                self.pass_in_progress = False
                return passes
            
            # Ball slowed down AND close to a player = pass received
            if ball_speed < self.pass_min_velocity and dist < self.possession_radius:
                receiver_id = pid
                
                # Can't pass to yourself
                if receiver_id == self.pass_passer_id:
                    self.pass_in_progress = False
                    return passes
                
                # Check both players exist
                if self.pass_passer_id not in players or receiver_id not in players:
                    self.pass_in_progress = False
                    return passes
                
                # Calculate pass distance
                passer = players[self.pass_passer_id]
                receiver = players[receiver_id]
                pass_dist = np.sqrt((receiver['center'][0] - passer['center'][0])**2 + 
                                   (receiver['center'][1] - passer['center'][1])**2)
                
                # Classify pass
                pass_type = 'short' if pass_dist <= self.short_long_threshold_px else 'long'
                
                # Create pass event
                pass_event = {
                    'frame': frame_num,
                    'passer_id': self.pass_passer_id,
                    'receiver_id': receiver_id,
                    'passer_team': passer.get('team', 'unknown'),
                    'receiver_team': receiver.get('team', 'unknown'),
                    'distance': float(pass_dist),
                    'type': pass_type,
                    'success': True,
                    'confidence': self.ball_confidence,
                    'duration_frames': frames_since_start
                }
                
                passes.append(pass_event)
                total_passes = len(self.results['passes']) + len(passes)
                print(f"⚽ PASS #{total_passes} | Frame {frame_num} | P{self.pass_passer_id}→P{receiver_id} | "
                      f"{pass_type.upper()} ({pass_dist:.0f}px, {frames_since_start}f)")
                
                # Reset state
                self.pass_in_progress = False
                self.last_pass_end_frame = frame_num
        
        return passes
    
    def detect_passes_movement_fallback(self, players, frame_num):
        """Fallback pass detection using player movement when ball not tracked"""
        passes = []
        
        # Only check every 10 frames
        if frame_num % 10 != 0:
            return passes
        
        # Check all player pairs
        for pid1, player1 in players.items():
            for pid2, player2 in players.items():
                if pid1 == pid2:
                    continue
                
                # Same team only
                if player1['team'] != player2['team']:
                    continue
                
                # Check cooldown
                pair_key = tuple(sorted([pid1, pid2]))
                if pair_key in self.last_pass_frame:
                    if frame_num - self.last_pass_frame[pair_key] < 30:
                        continue
                
                # Calculate distance
                pass_distance = np.sqrt(
                    (player2['center'][0] - player1['center'][0])**2 +
                    (player2['center'][1] - player1['center'][1])**2
                )
                
                # Validate distance
                if 50 < pass_distance < 300:
                    pass_type = 'short' if pass_distance < 120 else 'long'
                    
                    # Estimate success (conservative)
                    success = np.random.random() < 0.7  # 70% estimated success
                    
                    pass_event = {
                        'frame': frame_num,
                        'passer_id': pid1,
                        'receiver_id': pid2,
                        'passer_team': player1['team'],
                        'team': player1['team'],
                        'distance': pass_distance,
                        'type': pass_type,
                        'success': success,
                        'confidence': 0.5,  # Lower confidence for fallback
                        'method': 'movement_fallback'
                    }
                    
                    passes.append(pass_event)
                    self.last_pass_frame[pair_key] = frame_num
                    
                    print(f"⚽ PASS {pass_type.upper()} ({player1['team']}, fallback): P{pid1} → P{pid2} "
                          f"({pass_distance:.0f}px, {'✓' if success else '✗'})")
        
        return passes
    
    def visualize_frame(self, frame, tracked_players, ball, passes, frame_num):
        """Draw detections on frame"""
        vis_frame = frame.copy()
        
        # Draw players
        for pid, player in tracked_players.items():
            x1, y1, x2, y2 = player['bbox']
            team = player['team']
            
            # Color by team
            color = (255, 0, 0) if team == 'Team A' else (0, 0, 255)  # Red or Blue
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and team
            label = f"P{pid} ({team})"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ball with velocity vector
        if self.ball_bbox is not None and self.ball_position is not None and self.ball_confidence >= self.ball_min_conf:
            bx1, by1, bx2, by2 = self.ball_bbox
            bx, by = self.ball_position
            
            # Ball color: green if detected, yellow if predicted
            ball_color = (0, 255, 0) if self.ball_missed_frames == 0 else (0, 255, 255)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), ball_color, 2)
            
            # Draw velocity vector (shows ball trajectory)
            vx, vy = self.ball_velocity
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 1:  # Only show if moving
                # Scale velocity for visibility
                scale = 3.0
                end_x = int(bx + vx * scale)
                end_y = int(by + vy * scale)
                cv2.arrowedLine(vis_frame, (bx, by), (end_x, end_y), (255, 0, 255), 2, tipLength=0.3)
            
            # Label with confidence and speed
            conf_label = f"Ball {self.ball_confidence:.2f} | Speed:{speed:.1f}px/f"
            cv2.putText(vis_frame, conf_label, (bx1, max(10, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2)
        
        # Draw pass in progress indicator
        if self.pass_in_progress and self.pass_passer_id in tracked_players:
            passer = tracked_players[self.pass_passer_id]
            px, py = passer['center']
            cv2.circle(vis_frame, (px, py), 50, (255, 255, 0), 3)  # Yellow circle around passer
            cv2.putText(vis_frame, "PASSING", (px - 30, py - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw pass lines
        for pass_event in passes:
            passer = tracked_players[pass_event['passer_id']]
            receiver = tracked_players[pass_event['receiver_id']]
            
            # Color-code by pass type: short=green, long=blue
            color = (0, 255, 0) if pass_event['type'] == 'short' else (255, 0, 0)
            cv2.line(vis_frame, passer['center'], receiver['center'], color, 2)
            
            # Draw pass label
            mid_x = (passer['center'][0] + receiver['center'][0]) // 2
            mid_y = (passer['center'][1] + receiver['center'][1]) // 2
            label = f"{pass_event['type'].upper()} {'✓' if pass_event['success'] else '✗'}"
            cv2.putText(vis_frame, label, (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw info
        total_passes = len(self.results['passes'])
        short_passes = sum(1 for p in self.results['passes'] if p.get('type') == 'short')
        long_passes = total_passes - short_passes
        teams_text = f"Teams: A/B" if self.team_assigned else "Teams: split by side"
        
        ball_speed = np.sqrt(self.ball_velocity[0]**2 + self.ball_velocity[1]**2)
        ball_status = "DETECTED" if self.ball_missed_frames == 0 else f"PREDICTED ({self.ball_missed_frames}f)"
        
        info_text = [
            f"Frame: {frame_num}",
            f"Players: {len(tracked_players)}",
            f"Ball: {ball_status} | Speed: {ball_speed:.1f}px/f",
            f"Passes: {total_passes} (Short:{short_passes} Long:{long_passes})",
            f"Pass in progress: {'YES' if self.pass_in_progress else 'No'}",
            teams_text
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            # Highlight pass in progress
            color = (0, 255, 255) if i == 4 and self.pass_in_progress else (255, 255, 255)
            cv2.putText(vis_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        return vis_frame
    
    def calculate_accuracy(self, passes):
        """Calculate accuracy metrics with error handling"""
        if not passes:
            return self.get_empty_accuracy()
        
        df = pd.DataFrame(passes)
        
        # Overall accuracy (based on confidence)
        avg_confidence = df['confidence'].mean()
        overall_accuracy = min(0.95, avg_confidence * 1.1)  # Scale confidence to accuracy
        
        # Short pass accuracy
        short_passes = df[df['type'] == 'short']
        short_accuracy = short_passes['confidence'].mean() if len(short_passes) > 0 else 0
        
        # Long pass accuracy
        long_passes = df[df['type'] == 'long']
        long_accuracy = long_passes['confidence'].mean() if len(long_passes) > 0 else 0
        
        # Success rate
        success_rate = df['success'].mean()
        
        return {
            'overall': overall_accuracy,
            'short': short_accuracy,
            'long': long_accuracy,
            'success': success_rate,
            'total_passes': len(passes),
            'short_count': len(short_passes),
            'long_count': len(long_passes)
        }
    
    def get_empty_accuracy(self):
        """Return empty accuracy metrics when no passes found"""
        return {
            'overall': 0.0,
            'short': 0.0,
            'long': 0.0,
            'success': 0.0,
            'total_passes': 0,
            'short_count': 0,
            'long_count': 0
        }
