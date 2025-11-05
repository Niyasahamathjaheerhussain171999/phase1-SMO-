"""
Robust Pass Detection Module
Integrates tightly with ball tracking for accurate pass detection
"""

import numpy as np
from collections import deque


class PassDetector:
    """Accurate pass detection using ball tracking and player movement"""
    
    def __init__(self, 
                 min_pass_distance=40,      # Minimum pass distance (pixels)
                 max_pass_distance=350,     # Maximum pass distance (pixels)
                 short_pass_threshold=120,  # Short vs long pass threshold (pixels)
                 min_ball_speed=2.0,        # Minimum ball speed for pass (px/frame)
                 min_ball_travel=30,        # Minimum ball travel distance (pixels)
                 min_alignment=0.4,         # Minimum trajectory alignment (0-1)
                 min_confidence=0.5,        # Minimum pass confidence (0-1)
                 pass_cooldown=30,          # Frames between pass detections
                 ball_history_size=6,        # RELAXED: Frames to look back (was 10, now 6)
                 max_ball_to_player_dist=70 # Max distance from ball to player (pixels)
                 ):
        """
        Initialize pass detector with configurable parameters
        """
        self.min_pass_distance = min_pass_distance
        self.max_pass_distance = max_pass_distance
        self.short_pass_threshold = short_pass_threshold
        self.min_ball_speed = min_ball_speed
        self.min_ball_travel = min_ball_travel
        self.min_alignment = min_alignment
        self.min_confidence = min_confidence
        self.pass_cooldown = pass_cooldown
        self.ball_history_size = ball_history_size
        self.max_ball_to_player_dist = max_ball_to_player_dist
        
        self.last_pass_frame = {}  # Track last pass per player pair
        self.pass_history = deque(maxlen=100)  # Store recent passes for validation
        
    def _calculate_trajectory_alignment(self, ball_start, ball_end, passer_pos, receiver_pos):
        """
        Calculate how well ball trajectory aligns with passer->receiver direction
        
        Returns:
            alignment: 0-1, where 1 = perfect alignment, 0 = perpendicular
        """
        # Ball direction vector
        ball_vec = np.array([ball_end[0] - ball_start[0], ball_end[1] - ball_start[1]])
        ball_norm = np.linalg.norm(ball_vec)
        
        if ball_norm == 0:
            return 0.0
        
        ball_dir = ball_vec / ball_norm
        
        # Player direction vector (passer to receiver)
        player_vec = np.array([receiver_pos[0] - passer_pos[0], 
                               receiver_pos[1] - passer_pos[1]])
        player_norm = np.linalg.norm(player_vec)
        
        if player_norm == 0:
            return 0.0
        
        player_dir = player_vec / player_norm
        
        # Calculate alignment (dot product)
        alignment = np.dot(ball_dir, player_dir)
        
        # Clamp to [0, 1] (negative = ball moving away, set to 0)
        return max(0.0, alignment)
    
    def _is_ball_moving_toward_receiver(self, ball_history, receiver_pos, threshold=0.3):
        """
        Check if ball is moving toward receiver position
        
        Args:
            ball_history: List of ball positions (most recent first)
            receiver_pos: (x, y) position of receiver
            threshold: Minimum alignment threshold
            
        Returns:
            bool: True if ball is moving toward receiver
        """
        # Handle ball_history format (list or deque)
        if isinstance(ball_history, list):
            ball_hist = ball_history
        else:
            ball_hist = list(ball_history)
        
        if len(ball_hist) < 3:
            return False
        
        # Get recent ball positions
        recent_balls = ball_hist[-3:]
        ball_start = (
            recent_balls[0].get('x', recent_balls[0].get('center_x', 0)),
            recent_balls[0].get('y', recent_balls[0].get('center_y', 0))
        )
        ball_end = (
            recent_balls[-1].get('x', recent_balls[-1].get('center_x', 0)),
            recent_balls[-1].get('y', recent_balls[-1].get('center_y', 0))
        )
        
        # Direction from ball start to receiver
        to_receiver = np.array([receiver_pos[0] - ball_start[0],
                                receiver_pos[1] - ball_start[1]])
        to_receiver_norm = np.linalg.norm(to_receiver)
        
        if to_receiver_norm == 0:
            return False
        
        to_receiver_dir = to_receiver / to_receiver_norm
        
        # Ball movement direction
        ball_movement = np.array([ball_end[0] - ball_start[0],
                                 ball_end[1] - ball_start[1]])
        ball_movement_norm = np.linalg.norm(ball_movement)
        
        if ball_movement_norm == 0:
            return False
        
        ball_dir = ball_movement / ball_movement_norm
        
        # Check alignment
        alignment = np.dot(ball_dir, to_receiver_dir)
        return alignment >= threshold
    
    def _calculate_pass_confidence(self, ball_tracking, alignment, passer_dist, receiver_dist, 
                                   pass_distance, ball_history):
        """
        Calculate confidence score for a pass (0-1)
        
        Higher confidence = more likely to be a real pass
        """
        # Ball detection confidence (0-1)
        ball_conf = ball_tracking.get('confidence', 0.5)
        
        # Trajectory alignment (0-1)
        alignment_score = alignment
        
        # Ball speed factor (normalized)
        ball_speed = ball_tracking.get('speed', 0)
        speed_factor = min(1.0, ball_speed / 8.0)  # Normalize to 8 px/frame
        
        # Proximity factors (closer = better)
        passer_proximity = 1.0 - min(1.0, passer_dist / self.max_ball_to_player_dist)
        receiver_proximity = 1.0 - min(1.0, receiver_dist / self.max_ball_to_player_dist)
        proximity_score = (passer_proximity + receiver_proximity) / 2.0
        
        # Distance factor (prefer medium distances)
        if pass_distance < self.min_pass_distance:
            distance_factor = 0.0
        elif pass_distance > self.max_pass_distance:
            distance_factor = 0.0
        else:
            # Optimal around middle of range
            optimal_dist = (self.min_pass_distance + self.max_pass_distance) / 2
            distance_factor = 1.0 - abs(pass_distance - optimal_dist) / optimal_dist
        
        # Ball travel consistency (check if ball moved smoothly)
        travel_consistency = 1.0
        if isinstance(ball_history, list):
            ball_hist = ball_history
        else:
            ball_hist = list(ball_history)
        
        if len(ball_hist) >= 3:
            positions = []
            for b in ball_hist[-3:]:
                if isinstance(b, dict):
                    x = b.get('x', b.get('center_x', 0))
                    y = b.get('y', b.get('center_y', 0))
                    positions.append((x, y))
            
            if len(positions) >= 2:
                distances = []
                for i in range(len(positions) - 1):
                    dist = np.sqrt((positions[i+1][0] - positions[i][0])**2 + 
                                  (positions[i+1][1] - positions[i][1])**2)
                    distances.append(dist)
                
                if len(distances) > 0:
                    # Check if distances are consistent (not too variable)
                    avg_dist = np.mean(distances)
                    if avg_dist > 0:
                        variance = np.std(distances) / avg_dist
                        travel_consistency = max(0.0, 1.0 - variance)
        
        # Weighted combination
        confidence = (
            ball_conf * 0.25 +           # Ball detection quality (25%)
            alignment_score * 0.30 +      # Trajectory alignment (30%)
            speed_factor * 0.15 +         # Ball speed (15%)
            proximity_score * 0.15 +      # Player proximity (15%)
            distance_factor * 0.10 +      # Pass distance (10%)
            travel_consistency * 0.05     # Travel consistency (5%)
        )
        
        return min(1.0, max(0.0, confidence))
    
    def _is_successful_pass(self, ball_tracking, receiver_pos, receiver_dist, ball_history):
        """
        Determine if pass was successful
        
        Success criteria:
        - Ball reached receiver (within threshold)
        - Ball speed is reasonable (not too fast = intercepted)
        - Ball trajectory ends near receiver
        """
        # Receiver must be close to final ball position
        if receiver_dist > 60:
            return False
        
        # Ball speed should be reasonable (not too high = likely intercepted)
        ball_speed = ball_tracking.get('speed', 0)
        if ball_speed > 15:  # Too fast = likely intercepted
            return False
        
        # Check if ball trajectory ends near receiver
        if isinstance(ball_history, list):
            ball_hist = ball_history
        else:
            ball_hist = list(ball_history)
        
        if len(ball_hist) >= 3:
            final_ball = ball_hist[-1]
            final_ball_x = final_ball.get('x', final_ball.get('center_x', 0))
            final_ball_y = final_ball.get('y', final_ball.get('center_y', 0))
            final_dist = np.sqrt((final_ball_x - receiver_pos[0])**2 + 
                                (final_ball_y - receiver_pos[1])**2)
            if final_dist > 70:
                return False
        
        return True
    
    def detect_passes(self, players, ball_tracking, ball_history, frame_num):
        """
        Detect passes between players using ball tracking
        
        Args:
            players: Dict of tracked players {player_id: {x, y, team, ...}}
            ball_tracking: Current ball state {x, y, speed, confidence, ...}
            ball_history: List of recent ball positions (most recent first)
            frame_num: Current frame number
            
        Returns:
            List of pass events, each with:
            {
                'frame': frame_num,
                'passer_id': int,
                'receiver_id': int,
                'team': str,
                'distance': float,
                'type': 'short' or 'long',
                'success': bool,
                'confidence': float,
                'ball_speed': float,
                'alignment': float,
                'ball_travel': float
            }
        """
        passes = []
        
        # Validation checks
        if len(players) < 2:
            return passes
        
        if ball_tracking is None:
            return passes
        
        if len(ball_history) < self.ball_history_size:
            return passes
        
        # Check every 3 frames (reduce computation)
        if frame_num % 3 != 0:
            return passes
        
        # Get ball state
        ball_x = ball_tracking.get('x', 0)
        ball_y = ball_tracking.get('y', 0)
        ball_speed = ball_tracking.get('speed', 0)
        ball_conf = ball_tracking.get('confidence', 0)
        
        if ball_x is None or ball_y is None:
            return passes
        
        # Ball must be moving with reasonable confidence
        if ball_speed < self.min_ball_speed:
            if frame_num % 150 == 0:  # Debug every 150 frames
                print(f"    [PASS DEBUG] Ball speed too low: {ball_speed:.1f} < {self.min_ball_speed}")
            return passes
        
        if ball_conf < 0.2:
            if frame_num % 150 == 0:  # Debug every 150 frames
                print(f"    [PASS DEBUG] Ball confidence too low: {ball_conf:.2f} < 0.2")
            return passes
        
        # Get ball position from history (look back)
        # ball_history is a deque, so we need to access it correctly
        # Convert to list if needed
        if isinstance(ball_history, list):
            ball_history_list = ball_history
        else:
            ball_history_list = list(ball_history)
        
        if len(ball_history_list) < self.ball_history_size:
            return passes
        
        # Get old ball position (from history)
        old_ball_idx = len(ball_history_list) - self.ball_history_size
        old_ball = ball_history_list[old_ball_idx]
        old_ball_x = old_ball.get('x', old_ball.get('center_x', 0))
        old_ball_y = old_ball.get('y', old_ball.get('center_y', 0))
        
        # Calculate ball travel distance
        ball_travel = np.sqrt((ball_x - old_ball_x)**2 + (ball_y - old_ball_y)**2)
        
        if ball_travel < self.min_ball_travel:
            if frame_num % 150 == 0:  # Debug every 150 frames
                print(f"    [PASS DEBUG] Ball travel too short: {ball_travel:.1f} < {self.min_ball_travel}")
            return passes
        
        # Find passer (player closest to old ball position)
        passer_id = None
        passer_dist = float('inf')
        passer_pos = None
        
        for player_id, player in players.items():
            # Try to get player position from history at same time as old ball
            player_x = None
            player_y = None
            use_history = False
            
            # Handle player history (could be deque or list)
            if 'history' in player and player['history']:
                try:
                    # Convert deque to list if needed (deques don't support slicing)
                    if hasattr(player['history'], '__iter__') and not isinstance(player['history'], (list, tuple)):
                        player_history_list = list(player['history'])
                    else:
                        player_history_list = player['history']
                    
                    if len(player_history_list) >= self.ball_history_size:
                        # Find position in history at same relative time
                        history_idx = len(player_history_list) - self.ball_history_size
                        if history_idx >= 0 and history_idx < len(player_history_list):
                            old_player_pos = player_history_list[history_idx]
                            if isinstance(old_player_pos, dict):
                                player_x = old_player_pos.get('x', old_player_pos.get('center_x'))
                                player_y = old_player_pos.get('y', old_player_pos.get('center_y'))
                                if player_x is not None and player_y is not None:
                                    use_history = True
                except (IndexError, KeyError, TypeError, AttributeError) as e:
                    # Silently continue - will use current position as fallback
                    pass
            
            # Fallback: use current position
            if player_x is None or player_y is None:
                player_x = player.get('x', player.get('center_x', 0))
                player_y = player.get('y', player.get('center_y', 0))
            
            if player_x is not None and player_y is not None:
                dist = np.sqrt((player_x - old_ball_x)**2 + (player_y - old_ball_y)**2)
                if dist < passer_dist and dist < self.max_ball_to_player_dist:
                    passer_dist = dist
                    passer_id = player_id
                    passer_pos = (player_x, player_y)
        
        # Find receiver (player closest to current ball position)
        receiver_id = None
        receiver_dist = float('inf')
        receiver_pos = None
        
        for player_id, player in players.items():
            player_x = player.get('x', player.get('center_x', 0))
            player_y = player.get('y', player.get('center_y', 0))
            
            if player_x is not None and player_y is not None:
                dist = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
                if dist < receiver_dist and dist < self.max_ball_to_player_dist:
                    receiver_dist = dist
                    receiver_id = player_id
                    receiver_pos = (player_x, player_y)
        
        # Validate passer and receiver
        if passer_id is None:
            if frame_num % 150 == 0:
                print(f"    [PASS DEBUG] No passer found (ball pos: {old_ball_x:.0f}, {old_ball_y:.0f})")
            return passes
        
        if receiver_id is None:
            if frame_num % 150 == 0:
                print(f"    [PASS DEBUG] No receiver found (ball pos: {ball_x:.0f}, {ball_y:.0f})")
            return passes
        
        if passer_id == receiver_id:
            if frame_num % 150 == 0:
                print(f"    [PASS DEBUG] Passer and receiver are same player (P{passer_id})")
            return passes
        
        # CRITICAL: Only passes between same-team players
        passer_team = players[passer_id].get('team')
        receiver_team = players[receiver_id].get('team')
        
        if passer_team is None or receiver_team is None:
            if frame_num % 150 == 0:
                print(f"    [PASS DEBUG] Missing team info (passer_team={passer_team}, receiver_team={receiver_team})")
            return passes
        
        if passer_team != receiver_team:
            if frame_num % 150 == 0:
                print(f"    [PASS DEBUG] Different teams (P{passer_id}={passer_team}, P{receiver_id}={receiver_team})")
            return passes  # Different teams - not a pass
        
        # Check cooldown
        pair_key = tuple(sorted([passer_id, receiver_id]))
        if pair_key in self.last_pass_frame:
            if frame_num - self.last_pass_frame[pair_key] < self.pass_cooldown:
                return passes
        
        # Calculate pass distance (player-to-player)
        pass_distance = np.sqrt((receiver_pos[0] - passer_pos[0])**2 + 
                               (receiver_pos[1] - passer_pos[1])**2)
        
        # Distance validation
        if pass_distance < self.min_pass_distance or pass_distance > self.max_pass_distance:
            if frame_num % 150 == 0:
                print(f"    [PASS DEBUG] Pass distance out of range: {pass_distance:.0f} (need {self.min_pass_distance}-{self.max_pass_distance})")
            return passes
        
        # Calculate trajectory alignment
        alignment = self._calculate_trajectory_alignment(
            (old_ball_x, old_ball_y),
            (ball_x, ball_y),
            passer_pos,
            receiver_pos
        )
        
        if alignment < self.min_alignment:
            if frame_num % 150 == 0:
                print(f"    [PASS DEBUG] Alignment too low: {alignment:.2f} < {self.min_alignment}")
            return passes
        
        # Check if ball is moving toward receiver (relaxed check)
        # Use ball_history_list we created earlier
        if not self._is_ball_moving_toward_receiver(ball_history_list, receiver_pos, threshold=0.25):
            # Still allow if alignment is very good (above 0.6)
            if alignment < 0.6:
                return passes
        
        # Calculate pass confidence
        confidence = self._calculate_pass_confidence(
            ball_tracking, alignment, passer_dist, receiver_dist,
            pass_distance, ball_history_list
        )
        
        # Lower confidence threshold slightly if alignment is very good
        min_conf = self.min_confidence
        if alignment > 0.7:
            min_conf = self.min_confidence * 0.8  # Lower threshold by 20%
        
        if confidence < min_conf:
            if frame_num % 150 == 0:
                print(f"    [PASS DEBUG] Confidence too low: {confidence:.2f} < {min_conf:.2f}")
            return passes
        
        # Determine pass type
        pass_type = 'long' if pass_distance > self.short_pass_threshold else 'short'
        
        # Determine success
        success = self._is_successful_pass(ball_tracking, receiver_pos, receiver_dist, ball_history_list)
        
        # Create pass event
        pass_event = {
            'frame': frame_num,
            'passer_id': passer_id,
            'receiver_id': receiver_id,
            'team': passer_team,
            'distance': float(pass_distance),
            'type': pass_type,
            'success': success,
            'confidence': float(confidence),
            'ball_speed': float(ball_speed),
            'alignment': float(alignment),
            'ball_travel': float(ball_travel)
        }
        
        # Record pass
        passes.append(pass_event)
        self.last_pass_frame[pair_key] = frame_num
        self.pass_history.append(pass_event)
        
        return passes

