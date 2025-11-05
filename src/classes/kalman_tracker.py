"""
Robust Multi-Object Tracker using Kalman Filter
Similar to SORT/ByteTrack but optimized for football players
"""

import numpy as np
from collections import deque
import cv2


class KalmanTracker:
    """Kalman filter-based tracker for maintaining consistent player IDs"""
    
    def __init__(self, max_age=30, min_hits=2, iou_threshold=0.3,
                 min_bbox_area=500, max_bbox_area=80000,
                 min_track_length=15, max_velocity_jump=100):
        """
        Args:
            max_age: Maximum frames to keep a track alive without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IoU threshold for matching detections to tracks
            min_bbox_area: Minimum bounding box area for valid player (pixels²)
            max_bbox_area: Maximum bounding box area for valid player (pixels²)
            min_track_length: Minimum track length before considering it valid
            max_velocity_jump: Maximum velocity change between frames (pixels/frame)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.min_bbox_area = min_bbox_area
        self.max_bbox_area = max_bbox_area
        self.min_track_length = min_track_length
        self.max_velocity_jump = max_velocity_jump
        
        self.tracks = {}  # track_id -> track_info
        self.next_id = 1
        self.frame_count = 0
        self.confirmed_tracks = set()  # Tracks that have passed minimum lifetime
        self.min_confirmation_frames = 8  # STRICT: Minimum 8 frames before confirming (prevents ghosts)
        
    def _iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def _create_kalman_filter(self):
        """Create Kalman filter for tracking"""
        kf = cv2.KalmanFilter(8, 4)  # 8 state vars, 4 measurements
        
        # State: [x, y, vx, vy, w, h, vw, vh]
        # Measurement: [x, y, w, h]
        
        # State transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 1, 0, 0, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 0, 0],  # vx = vx
            [0, 0, 0, 1, 0, 0, 0, 0],  # vy = vy
            [0, 0, 0, 0, 1, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 0, 0, 1, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1]   # vh = vh
        ], dtype=np.float32)
        
        # Measurement matrix (observe position and size)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise (how much we trust the model)
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # Measurement noise (how much we trust the measurements)
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        
        # Error covariance
        kf.errorCovPost = np.eye(8, dtype=np.float32)
        
        return kf
    
    def _predict(self, track_id):
        """Predict next state using Kalman filter"""
        track = self.tracks[track_id]
        kf = track['kalman']
        
        # Predict
        prediction = kf.predict()
        
        # Extract predicted bounding box
        x = prediction[0, 0]
        y = prediction[1, 0]
        w = prediction[4, 0]
        h = prediction[5, 0]
        
        return (x, y, w, h)
    
    def _update(self, track_id, detection):
        """Update track with new detection"""
        track = self.tracks[track_id]
        kf = track['kalman']
        
        # Measurement: [x, y, w, h]
        x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
        measurement = np.array([[x], [y], [w], [h]], dtype=np.float32)
        
        # Update Kalman filter
        kf.correct(measurement)
        
        # Get updated state
        state = kf.statePost
        
        # Update track info
        track['hits'] += 1
        track['age'] = 0
        track['time_since_update'] = 0
        track['last_detection'] = detection
        track['history'].append({
            'frame': self.frame_count,
            'x': float(state[0, 0]),
            'y': float(state[1, 0]),
            'width': float(state[4, 0]),
            'height': float(state[5, 0]),
            'vx': float(state[2, 0]),
            'vy': float(state[3, 0]),
            'detected': True
        })
        
        # Update team and jersey color if available
        if 'team' in detection:
            track['team'] = detection['team']
        if 'jersey_color' in detection:
            track['jersey_color'] = detection['jersey_color']
    
    def _interpolate(self, track_id, current_frame):
        """Interpolate position for missing detection"""
        track = self.tracks[track_id]
        history = track['history']
        
        if len(history) < 2:
            return None
        
        # Predict using Kalman filter
        predicted_box = self._predict(track_id)
        x, y, w, h = predicted_box
        
        # Get last known velocity
        last_state = track['kalman'].statePost
        vx = last_state[2, 0]
        vy = last_state[3, 0]
        
        # Interpolated position
        interpolated = {
            'frame': current_frame,
            'x': float(x),
            'y': float(y),
            'width': float(w),
            'height': float(h),
            'vx': float(vx),
            'vy': float(vy),
            'detected': False  # Mark as interpolated
        }
        
        track['history'].append(interpolated)
        track['time_since_update'] += 1
        track['age'] += 1
        
        return interpolated
    
    def update(self, detections, frame_num):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections, each with keys: x, y, width, height, confidence, etc.
            frame_num: Current frame number
            
        Returns:
            Dictionary of tracked objects: {track_id: {x, y, width, height, team, confidence, ...}}
        """
        self.frame_count = frame_num
        
        # Convert detections to boxes for matching
        detection_boxes = []
        for det in detections:
            x = det['x'] - det['width'] / 2  # Convert center to top-left
            y = det['y'] - det['height'] / 2
            detection_boxes.append((x, y, det['width'], det['height']))
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        if len(self.tracks) > 0 and len(detections) > 0:
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            track_ids = list(self.tracks.keys())
            
            for i, track_id in enumerate(track_ids):
                # Predict current position
                predicted_box = self._predict(track_id)
                track_box = (predicted_box[0] - predicted_box[2]/2, 
                            predicted_box[1] - predicted_box[3]/2,
                            predicted_box[2], predicted_box[3])
                
                for j, det_box in enumerate(detection_boxes):
                    iou_matrix[i, j] = self._iou(track_box, det_box)
            
            # Greedy matching (highest IoU first)
            while True:
                if iou_matrix.size == 0:
                    break
                
                # Find max IoU
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]
                
                if max_iou < self.iou_threshold:
                    break
                
                track_idx, det_idx = max_idx
                track_id = track_ids[track_idx]
                
                # Match
                matched_tracks.add(track_id)
                matched_detections.add(det_idx)
                
                # Update track
                self._update(track_id, detections[det_idx])
                
                # Check if track should be confirmed (STRICT: must be old enough AND have enough hits)
                track = self.tracks[track_id]
                track_age = frame_num - track.get('created_frame', frame_num)
                if track_age >= self.min_confirmation_frames and track['hits'] >= self.min_hits and len(track['history']) >= 5:
                    self.confirmed_tracks.add(track_id)
                
                # Remove from matrix
                iou_matrix[track_idx, :] = -1
                iou_matrix[:, det_idx] = -1
        
        # Update unmatched tracks (interpolate)
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self._interpolate(track_id, frame_num)
        
        # Create new tracks for unmatched detections (with validation)
        # FIRST: Check if new detection might be duplicate of existing unconfirmed track
        for i, det in enumerate(detections):
            if i not in matched_detections:
                # VALIDATION: Filter out ghost detections
                bbox_area = det['width'] * det['height']
                
                # Check bounding box size (must be realistic player size)
                if bbox_area < self.min_bbox_area or bbox_area > self.max_bbox_area:
                    continue  # Skip - invalid size
                
                # Check confidence (if available) - RELAXED for debugging
                if det.get('confidence', 0.5) < 0.15:
                    continue  # Skip - very low confidence detection only
                
                # STRICT: Check if this detection overlaps with existing unconfirmed tracks (merge duplicates)
                det_box = (det['x'] - det['width']/2, det['y'] - det['height']/2, 
                          det['width'], det['height'])
                merged = False
                
                for track_id, track in self.tracks.items():
                    if track_id in self.confirmed_tracks:
                        continue  # Skip confirmed tracks
                    
                    # Check IoU with this track
                    last_state = track['kalman'].statePost
                    track_box = (last_state[0, 0] - last_state[4, 0]/2,
                                last_state[1, 0] - last_state[5, 0]/2,
                                last_state[4, 0], last_state[5, 0])
                    
                    iou = self._iou(det_box, track_box)
                    # STRICT: Higher IoU threshold (0.6) to merge duplicates
                    if iou > 0.6:  # Very high overlap = duplicate
                        # Merge: update existing track instead of creating new one
                        self._update(track_id, det)
                        matched_detections.add(i)  # Mark as matched
                        merged = True
                        break
                
                if merged:
                    continue  # Already merged, don't create new track
                
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                
                # Create new track
                kf = self._create_kalman_filter()
                
                # Initialize Kalman filter
                x, y = det['x'], det['y']
                w, h = det['width'], det['height']
                kf.statePre = np.array([[x], [y], [0], [0], [w], [h], [0], [0]], dtype=np.float32)
                kf.statePost = np.array([[x], [y], [0], [0], [w], [h], [0], [0]], dtype=np.float32)
                
                self.tracks[track_id] = {
                    'kalman': kf,
                    'hits': 1,
                    'age': 0,
                    'time_since_update': 0,
                    'team': det.get('team', None),
                    'jersey_color': det.get('jersey_color', None),
                    'history': deque([{
                        'frame': frame_num,
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'vx': 0,
                        'vy': 0,
                        'detected': True
                    }], maxlen=100),
                    'last_detection': det,
                    'created_frame': frame_num,
                    'consecutive_hits': 1  # Track consecutive hits
                }
        
        # Remove old tracks and ghost tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            # Remove if too old
            if track['time_since_update'] > self.max_age:
                tracks_to_remove.append(track_id)
                continue
            
            # Remove ghost tracks (RELAXED for debugging - less aggressive)
            track_length = len(track['history'])
            if track_length > 0:
                track_age = frame_num - track.get('created_frame', frame_num - track_length)
                # Only remove if track is very old AND has very few hits
                if track_age > self.min_track_length * 2 and track['hits'] < 2:
                    # Track has been around a long time but very few detections = ghost
                    tracks_to_remove.append(track_id)
                    continue
                
                # Check for unrealistic movement (teleporting = ghost) - RELAXED
                if track_length >= 2:
                    try:
                        history_list = list(track['history'])
                        recent_pos = history_list[-1]
                        prev_pos = history_list[-2]
                        
                        if recent_pos.get('detected', True) and prev_pos.get('detected', True):
                            dx = recent_pos['x'] - prev_pos['x']
                            dy = recent_pos['y'] - prev_pos['y']
                            distance = np.sqrt(dx**2 + dy**2)
                            
                            # Only remove if moved impossibly far (relaxed threshold)
                            if distance > self.max_velocity_jump * 3:  # Very relaxed
                                tracks_to_remove.append(track_id)
                                continue
                    except (KeyError, IndexError, TypeError):
                        pass  # Skip movement check if data is invalid
                
                # Check bounding box size consistency - RELAXED
                if track_length >= 5:  # Need more samples
                    try:
                        sizes = [h['width'] * h['height'] for h in list(track['history'])[-5:]]
                        if len(sizes) >= 3:
                            size_variance = np.std(sizes) / (np.mean(sizes) + 1e-6)
                            # Only remove if size varies extremely (relaxed threshold)
                            if size_variance > 0.8:  # Very relaxed
                                tracks_to_remove.append(track_id)
                                continue
                    except (KeyError, IndexError, TypeError):
                        pass  # Skip size check if data is invalid
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Build output (STRICT: only confirmed tracks to prevent ghost IDs)
        tracked_objects = {}
        for track_id, track in self.tracks.items():
            # STRICT: Only output confirmed tracks (passed minimum lifetime)
            if track_id not in self.confirmed_tracks:
                continue  # Skip unconfirmed tracks - prevents ghost IDs
            
            # Additional validation: must have enough hits and history
            if track['hits'] < self.min_hits or len(track['history']) < 3:
                continue
            
            last_state = track['kalman'].statePost
            
            # Final validation: check current bbox size
            current_area = float(last_state[4, 0]) * float(last_state[5, 0])
            if current_area < self.min_bbox_area or current_area > self.max_bbox_area:
                continue  # Skip - invalid size
            
            tracked_objects[track_id] = {
                'id': track_id,
                'x': float(last_state[0, 0]),
                'y': float(last_state[1, 0]),
                'width': float(last_state[4, 0]),
                'height': float(last_state[5, 0]),
                'vx': float(last_state[2, 0]),
                'vy': float(last_state[3, 0]),
                'speed': float(np.sqrt(last_state[2, 0]**2 + last_state[3, 0]**2)),
                'team': track['team'],
                'jersey_color': track['jersey_color'],
                'confidence': track['last_detection'].get('confidence', 0.5),
                'hits': track['hits'],
                'age': track['age'],
                'time_since_update': track['time_since_update'],
                'history': list(track['history'])[-30:],  # Last 30 frames
                'is_interpolated': track['time_since_update'] > 0
            }
        
        return tracked_objects

