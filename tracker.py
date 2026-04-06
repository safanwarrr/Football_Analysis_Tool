import torch
# Monkey-patch torch.load to use weights_only=False for trusted ultralytics models
import functools
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(f, *args, **kwargs):
    # Set weights_only=False for ultralytics model files
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2

class Tracker:
    def __init__(self, model_path):
        """Initialize YOLO model and tracker."""
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames):
        """Run detection on all frames."""
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            detections_batch = self.model.predict(batch, conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get object tracks across all frames.
        
        Returns dictionary with keys: 'players', 'referees', 'ball'
        Each contains frame-wise tracking information.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        detections = self.detect_frames(frames)
        
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv['person']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                # Ball detection (sports ball in COCO)
                if cls_id == cls_names_inv.get('sports ball'):
                    # Only keep the first ball detected per frame
                    if not tracks["ball"][frame_num]:
                        tracks["ball"][frame_num][track_id] = {"bbox": bbox}
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, speed=None, distance=None):
        """Draw ellipse at the bottom of bounding box with speed/distance info."""
        y2 = int(bbox[3])
        x_center, _ = self.get_center_of_bbox(bbox)
        width = self.get_width(bbox)
        
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        if speed is not None and distance is not None:
            # Display speed and distance in a neat white box
            speed_text = f"{speed:.1f} km/h"
            distance_text = f"{distance:.1f}m"
            
            rectangle_width = 85
            rectangle_height = 35
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = y2 + 10
            y2_rect = y1_rect + rectangle_height
            
            # Draw white filled background
            cv2.rectangle(frame,
                         (int(x1_rect), int(y1_rect)),
                         (int(x2_rect), int(y2_rect)),
                         (255, 255, 255),
                         cv2.FILLED)
            
            # Draw border
            cv2.rectangle(frame,
                         (int(x1_rect), int(y1_rect)),
                         (int(x2_rect), int(y2_rect)),
                         (0, 0, 0),
                         2)
            
            # Draw speed text (top line)
            cv2.putText(
                frame,
                speed_text,
                (int(x1_rect + 5), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1
            )
            
            # Draw distance text (bottom line)
            cv2.putText(
                frame,
                distance_text,
                (int(x1_rect + 5), int(y1_rect + 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1
            )
        
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        """Draw triangle above player (indicates ball possession)."""
        y = int(bbox[1])
        x, _ = self.get_center_of_bbox(bbox)
        
        # Larger, more visible triangle
        triangle_points = np.array([
            [x, y - 10],
            [x - 20, y - 40],
            [x + 20, y - 40],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 3)
        
        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """Draw team ball possession statistics."""
        # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1400, 870), (1850, 970), (255, 255, 255), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (1400, 870), (1850, 970), (0, 0, 0), 2)
        
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Get number of frames each team had control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames > 0:
            team_1_pct = team_1_num_frames / total_frames
            team_2_pct = team_2_num_frames / total_frames
        else:
            team_1_pct = 0
            team_2_pct = 0
        
        cv2.putText(frame, f"Team 1: {team_1_pct * 100:.1f}%", (1420, 910), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2: {team_2_pct * 100:.1f}%", (1420, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    def draw_annotations(self, video_frames, tracks, team_ball_control, players_with_ball=None):
        """Draw all annotations on video frames."""
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            
            # Determine which player has the ball in this frame
            player_with_ball_id = None
            if players_with_ball is not None and frame_num < len(players_with_ball):
                player_with_ball_id = players_with_ball[frame_num]
            
            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                speed = player.get("speed", 0)
                distance = player.get("distance", 0)
                frame = self.draw_ellipse(frame, player["bbox"], color, speed=speed, distance=distance)
                
                # Draw green triangle on player with ball
                if player_with_ball_id is not None and track_id == player_with_ball_id:
                    frame = self.draw_triangle(frame, player["bbox"], (0, 255, 0))
            
            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), speed=None, distance=None)
            
            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_video_frames.append(frame)
        
        return output_video_frames
    
    @staticmethod
    def get_center_of_bbox(bbox):
        """Get center of bounding box."""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    @staticmethod
    def get_width(bbox):
        """Get width of bounding box."""
        return bbox[2] - bbox[0]
