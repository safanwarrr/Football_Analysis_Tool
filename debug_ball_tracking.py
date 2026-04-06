#!/usr/bin/env python3
"""Debug ball tracking with visual output."""

import pickle
import cv2
from utils.video_utils import read_video
from team_assignment.player_ball_assigner import PlayerBallAssigner

# Load tracks
with open('stubs/track_stubs.pkl', 'rb') as f:
    tracks = pickle.load(f)

# Load video
video_frames = read_video('/Users/safanwar/Documents/Untitled.mp4')

# Check some frames
player_ball_assigner = PlayerBallAssigner()

for frame_num in [0, 10, 20, 30, 50]:
    print(f"\n=== Frame {frame_num} ===")
    
    # Get ball
    ball_dict = tracks['ball'][frame_num]
    if ball_dict:
        ball_bbox = list(ball_dict.values())[0].get('bbox', [])
        ball_center = ((ball_bbox[0] + ball_bbox[2]) // 2, (ball_bbox[1] + ball_bbox[3]) // 2)
        print(f"Ball position: {ball_center}")
        print(f"Ball bbox: {ball_bbox}")
    else:
        print("No ball detected")
        continue
    
    # Get players
    player_track = tracks['players'][frame_num]
    print(f"Players in frame: {list(player_track.keys())}")
    
    # Calculate distances
    print("\nPlayer distances from ball:")
    for player_id, player in player_track.items():
        player_bbox = player['bbox']
        player_center = ((player_bbox[0] + player_bbox[2]) // 2, (player_bbox[1] + player_bbox[3]) // 2)
        
        # Calculate distance from ball to player center
        distance = ((ball_center[0] - player_center[0])**2 + (ball_center[1] - player_center[1])**2)**0.5
        print(f"  Player {player_id}: center={player_center}, distance={distance:.1f}px")
    
    # Who gets assigned?
    assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
    print(f"\nAssigned to player: {assigned_player}")
    
    # Save annotated frame
    frame = video_frames[frame_num].copy()
    
    # Draw ball
    cv2.circle(frame, ball_center, 10, (0, 255, 0), -1)
    cv2.circle(frame, ball_center, 10, (0, 0, 0), 2)
    
    # Draw players with distances
    for player_id, player in player_track.items():
        player_bbox = player['bbox']
        player_center = ((player_bbox[0] + player_bbox[2]) // 2, (player_bbox[1] + player_bbox[3]) // 2)
        
        color = (0, 0, 255) if player_id == assigned_player else (255, 0, 0)
        cv2.rectangle(frame, (int(player_bbox[0]), int(player_bbox[1])), 
                     (int(player_bbox[2]), int(player_bbox[3])), color, 2)
        cv2.putText(frame, str(player_id), player_center, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(f'debug_frame_{frame_num}.jpg', frame)
    print(f"Saved debug_frame_{frame_num}.jpg")
