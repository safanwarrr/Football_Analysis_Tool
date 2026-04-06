#!/usr/bin/env python3
"""
Football Analysis System
Detects and tracks players, referees, and footballs using YOLO.
Assigns players to teams based on t-shirt colors using K-means clustering.
Measures ball acquisition, camera movement, and player speed/distance.
"""

from utils.video_utils import read_video, save_video
from utils.bbox_utils import get_center_of_bbox, get_foot_position
from trackers.tracker import Tracker
from team_assignment.team_assigner import TeamAssigner
from team_assignment.player_ball_assigner import PlayerBallAssigner
from camera_movement.camera_movement_estimator import CameraMovementEstimator
from perspective.view_transformer import ViewTransformer
from perspective.speed_and_distance_estimator import SpeedAndDistanceEstimator
from scoring_probability.probability_calculator import ScoringProbabilityCalculator
import numpy as np
import cv2

def main():
    # Read video
    video_path = '/Users/safanwar/Documents/Untitled.mp4'
    video_frames = read_video(video_path)
    
    # Specify the player with the ball (set to None for automatic detection)
    PLAYER_WITH_BALL = 2  # Set this to the player ID who has the ball, or None for auto-detection
    
    # Initialize tracker
    tracker = Tracker('models/yolov8x.pt')
    
    # Get object tracks
    tracks = tracker.get_object_tracks(video_frames, 
                                        read_from_stub=True, 
                                        stub_path='stubs/track_stubs.pkl')
    
    # Get camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    
    # Add positions to tracks
    for object_name, object_tracks in tracks.items():
        for frame_num, track in enumerate(object_tracks):
            for track_id, track_info in track.items():
                bbox = track_info['bbox']
                if object_name == 'ball':
                    position = get_center_of_bbox(bbox)
                else:
                    position = get_foot_position(bbox)
                tracks[object_name][frame_num][track_id]['position'] = position
    
    # Adjust positions for camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # Apply perspective transformation
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate ball positions
    tracks["ball"] = interpolate_ball_positions(tracks["ball"])
    
    # Calculate speed and distance
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    # Assign teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                   track['bbox'],
                                                   player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # Assign ball possession
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    players_with_ball = []  # Track which player has ball in each frame
    
    for frame_num, player_track in enumerate(tracks['players']):
        # Get ball bbox - use any track_id since ball might have different IDs
        ball_dict = tracks['ball'][frame_num]
        ball_bbox = []
        if ball_dict:
            # Get the first (and likely only) ball detection
            ball_bbox = list(ball_dict.values())[0].get('bbox', [])
        
        if len(ball_bbox) == 0:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            players_with_ball.append(None)
            continue
        
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            players_with_ball.append(assigned_player)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            players_with_ball.append(None)
    
    team_ball_control = np.array(team_ball_control)
    
    # Draw annotations
    # If manual player is specified, use that; otherwise use automatic detection
    if PLAYER_WITH_BALL is not None:
        players_with_ball_final = [PLAYER_WITH_BALL] * len(video_frames)
    else:
        players_with_ball_final = players_with_ball
    
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control, players_with_ball=players_with_ball_final)
    
    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    # Calculate and draw scoring probability for player 2
    probability_calculator = ScoringProbabilityCalculator()
    for frame_num, frame in enumerate(output_video_frames):
        # Get player 2's position, team, and speed
        if PLAYER_WITH_BALL in tracks['players'][frame_num]:
            player_info = tracks['players'][frame_num][PLAYER_WITH_BALL]
            player_position = player_info.get('position')
            player_team = player_info.get('team', 1)
            player_speed = player_info.get('speed', 0)
            
            # Count nearby opponents (within 150 pixels)
            nearby_opponents = 0
            if player_position is not None:
                for other_id, other_player in tracks['players'][frame_num].items():
                    if other_id != PLAYER_WITH_BALL:
                        other_team = other_player.get('team', 0)
                        other_position = other_player.get('position')
                        
                        # Check if opposing team and close by
                        if other_team != player_team and other_position is not None:
                            distance = ((player_position[0] - other_position[0])**2 + 
                                      (player_position[1] - other_position[1])**2)**0.5
                            if distance < 150:  # Within 150 pixels = nearby
                                nearby_opponents += 1
                
                probability = probability_calculator.calculate_probability(
                    player_position, player_team, player_speed, nearby_opponents)
                output_video_frames[frame_num] = probability_calculator.draw_probability(frame, probability, player_team)
    
    # Save video
    save_video(output_video_frames, 'output/output_video.mp4')
    
    print("✓ Video processing complete!")
    print(f"✓ Output saved to: output/output_video.mp4")
    print(f"✓ Processed {len(video_frames)} frames")

def interpolate_ball_positions(ball_positions):
    """Interpolate missing ball positions."""
    # Extract ball bboxes, handling any track_id
    ball_bboxes = []
    for frame_ball in ball_positions:
        if frame_ball:
            # Get the first ball detection (there should only be one)
            bbox = list(frame_ball.values())[0].get('bbox', [])
            ball_bboxes.append(bbox if bbox else [None, None, None, None])
        else:
            ball_bboxes.append([None, None, None, None])
    
    df_ball_positions = pd.DataFrame(ball_bboxes, columns=['x1', 'y1', 'x2', 'y2'])
    
    # Interpolate missing values
    df_ball_positions = df_ball_positions.interpolate()
    df_ball_positions = df_ball_positions.bfill()
    
    # Reconstruct ball_positions with track_id 1 for consistency
    ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
    
    return ball_positions

if __name__ == '__main__':
    import pandas as pd
    import os
    
    # Create necessary directories
    os.makedirs('stubs', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    main()
