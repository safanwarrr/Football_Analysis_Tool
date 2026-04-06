#!/usr/bin/env python3
"""Debug position_transformed values."""

from utils.video_utils import read_video
from utils.bbox_utils import get_foot_position
from trackers.tracker import Tracker
from camera_movement.camera_movement_estimator import CameraMovementEstimator
from perspective.view_transformer import ViewTransformer
from perspective.speed_and_distance_estimator import SpeedAndDistanceEstimator

# Read video
video_path = '/Users/safanwar/Documents/Untitled.mp4'
video_frames = read_video(video_path)

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
            position = get_foot_position(bbox)
            tracks[object_name][frame_num][track_id]['position'] = position

# Adjust positions for camera movement
camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

# Apply perspective transformation
view_transformer = ViewTransformer()
view_transformer.add_transformed_position_to_tracks(tracks)

# Check frame 10
print("Frame 10 - Position data:")
for player_id, player_info in list(tracks['players'][10].items())[:5]:
    position = player_info.get('position')
    position_adjusted = player_info.get('position_adjusted')
    position_transformed = player_info.get('position_transformed')
    print(f"Player {player_id}:")
    print(f"  position: {position}")
    print(f"  position_adjusted: {position_adjusted}")
    print(f"  position_transformed: {position_transformed}")

# Calculate speed
print("\n\nCalculating speeds...")
speed_estimator = SpeedAndDistanceEstimator()
speed_estimator.add_speed_and_distance_to_tracks(tracks)

print("\n\nFrame 10 - Speed data:")
for player_id, player_info in list(tracks['players'][10].items())[:5]:
    speed = player_info.get('speed', 'NOT SET')
    distance = player_info.get('distance', 'NOT SET')
    print(f"Player {player_id}: speed={speed}, distance={distance}")
