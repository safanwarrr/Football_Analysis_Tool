#!/usr/bin/env python3
"""Debug ball assignment."""

import pickle
from team_assignment.player_ball_assigner import PlayerBallAssigner

with open('stubs/track_stubs.pkl', 'rb') as f:
    tracks = pickle.load(f)

player_ball_assigner = PlayerBallAssigner()

# Check frames 10-20
print("Ball assignment for frames 10-20:\n")
for frame_num in range(10, min(21, len(tracks['players']))):
    # Get ball bbox
    ball_dict = tracks['ball'][frame_num]
    ball_bbox = []
    if ball_dict:
        ball_bbox = list(ball_dict.values())[0].get('bbox', [])
    
    if len(ball_bbox) > 0:
        player_track = tracks['players'][frame_num]
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        print(f"Frame {frame_num}:")
        print(f"  Ball bbox: {ball_bbox}")
        print(f"  Ball detected: Yes")
        print(f"  Assigned to player: {assigned_player}")
        print(f"  Players in frame: {list(player_track.keys())}")
    else:
        print(f"Frame {frame_num}: No ball detected")
    print()
