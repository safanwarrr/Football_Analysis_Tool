#!/usr/bin/env python3
"""Check if speed values are being calculated correctly."""

import pickle

with open('stubs/track_stubs.pkl', 'rb') as f:
    tracks = pickle.load(f)

# Check first 10 frames for player speeds
print("Checking speed values in first 10 frames:\n")
for frame_num in range(min(10, len(tracks['players']))):
    print(f"Frame {frame_num}:")
    for player_id, player_info in tracks['players'][frame_num].items():
        speed = player_info.get('speed', 'NOT SET')
        distance = player_info.get('distance', 'NOT SET')
        print(f"  Player {player_id}: speed={speed}, distance={distance}")
    print()
