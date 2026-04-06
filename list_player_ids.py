#!/usr/bin/env python3
"""
Helper script to list all player track IDs in the video.
Use this to identify which player scores the goal.
"""

import pickle
import os

def list_player_ids():
    stub_path = 'stubs/track_stubs.pkl'
    
    if not os.path.exists(stub_path):
        print("Error: track_stubs.pkl not found. Please run main.py first.")
        return
    
    with open(stub_path, 'rb') as f:
        tracks = pickle.load(f)
    
    # Collect all unique player IDs
    all_player_ids = set()
    for frame_players in tracks['players']:
        all_player_ids.update(frame_players.keys())
    
    print(f"Total unique player IDs: {len(all_player_ids)}")
    print(f"Player IDs: {sorted(all_player_ids)}")
    print("\nTo mark a player as the goal scorer:")
    print("1. Identify the player's ID from the list above")
    print("2. Edit main.py and set GOAL_SCORER_ID = <player_id>")
    print("3. Run main.py again")

if __name__ == '__main__':
    list_player_ids()
