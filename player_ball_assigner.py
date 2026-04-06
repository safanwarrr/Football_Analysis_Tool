import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 40
        
    def assign_ball_to_player(self, players, ball_bbox):
        """Assign ball to the closest player."""
        ball_position = self.get_center_of_bbox(ball_bbox)
        
        minimum_distance = float('inf')
        assigned_player = -1
        
        for player_id, player in players.items():
            player_bbox = player['bbox']
            
            # Get distance from left foot, right foot, and center
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance_center = measure_distance(self.get_center_of_bbox(player_bbox), ball_position)
            
            distance = min(distance_left, distance_right, distance_center)
            
            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id
        
        return assigned_player
    
    @staticmethod
    def get_center_of_bbox(bbox):
        """Get center of bounding box."""
        return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
