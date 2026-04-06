import numpy as np
import cv2

class ScoringProbabilityCalculator:
    def __init__(self):
        # Define pitch dimensions (approximate)
        self.pitch_length = 1920  # pixels
        self.pitch_width = 1080   # pixels
        
        # Goal locations (assuming goals at top and bottom)
        self.top_goal_center = (self.pitch_length // 2, 50)
        self.bottom_goal_center = (self.pitch_length // 2, self.pitch_width - 50)
    
    def calculate_probability(self, player_position, team, player_speed=0, nearby_opponents=0):
        """
        Calculate scoring probability based on player position, speed, and pressure.
        
        Args:
            player_position: (x, y) tuple of player position
            team: 1 or 2 indicating which team
            player_speed: player's current speed in km/h
            nearby_opponents: number of opposing players within close range
        
        Returns:
            probability: float between 0 and 100
        """
        if player_position is None:
            return 0.0
        
        x, y = player_position
        
        # Determine which goal to aim for based on team
        # Team 1 aims for top goal, Team 2 aims for bottom goal
        if team == 1:
            target_goal = self.top_goal_center
        else:
            target_goal = self.bottom_goal_center
        
        # Calculate distance to goal
        distance = np.sqrt((x - target_goal[0])**2 + (y - target_goal[1])**2)
        
        # Calculate angle to goal (central positions have better angles)
        angle_factor = 1.0 - abs(x - target_goal[0]) / (self.pitch_length / 2)
        angle_factor = max(0.3, angle_factor)  # Minimum 30% angle factor
        
        # Normalize distance (closer = higher probability)
        # Max distance is diagonal of pitch
        max_distance = np.sqrt(self.pitch_length**2 + self.pitch_width**2)
        distance_factor = 1.0 - (distance / max_distance)
        
        # Base probability calculation
        # Close to goal and central = high probability
        base_probability = distance_factor * angle_factor * 100
        
        # Add positional zones
        # Very close to goal (penalty box area) = boost
        if distance < 200:
            base_probability = min(95, base_probability * 1.5)
        elif distance < 400:
            base_probability = min(80, base_probability * 1.3)
        
        # Speed/momentum factor
        # Players moving quickly towards goal have higher probability
        # Hesitating or standing still reduces probability
        speed_factor = 1.0
        if player_speed < 2.0:  # Standing still or very slow (hesitating)
            speed_factor = 0.7  # 30% reduction
        elif player_speed < 5.0:  # Slow movement
            speed_factor = 0.85  # 15% reduction
        elif player_speed > 15.0:  # Fast sprint
            speed_factor = 1.1  # 10% boost
        
        base_probability *= speed_factor
        
        # Space/pressure factor
        # More opponents nearby = less space = lower probability
        pressure_factor = 1.0
        if nearby_opponents == 0:
            pressure_factor = 1.2  # 20% boost - open space
        elif nearby_opponents == 1:
            pressure_factor = 1.0  # Normal - one defender
        elif nearby_opponents == 2:
            pressure_factor = 0.75  # 25% reduction - two defenders
        elif nearby_opponents >= 3:
            pressure_factor = 0.5  # 50% reduction - heavily marked
        
        base_probability *= pressure_factor
        
        # Cap between 5% and 95%
        probability = max(5.0, min(95.0, base_probability))
        
        return probability
    
    def draw_probability(self, frame, probability, team):
        """Draw scoring probability on frame."""
        # Create smaller overlay box in bottom right
        overlay = frame.copy()
        box_x1, box_y1 = frame.shape[1] - 220, frame.shape[0] - 90
        box_x2, box_y2 = frame.shape[1] - 20, frame.shape[0] - 20
        
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 2)
        
        # Determine color based on probability
        if probability >= 70:
            prob_color = (0, 200, 0)  # Green
        elif probability >= 40:
            prob_color = (0, 165, 255)  # Orange
        else:
            prob_color = (0, 0, 200)  # Red
        
        # Draw text - more compact
        prob_text = f"{probability:.1f}%"
        team_text = f"T{team} Goal"
        
        cv2.putText(frame, team_text, 
                   (box_x1 + 10, box_y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(frame, prob_text, 
                   (box_x1 + 10, box_y1 + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, prob_color, 2)
        
        return frame
