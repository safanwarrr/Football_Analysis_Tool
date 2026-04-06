import cv2
import numpy as np

class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24
        
    def add_speed_and_distance_to_tracks(self, tracks):
        """Calculate and add speed and distance information to tracks."""
        total_distance = {}
        
        for object_name, object_tracks in tracks.items():
            if object_name == "ball" or object_name == "referees":
                continue
            
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
                
                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue
                    
                    # Use position_adjusted (camera-adjusted pixel coordinates)
                    start_position = object_tracks[frame_num][track_id].get('position_adjusted')
                    end_position = object_tracks[last_frame][track_id].get('position_adjusted')
                    
                    if start_position is None or end_position is None:
                        continue
                    
                    # Calculate distance in pixels
                    distance_covered_pixels = self.calculate_distance(start_position, end_position)
                    
                    # Convert pixels to approximate meters (rough estimate: 1 pixel ≈ 0.1 meters)
                    # This is a rough approximation and should be calibrated for accurate results
                    pixels_per_meter = 10  # Adjust this based on your video
                    distance_covered = distance_covered_pixels / pixels_per_meter
                    
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6
                    
                    if track_id not in total_distance:
                        total_distance[track_id] = 0
                    
                    total_distance[track_id] += distance_covered
                    
                    for frame_num_batch in range(frame_num, last_frame + 1):
                        if track_id not in tracks[object_name][frame_num_batch]:
                            continue
                        tracks[object_name][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object_name][frame_num_batch][track_id]['distance'] = total_distance[track_id]
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def draw_speed_and_distance(self, frames, tracks):
        """Draw speed and distance annotations on frames."""
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            for object_name, object_tracks in tracks.items():
                if object_name == "ball" or object_name == "referees":
                    continue
                
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:
                            continue
                        
                        bbox = track_info['bbox']
                        position = self.get_center_of_bbox(bbox)
                        position = list(position)
                        position[1] += 40
                        
                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            output_frames.append(frame)
        
        return output_frames
    
    @staticmethod
    def get_center_of_bbox(bbox):
        """Get center of bounding box."""
        return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
