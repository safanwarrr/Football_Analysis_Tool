import numpy as np
import cv2

class ViewTransformer:
    def __init__(self):
        # Football pitch dimensions in meters (standard size)
        court_width = 68
        court_length = 23.32  # Partial pitch visible
        
        # Pixel vertices of the visible pitch area (to be adjusted based on video)
        # These are typical coordinates for a broadcast angle
        self.pixel_vertices = np.array([
            [110, 1035],  # Bottom left
            [265, 275],   # Top left
            [910, 260],   # Top right
            [1640, 915]   # Bottom right
        ])
        
        # Target vertices in meters (bird's eye view)
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])
        
        # Convert to float32 for cv2.getPerspectiveTransform
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)
        
        # Get transformation matrix
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
    
    def transform_point(self, point):
        """Transform a point from pixel coordinates to meters."""
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None
        
        reshaped_point = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transform_point.reshape(-1, 2)[0]
    
    def add_transformed_position_to_tracks(self, tracks):
        """Add transformed positions (in meters) to all tracks."""
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object_name][frame_num][track_id]['position_transformed'] = position_transformed
