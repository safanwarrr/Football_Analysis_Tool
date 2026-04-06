import numpy as np

def get_center_of_bbox(bbox):
    """Get center point of bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    """Get width of bounding box."""
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_foot_position(bbox):
    """Get foot position (bottom center) of bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
