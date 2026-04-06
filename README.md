# ⚽ Football Analysis System

A comprehensive computer vision system for football match analysis using YOLO and advanced CV techniques.

## 🎯 Features

- **Object Detection & Tracking**: Detect and track players, referees, and footballs using YOLOv8
- **Team Assignment**: Automatically assign players to teams based on t-shirt colors using K-means clustering
- **Ball Possession Analysis**: Calculate and display ball acquisition percentage for each team
- **Camera Movement Compensation**: Use optical flow to measure and compensate for camera movement
- **Perspective Transformation**: Transform pixel coordinates to real-world meters on the pitch
- **Speed & Distance Metrics**: Calculate and display player speed (km/h) and distance covered (meters)
- **Custom Model Training**: Fine-tune YOLO on football-specific datasets for improved accuracy

## 📁 Project Structure

```
football_analysis/
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── utils/                      # Utility modules
│   ├── video_utils.py         # Video I/O operations
│   └── bbox_utils.py          # Bounding box utilities
│
├── trackers/                   # Object tracking
│   └── tracker.py             # YOLO-based tracker with ByteTrack
│
├── team_assignment/            # Team classification
│   ├── team_assigner.py       # K-means based team assignment
│   └── player_ball_assigner.py # Ball possession logic
│
├── camera_movement/            # Camera motion detection
│   └── camera_movement_estimator.py
│
├── perspective/                # Perspective transformation
│   ├── view_transformer.py    # Pixel to meter conversion
│   └── speed_and_distance_estimator.py
│
├── training/                   # Custom YOLO training
│   └── football_trainer.py    # Training script
│
├── models/                     # YOLO model weights
├── output/                     # Processed videos
└── stubs/                      # Cached processing results
```

## 🚀 Installation

1. **Clone or navigate to the project directory:**
```bash
cd football_analysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

The main dependencies are:
- `ultralytics` (YOLOv8)
- `opencv-python` (Computer Vision)
- `scikit-learn` (K-means clustering)
- `supervision` (Tracking utilities)
- `numpy`, `pandas`, `matplotlib`

## 📦 Download YOLO Model

The system uses YOLOv8x by default. The model will be automatically downloaded on first run, or you can download it manually:

```bash
# The model will auto-download to models/yolov8x.pt on first run
# Or download manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -P models/
```

## 💻 Usage

### Basic Usage

Run the analysis on your video:

```bash
cd football_analysis
python main.py
```

The script will:
1. Read the video from `/Users/safanwar/Documents/Untitled.mp4`
2. Detect and track all objects
3. Assign players to teams
4. Calculate ball possession
5. Measure camera movement
6. Calculate player speeds and distances
7. Generate annotated output video

**Output:** `output/output_video.mp4`

### Customize Input Video

Edit `main.py` and change the video path:

```python
video_path = '/path/to/your/video.mp4'
```

### Performance Optimization

The system caches results in `stubs/` directory to speed up re-runs. To force reprocessing:

```python
# In main.py, set read_from_stub=False
tracks = tracker.get_object_tracks(video_frames, 
                                    read_from_stub=False,  # Changed
                                    stub_path='stubs/track_stubs.pkl')
```

## 🎓 Training Custom YOLO Model

To improve detection accuracy, you can train a custom YOLO model on football-specific data:

1. **Prepare your dataset** in YOLO format:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

2. **Configure the dataset:**
```bash
cd training
python football_trainer.py  # Creates football_data.yaml
```

3. **Edit `football_data.yaml`** to point to your dataset

4. **Uncomment and run training code** in `football_trainer.py`

5. **Use the trained model:**
```python
# In main.py, change model path:
tracker = Tracker('runs/detect/football_yolov8/weights/best.pt')
```

### Dataset Sources

- [Roboflow Football Datasets](https://universe.roboflow.com/)
- [Football Player Detection Dataset](https://www.kaggle.com/datasets/)
- Create your own using labeling tools like [LabelImg](https://github.com/heartexlabs/labelImg)

## 🎨 Output Visualization

The output video includes:
- **Player bounding boxes** with team colors
- **Player IDs** for tracking
- **Ball indicator** (green triangle)
- **Ball possession** marker on player
- **Team ball control percentage** (overlay)
- **Camera movement** (X/Y displacement)
- **Player speed** (km/h) and **distance** (meters) above each player

## 🔧 Configuration

### Perspective Transformation

The pitch coordinates are defined in `perspective/view_transformer.py`. Adjust the `pixel_vertices` to match your camera angle:

```python
self.pixel_vertices = np.array([
    [110, 1035],   # Bottom left
    [265, 275],    # Top left
    [910, 260],    # Top right
    [1640, 915]    # Bottom right
])
```

### Team Color Clustering

Adjust K-means parameters in `team_assignment/team_assigner.py` if team detection is inaccurate.

### Camera Movement Sensitivity

Modify `minimum_distance` in `camera_movement/camera_movement_estimator.py`:

```python
self.minimum_distance = 5  # Increase for less sensitive detection
```

## 📊 System Pipeline

```
Input Video
    ↓
1. YOLO Detection (players, referees, ball)
    ↓
2. ByteTrack Tracking (maintain IDs)
    ↓
3. Team Assignment (K-means on t-shirt colors)
    ↓
4. Ball Possession (proximity-based)
    ↓
5. Camera Movement (optical flow)
    ↓
6. Position Adjustment (camera compensation)
    ↓
7. Perspective Transform (pixels → meters)
    ↓
8. Speed & Distance Calculation
    ↓
9. Visualization & Annotation
    ↓
Output Video
```

## 🐛 Troubleshooting

**Issue: "No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**Issue: Ball not detected**
- YOLO's "sports ball" class may not detect footballs well
- Train a custom model with football-specific data
- Adjust detection confidence threshold in `tracker.py`

**Issue: Wrong team assignments**
- Adjust the number of K-means clusters
- Ensure good lighting and clear t-shirt colors in video
- Use first frame with all players visible

**Issue: Inaccurate speed measurements**
- Verify perspective transformation keypoints
- Check frame rate setting (default: 24 fps)
- Ensure camera movement is properly detected

## 📝 Technical Details

- **Detection Model**: YOLOv8x (pre-trained on COCO)
- **Tracking Algorithm**: ByteTrack
- **Clustering**: K-means (k=2 for teams, k=2 for player/background)
- **Optical Flow**: Lucas-Kanade method
- **Perspective Transform**: OpenCV's `getPerspectiveTransform`
- **Video Codec**: MP4V

## 🤝 Contributing

To extend the system:
1. Add new metrics in `perspective/` directory
2. Improve team assignment logic in `team_assignment/`
3. Enhance visualization in `trackers/tracker.py`
4. Add referee tracking capabilities

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- ByteTrack tracking algorithm
- OpenCV community

---

**Author**: Football Analysis Team  
**Version**: 1.0.0  
**Last Updated**: January 2026
