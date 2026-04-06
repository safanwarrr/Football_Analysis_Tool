# 🚀 Quick Start Guide

## Running the Football Analysis System

### Step 1: Verify Setup

Run the test script to ensure everything is properly installed:

```bash
cd /Users/safanwar/football_analysis
python3 test_setup.py
```

You should see "All tests passed! System is ready to use."

### Step 2: Run the Analysis

Simply run the main script:

```bash
python3 main.py
```

**What happens:**
1. The system reads your video (`/Users/safanwar/Documents/Untitled.mp4`)
2. Downloads YOLOv8 model (first run only, ~130 MB)
3. Detects and tracks all players, referees, and the ball
4. Assigns players to teams based on t-shirt colors
5. Calculates ball possession for each team
6. Measures camera movement
7. Calculates player speeds and distances
8. Generates annotated output video

**Output:** `output/output_video.mp4`

### Step 3: View Results

Open the output video:

```bash
open output/output_video.mp4
```

Or on your system's video player.

## ⚡ Performance Notes

- **First run**: Slower (downloads model, processes everything)
- **Subsequent runs**: Faster (uses cached data from `stubs/`)
- **Video size**: Your input is 0.78 MB, so processing should be quick
- **Expected time**: 1-5 minutes depending on video length and hardware

## 🔧 Troubleshooting

### If the ball is not detected:

The pre-trained YOLO model may not detect footballs reliably. Options:

1. **Adjust detection confidence** in `trackers/tracker.py`:
   ```python
   detections_batch = self.model.predict(batch, conf=0.05)  # Lower threshold
   ```

2. **Train a custom model** (see README.md for instructions)

### If you want to process a different video:

Edit line 22 in `main.py`:

```python
video_path = '/path/to/your/video.mp4'
```

### If you want to force re-processing:

Delete the cached data:

```bash
rm -rf stubs/*.pkl
```

Then run `python3 main.py` again.

## 📊 Understanding the Output

The annotated video shows:

- **Colored ellipses** at player feet (team colors)
- **Player IDs** in boxes
- **Green triangle** for ball
- **Red triangle** on player with ball possession
- **Ball control percentage** overlay (bottom right)
- **Camera movement** overlay (top left)
- **Speed (km/h)** and **distance (m)** above each player

## 🎓 Next Steps

1. **Fine-tune for your video**: Adjust perspective transform coordinates in `perspective/view_transformer.py`
2. **Train custom model**: Follow instructions in `training/football_trainer.py`
3. **Process multiple videos**: Create a batch processing script
4. **Export metrics**: Add CSV export functionality to save player statistics

## 💡 Tips

- Use high-quality video for better results
- Ensure good lighting and clear t-shirt colors
- For best results, use a video with a relatively static camera angle
- The system works best with standard broadcast camera angles

## 📞 Need Help?

Check the full README.md for detailed documentation and configuration options.

---

**Ready to analyze your match! ⚽**
