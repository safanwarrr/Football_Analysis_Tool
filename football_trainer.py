#!/usr/bin/env python3
"""
Football-specific YOLO training script
Fine-tunes YOLOv8 on football dataset for improved detection of players, referees, and ball.
"""

from ultralytics import YOLO

class FootballTrainer:
    def __init__(self, base_model='yolov8x.pt'):
        """Initialize trainer with base YOLO model."""
        self.model = YOLO(base_model)
    
    def train(self, data_yaml, epochs=100, imgsz=640, batch=16):
        """
        Train the model on football dataset.
        
        Args:
            data_yaml: Path to data.yaml file with dataset configuration
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='football_yolov8',
            patience=20,
            save=True,
            device=0,  # Use GPU if available
            workers=8,
            optimizer='Adam',
            lr0=0.001,
            weight_decay=0.0005,
            warmup_epochs=5,
            mosaic=1.0,
            augment=True
        )
        
        return results
    
    def validate(self, data_yaml):
        """Validate the trained model."""
        metrics = self.model.val(data=data_yaml)
        return metrics
    
    def export(self, format='onnx'):
        """Export model to different format."""
        self.model.export(format=format)

def create_dataset_config():
    """
    Create a sample data.yaml configuration file for football dataset.
    
    You need to prepare a dataset with the following structure:
    dataset/
        train/
            images/
            labels/
        val/
            images/
            labels/
    
    Labels should be in YOLO format (class x_center y_center width height)
    Classes: 0=player, 1=referee, 2=ball, 3=goalkeeper
    """
    config = """
# Football Detection Dataset Configuration
path: ../dataset  # Dataset root directory
train: train/images  # Train images
val: val/images  # Validation images

# Classes
names:
  0: player
  1: referee
  2: ball
  3: goalkeeper

# Number of classes
nc: 4
"""
    
    with open('football_data.yaml', 'w') as f:
        f.write(config)
    
    print("✓ Created football_data.yaml configuration file")
    print("\nNext steps:")
    print("1. Prepare your dataset in the required structure")
    print("2. You can use Roboflow or similar platforms to get football datasets")
    print("3. Update the 'path' in football_data.yaml to point to your dataset")
    print("4. Run: python football_trainer.py")

if __name__ == '__main__':
    # Example usage
    print("Football YOLO Training Script")
    print("=" * 50)
    
    # Create dataset configuration
    create_dataset_config()
    
    # Uncomment below to train (after preparing dataset)
    """
    trainer = FootballTrainer('yolov8x.pt')
    
    # Train the model
    results = trainer.train(
        data_yaml='football_data.yaml',
        epochs=100,
        imgsz=640,
        batch=16
    )
    
    # Validate
    metrics = trainer.validate('football_data.yaml')
    print(f"Validation metrics: {metrics}")
    
    # The trained model will be saved in runs/detect/football_yolov8/weights/best.pt
    """
    
    print("\n" + "=" * 50)
    print("Training script ready!")
    print("Prepare your dataset and uncomment the training code to start.")
