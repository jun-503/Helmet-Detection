# Improved YOLOv8 Helmet Detection

This is an enhanced version of the YOLOv8 training script for helmet detection with comprehensive features, better configuration management, and advanced training capabilities.

## 📂 Dataset
- **Format:** Pascal VOC (`.png` + `.xml`)  
- **Size:** 764 images + annotations  
- **Classes:**  
  - `helmet`  

The `.xml` files were converted to YOLO format using a custom script (`convert_labels.py`).

**Dataset split:**
- **Train:** 70%  
- **Validation:** 20%  
- **Test:** 10%  

---
## 🚀 Key Improvements

### 1. **Object-Oriented Design**
- Clean, modular `HelmetDetectionTrainer` class
- Easy to extend and customize
- Better code organization and reusability

### 2. **Comprehensive Configuration Management**
- YAML-based configuration system
- Default configurations with sensible values
- Easy parameter tuning without code changes
- Automatic config file generation

### 3. **Advanced Training Features**
- **Data Augmentation**: HSV, rotation, translation, scaling, flipping, mosaic, mixup
- **Optimization**: Multiple optimizers (SGD, Adam, AdamW, RMSProp)
- **Learning Rate Scheduling**: Cosine annealing, warmup, decay
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) for faster training
- **Early Stopping**: Configurable patience to prevent overfitting

### 4. **Model Management**
- **Multiple Model Sizes**: Support for YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- **Model Export**: ONNX, TorchScript, TFLite formats
- **Model Information**: Detailed model statistics and parameters
- **Checkpoint Management**: Automatic saving and resuming

### 5. **Comprehensive Validation & Evaluation**
- **Detailed Metrics**: mAP, precision, recall, F1-score
- **Visualization**: Training curves, confusion matrices, PR curves
- **Validation Plots**: Automatic generation of evaluation plots
- **JSON Export**: Structured results for further analysis

### 6. **Inference & Prediction**
- **Batch Inference**: Process multiple images/videos
- **Confidence Filtering**: Adjustable confidence thresholds
- **Result Saving**: Multiple output formats (images, labels, crops)
- **Real-time Processing**: Support for webcam and video streams

### 7. **Monitoring & Logging**
- **Progress Tracking**: Detailed training progress
- **Model Statistics**: Parameter counts, model size, device info
- **Training Curves**: Automatic plotting and saving
- **Verbose Logging**: Comprehensive training information

## 📁 File Structure

```
Helmet Detection/
├── yolov8_improved.py          # Main improved training script
├── example_usage.py            # Usage examples and demos
├── requirements_improved.txt   # Enhanced dependencies
├── README_improved.md          # This documentation
├── config.yaml                 # Auto-generated configuration file
├── data.yaml                   # Dataset configuration
├── dataset/                    # YOLO format dataset
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── runs/                       # Training outputs
    └── detect/
        └── helmet_detection_*/
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            ├── results.png
            ├── confusion_matrix.png
            └── ...
```

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_improved.txt
   ```

2. **Verify Installation**:
   ```python
   from ultralytics import YOLO
   print("YOLO installation successful!")
   ```

## 🚀 Quick Start

### Basic Usage

```python
from yolov8_improved import HelmetDetectionTrainer

# Initialize trainer (creates default config.yaml)
trainer = HelmetDetectionTrainer()

# Train the model
results = trainer.train()

# Validate the model
val_results = trainer.validate()

# Run inference on test images
results = trainer.predict("dataset/images/test")
```

### Custom Configuration

```python
# Create custom config
custom_config = {
    'model': {'name': 'yolov8s.pt'},
    'training': {'epochs': 100, 'patience': 30},
    'data': {'batch': 32, 'imgsz': 640},
    'optimization': {'lr0': 0.01, 'momentum': 0.937}
}

# Save and use custom config
import yaml
with open('my_config.yaml', 'w') as f:
    yaml.dump(custom_config, f)

trainer = HelmetDetectionTrainer('my_config.yaml')
results = trainer.train()
```

## ⚙️ Configuration Options

### Model Configuration
```yaml
model:
  name: "yolov8n.pt"  # Model size: n, s, m, l, x
  pretrained: true     # Use pretrained weights
```

### Training Configuration
```yaml
training:
  epochs: 100          # Number of training epochs
  patience: 20         # Early stopping patience
  device: "auto"       # Device selection
  project: "runs/detect"  # Output directory
  name: "helmet_detection"  # Experiment name
```

### Data Configuration
```yaml
data:
  yaml_path: "data.yaml"  # Dataset config file
  imgsz: 640             # Image size
  batch: 16              # Batch size
  workers: 8             # Number of workers
```

### Optimization Configuration
```yaml
optimization:
  optimizer: "auto"     # Optimizer type
  lr0: 0.01            # Initial learning rate
  lrf: 0.01            # Final learning rate
  momentum: 0.937       # SGD momentum
  weight_decay: 0.0005  # Weight decay
  warmup_epochs: 3      # Warmup epochs
```

### Augmentation Configuration
```yaml
augmentation:
  hsv_h: 0.015         # HSV hue augmentation
  hsv_s: 0.7           # HSV saturation augmentation
  hsv_v: 0.4           # HSV value augmentation
  degrees: 0.0         # Rotation degrees
  translate: 0.1       # Translation fraction
  scale: 0.5           # Scale range
  fliplr: 0.5          # Horizontal flip probability
  mosaic: 1.0          # Mosaic augmentation probability
  mixup: 0.0           # Mixup augmentation probability
```

## 📊 Advanced Features

### 1. Hyperparameter Tuning
```python
# Test different model sizes
configs = [
    {'model': {'name': 'yolov8n.pt'}},
    {'model': {'name': 'yolov8s.pt'}},
    {'model': {'name': 'yolov8m.pt'}}
]

best_model = None
best_mAP = 0

for config in configs:
    trainer = HelmetDetectionTrainer(config)
    results = trainer.train()
    val_results = trainer.validate()
    
    # Compare results and select best
    if val_results.box.map50 > best_mAP:
        best_mAP = val_results.box.map50
        best_model = trainer
```

### 2. Model Export
```python
# Export to multiple formats
exported_paths = trainer.export_model(
    formats=['onnx', 'torchscript', 'tflite']
)
```

### 3. Custom Inference
```python
# Run inference with custom settings
results = trainer.predict(
    source="path/to/images",
    conf=0.25,          # Confidence threshold
    iou=0.45,           # NMS IoU threshold
    save_results=True,  # Save results
    save_txt=True,      # Save labels
    save_conf=True      # Save confidence scores
)
```

## 📈 Monitoring Training

### Training Curves
```python
# Plot and save training curves
trainer.plot_training_curves("training_curves.png")
```

### Model Information
```python
# Get detailed model statistics
info = trainer.get_model_info()
print(f"Model size: {info['model_size_mb']:.2f} MB")
print(f"Parameters: {info['num_parameters']:,}")
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `batch: 8` or `batch: 4`
   - Reduce image size: `imgsz: 416`
   - Use smaller model: `yolov8n.pt`

2. **Slow Training**:
   - Increase workers: `workers: 16`
   - Enable AMP: `amp: true`
   - Use GPU: `device: "cuda"`

3. **Poor Performance**:
   - Increase epochs: `epochs: 200`
   - Adjust learning rate: `lr0: 0.005`
   - Enable more augmentation
   - Use larger model: `yolov8m.pt` or `yolov8l.pt`

### Performance Tips

1. **For Speed**: Use `yolov8n.pt` with `imgsz: 416`
2. **For Accuracy**: Use `yolov8l.pt` or `yolov8x.pt` with `imgsz: 640`
3. **For Balance**: Use `yolov8s.pt` or `yolov8m.pt`

## 📚 Examples

Run the example script to see different usage patterns:

```bash
python example_usage.py
```

Available examples:
1. Basic Training
2. Custom Configuration
3. Inference
4. Model Export
5. Hyperparameter Tuning

## 🤝 Contributing

Feel free to extend the `HelmetDetectionTrainer` class with additional features:

- Custom loss functions
- Advanced augmentation strategies
- Model ensemble methods
- Real-time monitoring
- Integration with experiment tracking tools (Weights & Biases, TensorBoard)

## 📄 License

This project follows the same license as the original YOLOv8 implementation.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base implementation
- The computer vision community for best practices and techniques

## Author
Muhammad Junaid Zia