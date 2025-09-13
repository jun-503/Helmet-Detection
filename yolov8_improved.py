import os
import yaml

import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import json

class HelmetDetectionTrainer:
    def __init__(self, config_path="config.yaml"):
        """Initialize the helmet detection trainer with configuration."""
        self.config = self.load_config(config_path)
        self.model = None
        self.results = None
        
    def load_config(self, config_path):
        """Load training configuration from YAML file."""
        default_config = {
            'model': {
                'name': 'yolov8n.pt',  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
                'pretrained': True
            },
            'data': {
                'yaml_path': 'data.yaml',
                'imgsz': 640,
                'batch': 16,
                'workers': 8
            },
            'training': {
                'epochs': 100,
                'patience': 20,
                'save_period': 10,
                'device': 'auto',  # 'auto', 'cpu', 'cuda', '0', '1', etc.
                'project': 'runs/detect',
                'name': f'helmet_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            },
            'optimization': {
                'optimizer': 'auto',  # 'auto', 'SGD', 'Adam', 'AdamW', 'RMSProp'
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1
            },
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0
            },
            'validation': {
                'val': True,
                'split': 0.2,
                'save_json': True,
                'save_hybrid': False,
                'conf': 0.001,
                'iou': 0.6,
                'max_det': 300,
                'half': False,
                'dnn': False,
                'plots': True
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge user config with defaults
            for key, value in user_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        else:
            # Create default config file
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            print(f"Created default config file: {config_path}")
            
        return default_config
    
    def setup_model(self):
        """Initialize the YOLO model."""
        model_name = self.config['model']['name']
        print(f"Loading model: {model_name}")
        
        if self.config['model']['pretrained']:
            self.model = YOLO(model_name)
        else:
            # Load from scratch (not recommended for most cases)
            self.model = YOLO(model_name.replace('.pt', '.yaml'))
            
        print(f"Model loaded successfully. Device: {self.model.device}")
        return self.model
    
    def train(self):
        """Train the model with comprehensive configuration."""
        if self.model is None:
            self.setup_model()
            
        print("Starting training...")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Prepare training arguments
        train_args = {
            'data': self.config['data']['yaml_path'],
            'epochs': self.config['training']['epochs'],
            'imgsz': self.config['data']['imgsz'],
            'batch': self.config['data']['batch'],
            'workers': self.config['data']['workers'],
            'device': self.config['training']['device'],
            'project': self.config['training']['project'],
            'name': self.config['training']['name'],
            'patience': self.config['training']['patience'],
            'save_period': self.config['training']['save_period'],
            
            # Optimization
            'optimizer': self.config['optimization']['optimizer'],
            'lr0': self.config['optimization']['lr0'],
            'lrf': self.config['optimization']['lrf'],
            'momentum': self.config['optimization']['momentum'],
            'weight_decay': self.config['optimization']['weight_decay'],
            'warmup_epochs': self.config['optimization']['warmup_epochs'],
            'warmup_momentum': self.config['optimization']['warmup_momentum'],
            'warmup_bias_lr': self.config['optimization']['warmup_bias_lr'],
            
            # Augmentation
            'hsv_h': self.config['augmentation']['hsv_h'],
            'hsv_s': self.config['augmentation']['hsv_s'],
            'hsv_v': self.config['augmentation']['hsv_v'],
            'degrees': self.config['augmentation']['degrees'],
            'translate': self.config['augmentation']['translate'],
            'scale': self.config['augmentation']['scale'],
            'shear': self.config['augmentation']['shear'],
            'perspective': self.config['augmentation']['perspective'],
            'flipud': self.config['augmentation']['flipud'],
            'fliplr': self.config['augmentation']['fliplr'],
            'mosaic': self.config['augmentation']['mosaic'],
            'mixup': self.config['augmentation']['mixup'],
            'copy_paste': self.config['augmentation']['copy_paste'],
            
            # Validation
            'val': self.config['validation']['val'],
            'split': self.config['validation']['split'],
            'save_json': self.config['validation']['save_json'],
            'save_hybrid': self.config['validation']['save_hybrid'],
            'conf': self.config['validation']['conf'],
            'iou': self.config['validation']['iou'],
            'max_det': self.config['validation']['max_det'],
            'half': self.config['validation']['half'],
            'dnn': self.config['validation']['dnn'],
            'plots': self.config['validation']['plots'],
            
            # Additional useful parameters
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val_period': 1
        }
        
        # Start training
        self.results = self.model.train(**train_args)
        
        print("Training completed!")
        return self.results
    
    def validate(self, model_path=None):
        """Validate the trained model."""
        if model_path is None:
            # Use the best model from training
            model_path = self.results.save_dir / 'weights' / 'best.pt'
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None
            
        print(f"Validating model: {model_path}")
        val_model = YOLO(model_path)
        
        val_args = {
            'data': self.config['data']['yaml_path'],
            'imgsz': self.config['data']['imgsz'],
            'batch': self.config['data']['batch'],
            'conf': self.config['validation']['conf'],
            'iou': self.config['validation']['iou'],
            'max_det': self.config['validation']['max_det'],
            'half': self.config['validation']['half'],
            'device': self.config['training']['device'],
            'plots': True,
            'save_json': True,
            'save_hybrid': False,
            'verbose': True
        }
        
        val_results = val_model.val(**val_args)
        return val_results
    
    def predict(self, source, model_path=None, save_results=True):
        """Run inference on new images/videos."""
        if model_path is None:
            model_path = self.results.save_dir / 'weights' / 'best.pt'
            
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None
            
        print(f"Running prediction with model: {model_path}")
        pred_model = YOLO(model_path)
        
        pred_args = {
            'source': source,
            'imgsz': self.config['data']['imgsz'],
            'conf': 0.25,  # Lower confidence for detection
            'iou': 0.45,
            'max_det': 1000,
            'device': self.config['training']['device'],
            'save': save_results,
            'save_txt': True,
            'save_conf': True,
            'save_crop': False,
            'show': False,
            'show_labels': True,
            'show_conf': True,
            'vid_stride': 1,
            'stream': False,
            'line_width': 3,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'boxes': True
        }
        
        results = pred_model.predict(**pred_args)
        return results
    
    def export_model(self, model_path=None, formats=['onnx', 'torchscript']):
        """Export the trained model to different formats."""
        if model_path is None:
            model_path = self.results.save_dir / 'weights' / 'best.pt'
            
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None
            
        print(f"Exporting model: {model_path}")
        export_model = YOLO(model_path)
        
        exported_paths = {}
        for fmt in formats:
            try:
                exported_path = export_model.export(format=fmt, imgsz=self.config['data']['imgsz'])
                exported_paths[fmt] = exported_path
                print(f"Exported to {fmt}: {exported_path}")
            except Exception as e:
                print(f"Failed to export to {fmt}: {e}")
                
        return exported_paths
    
    def plot_training_curves(self, save_path=None):
        """Plot and save training curves."""
        if self.results is None:
            print("No training results available. Train the model first.")
            return None
            
        # Plot training curves
        self.results.plot()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        return self.results
    
    def get_model_info(self):
        """Get detailed information about the model."""
        if self.model is None:
            self.setup_model()
            
        info = {
            'model_name': self.config['model']['name'],
            'device': str(self.model.device),
            'num_parameters': sum(p.numel() for p in self.model.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.model.parameters()) / (1024 * 1024)
        }
        
        print("Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
        return info

def main():
    """Main training pipeline."""
    # Initialize trainer
    trainer = HelmetDetectionTrainer()
    
    # Get model information
    trainer.get_model_info()
    
    # Train the model
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    results = trainer.train()
    
    # Validate the model
    print("\n" + "="*50)
    print("VALIDATING MODEL")
    print("="*50)
    val_results = trainer.validate()
    
    # Plot training curves
    print("\n" + "="*50)
    print("GENERATING PLOTS")
    print("="*50)
    trainer.plot_training_curves()
    
    # Export model
    print("\n" + "="*50)
    print("EXPORTING MODEL")
    print("="*50)
    exported_paths = trainer.export_model()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Results saved to: {results.save_dir}")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Last model: {results.save_dir}/weights/last.pt")
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()
