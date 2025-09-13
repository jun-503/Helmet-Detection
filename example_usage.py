"""
Example usage of the improved YOLOv8 helmet detection trainer.
This script demonstrates different ways to use the HelmetDetectionTrainer class.
"""

from yolov8_improved import HelmetDetectionTrainer
import os

def example_basic_training():
    """Basic training example with default configuration."""
    print("=== Basic Training Example ===")
    
    # Initialize trainer with default config
    trainer = HelmetDetectionTrainer()
    
    # Train the model
    results = trainer.train()
    
    # Validate the model
    val_results = trainer.validate()
    
    print("Basic training completed!")

def example_custom_config():
    """Example with custom configuration."""
    print("=== Custom Configuration Example ===")
    
    # Create custom config
    custom_config = {
        'model': {
            'name': 'yolov8s.pt',  # Use small model instead of nano
        },
        'training': {
            'epochs': 50,  # Fewer epochs for quick test
            'patience': 10,
        },
        'data': {
            'batch': 8,  # Smaller batch size
            'imgsz': 512,  # Smaller image size
        },
        'optimization': {
            'lr0': 0.005,  # Lower learning rate
        }
    }
    
    # Save custom config
    import yaml
    with open('custom_config.yaml', 'w') as f:
        yaml.dump(custom_config, f)
    
    # Initialize trainer with custom config
    trainer = HelmetDetectionTrainer('custom_config.yaml')
    
    # Train the model
    results = trainer.train()
    
    print("Custom configuration training completed!")

def example_inference():
    """Example of running inference on new images."""
    print("=== Inference Example ===")
    
    # Initialize trainer
    trainer = HelmetDetectionTrainer()
    
    # Train a model first (or load existing)
    if not os.path.exists('runs/detect/helmet_detection_*/weights/best.pt'):
        print("No trained model found. Training first...")
        trainer.train()
    
    # Run inference on test images
    test_images = "dataset/images/test"  # Path to test images
    if os.path.exists(test_images):
        results = trainer.predict(
            source=test_images,
            save_results=True
        )
        print(f"Inference completed! Results saved.")
    else:
        print(f"Test images directory not found: {test_images}")

def example_model_export():
    """Example of exporting trained model to different formats."""
    print("=== Model Export Example ===")
    
    # Initialize trainer
    trainer = HelmetDetectionTrainer()
    
    # Train a model first (or load existing)
    if not os.path.exists('runs/detect/helmet_detection_*/weights/best.pt'):
        print("No trained model found. Training first...")
        trainer.train()
    
    # Export to different formats
    exported_paths = trainer.export_model(
        formats=['onnx', 'torchscript', 'tflite']
    )
    
    print("Model export completed!")
    for format_name, path in exported_paths.items():
        print(f"  {format_name}: {path}")

def example_hyperparameter_tuning():
    """Example of hyperparameter tuning with different configurations."""
    print("=== Hyperparameter Tuning Example ===")
    
    # Different model sizes to try
    model_configs = [
        {'model': {'name': 'yolov8n.pt'}, 'training': {'epochs': 30}},
        {'model': {'name': 'yolov8s.pt'}, 'training': {'epochs': 30}},
        {'model': {'name': 'yolov8m.pt'}, 'training': {'epochs': 20}},
    ]
    
    best_model = None
    best_mAP = 0
    
    for i, config in enumerate(model_configs):
        print(f"\n--- Testing Configuration {i+1}/{len(model_configs)} ---")
        
        # Save config
        import yaml
        config_file = f'tune_config_{i}.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Train model
        trainer = HelmetDetectionTrainer(config_file)
        results = trainer.train()
        
        # Get validation results
        val_results = trainer.validate()
        
        # Extract mAP50 (you might need to adjust this based on actual results structure)
        try:
            mAP50 = val_results.box.map50 if hasattr(val_results, 'box') else 0
            print(f"mAP50: {mAP50:.4f}")
            
            if mAP50 > best_mAP:
                best_mAP = mAP50
                best_model = trainer
        except:
            print("Could not extract mAP50")
        
        # Clean up config file
        os.remove(config_file)
    
    print(f"\nBest model achieved mAP50: {best_mAP:.4f}")
    return best_model

def main():
    """Run all examples."""
    print("Helmet Detection Trainer Examples")
    print("=" * 50)
    
    # Choose which example to run
    examples = {
        '1': ('Basic Training', example_basic_training),
        '2': ('Custom Configuration', example_custom_config),
        '3': ('Inference', example_inference),
        '4': ('Model Export', example_model_export),
        '5': ('Hyperparameter Tuning', example_hyperparameter_tuning),
    }
    
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\nEnter example number (1-5) or 'all' to run all: ").strip()
    
    if choice == 'all':
        for name, func in examples.values():
            print(f"\n{'='*20} {name} {'='*20}")
            try:
                func()
            except Exception as e:
                print(f"Error in {name}: {e}")
    elif choice in examples:
        name, func = examples[choice]
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            func()
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Invalid choice. Running basic training...")
        example_basic_training()

if __name__ == "__main__":
    main()
