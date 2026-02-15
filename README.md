# Plant Disease Detection Model

## Model Information
- Architecture: efficientnet_b0
- Input size: 224x224
- Classes: ['Healthy', 'Diseased']
- Temperature: 1.0

## Usage
1. Load the ONNX model using ONNX Runtime
2. Preprocess images: resize to 224x224, normalize with ImageNet mean/std
3. Run inference to get calibrated probabilities
4. Apply threshold for "uncertain" cases

## Files
- `best_model.onnx`: ONNX model
- `model_metadata.json`: Model configuration and metrics

## Uncertainty Thresholding
Use MC-Dropout or MSP/Energy scores to detect uncertain predictions that need review.
