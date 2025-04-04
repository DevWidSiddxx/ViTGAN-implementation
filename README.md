# Generative Adversarial Network (GAN) - Image Quality Evaluation

## Project Overview
This project implements a Generative Adversarial Network (GAN) and evaluates generated images using various quality metrics. The model uses **EfficientNetB0** for feature extraction and applies standard image similarity and classification metrics.

## Installation
Ensure you have the required dependencies installed before running the code.

```bash
pip install tensorflow opencv-python numpy scikit-image scipy scikit-learn
```

## Key Features
- **Image Quality Metrics:**
  - **SSIM (Structural Similarity Index Measurement)** - Higher is better
  - **PSNR (Peak Signal-to-Noise Ratio)** - Higher is better
  - **FID (Fréchet Inception Distance)** - Lower is better
  
- **Classification Metrics:**
  - **Accuracy**
  - **Precision**
  - **Recall (TPR - True Positive Rate)**
  - **F1 Score**
  - **False Positive Rate (FPR)**
  - **AUROC (Area Under the ROC Curve)**

## Usage
Run the following command to execute the model and generate results:

```python
python generate.py
```



## Model Details
- **Base Model:** EfficientNetB0
- **Dataset:** Custom dataset (Modify in `generate.py`)
- **Epochs:** To be configured based on dataset size
- **Hyperparameters:** Learning rate, batch size (Modify as needed)

## Troubleshooting
If you encounter import errors, ensure all dependencies are installed. You may also run:

```bash
pip install --upgrade tensorflow keras numpy
```

## Contributing
Feel free to fork this repository and enhance the project!

## License
MIT License
