# VITGAN: Vision Transformer GAN for Image Generation and Analysis

## Overview
VITGAN (Vision Transformer GAN) is an advanced deep learning model that leverages Vision Transformers (ViT) and Generative Adversarial Networks (GANs) for high-quality image generation and analysis. This project integrates **EfficientNet**, **InceptionV3**, and **structural similarity metrics** to assess image quality and model performance.

## Features
- **Vision Transformer (ViT) based GAN** for image generation.
- **EfficientNet & InceptionV3** for feature extraction.
- **Image Quality Metrics**: SSIM, PSNR, and FID.
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score, AUROC.
- **Pre-trained Model Weights** for faster training.

## Installation
To set up the environment, install the required dependencies:

```bash
pip install tensorflow numpy opencv-python scikit-image scipy scikit-learn
```

## Dataset
Ensure the dataset is structured properly before training:
```
/data
├── train
│   ├── class_1
│   ├── class_2
│   └── ...
├── test
│   ├── class_1
│   ├── class_2
│   └── ...
```

## Training
To train VITGAN, run:
```python
python train.py --epochs 50 --batch_size 32 --learning_rate 0.0002
```
### Hyperparameters
- **Epochs:** 50
- **Batch Size:** 32
- **Learning Rate:** 0.0002

## Evaluation
Run evaluation on test images:
```python
evaluate.py --model_path vitgan_model.h5
```

### Metrics
- **SSIM**: Measures structural similarity.
- **PSNR**: Peak Signal-to-Noise Ratio.
- **FID**: Fréchet Inception Distance.
- **AUROC, Precision, Recall, F1-score** for classification tasks.


## Troubleshooting
If you encounter issues:
- Ensure all dependencies are installed properly.
- Check the dataset structure.
- Verify model parameters before training.

## Contributors
- **Siddharth** - Project Lead & Developer
- - **Abinaya Vina** -  Developer
  - - **Prajasree** -  Developer

## License
This project is licensed under the MIT License.


