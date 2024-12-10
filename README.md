# Cats vs. Dogs Image Classifier

## Description
This project implements a simple Convolutional Neural Network (CNN) to classify images of cats and dogs. The model is trained on 8,000 images and validated on 2,000 images from the Cats vs. Dogs dataset.

## Preprocessing
- **Rescaling:** All images are normalized to pixel values in the range `[0, 1]`.
- **Data Augmentation:** Applied rotation, shifting, zooming, and flipping to improve generalization.

## Model Architecture
1. **Conv2D Layer 1:** 32 filters, 3x3 kernel, ReLU activation, followed by MaxPooling.
2. **Conv2D Layer 2:** 64 filters, 3x3 kernel, ReLU activation, followed by MaxPooling.
3. **Conv2D Layer 3:** 128 filters, 3x3 kernel, ReLU activation, followed by MaxPooling.
4. **Fully Connected Layer:** 128 units with ReLU activation and 50% dropout.
5. **Output Layer:** Single neuron with Sigmoid activation for binary classification.

## Results
- **Validation Accuracy:** ~82% after 15 epochs.
- **Training and Validation Curves:** See `results/` folder.

## Instructions
1. Run the script:
   ```bash
   python model.py
