# Cotton Disease Prediction Using Deep Learning

## Overview
This project focuses on building a deep learning-based image classification model to identify and classify diseases in cotton plants. Using Python and PyTorch, the model detects four categories of images:
1. Fresh Cotton Leaf
2. Fresh Cotton Plant
3. Diseased Cotton Leaf
4. Diseased Cotton Plant

The project leverages a convolutional neural network (CNN) to process and analyze images, achieving high accuracy on validation and test datasets.

## Features
- **Image Dataset**: Includes images of fresh and diseased cotton leaves and plants.
- **Data Preprocessing**: 
  - Image resizing to 32x32 pixels.
  - Data normalization using ImageNet statistics for optimal training.
  - Augmentation through PyTorch's `transforms` for better generalization.
- **Deep Learning Model**: 
  - A custom CNN architecture with multiple convolutional layers, ReLU activation, max-pooling layers, and fully connected layers.
  - Trained with the Adam optimizer and Cross-Entropy loss function.
- **Device Optimization**: Supports GPU acceleration for faster training and inference.
- **Performance Evaluation**:
  - Visualization of training and validation accuracy/loss trends.
  - Test set predictions with visualization for qualitative analysis.

## Repository Contents
- **Notebook**: The complete implementation of the model in a Python notebook using Google Colab.
- **Scripts**:
  - Dataset loading and preprocessing.
  - Custom CNN model definition.
  - Training, evaluation, and testing functions.
- **Model**: Trained model weights saved as `cotton_disease_model.pth`.

## Key Metrics
- Validation accuracy: ~91%
- Validation loss: Reduced steadily across epochs.
- Model generalizes well across unseen data.

## How to Use
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/cotton-disease-prediction.git
   cd cotton-disease-prediction
   ```
2. **Prepare the Dataset**:
   - Organize the dataset into `train`, `val`, and `test` directories.
   - Ensure the directory structure matches:
     ```
     dataset/
       train/
         fresh cotton leaf/
         fresh cotton plant/
         diseased cotton leaf/
         diseased cotton plant/
       val/
       test/
     ```
3. **Run the Notebook**:
   - Open the provided notebook in Google Colab or Jupyter Notebook.
   - Execute all cells to train and evaluate the model.

4. **Test the Model**:
   - Use the saved model weights (`cotton_disease_model.pth`) to predict classes for new images.

## Results
- Example test results:
  - **Input**: Diseased Cotton Leaf
  - **Predicted**: Diseased Cotton Leaf
- Model predictions are visualized alongside ground truth labels for evaluation.

## Dependencies
- Python 3.7+
- PyTorch
- Torchvision
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install torch torchvision matplotlib numpy
```

## Future Work
- Expand the dataset with additional classes for more diseases.
- Fine-tune the model with transfer learning from pre-trained networks.
- Deploy the model as a web or mobile application.

## Acknowledgments
- This project was inspired by the need for smart agricultural solutions for disease detection.
- Dataset and preprocessing guidelines are tailored for cotton plants.

