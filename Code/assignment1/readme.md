# Project Name

## Description

Assignments 1 for Deciphex Interview

## Code Structure and Introduction

### Data Directory

The `data` directory is organized as follows:

- `train`
  - `images`: Contains the original training images.
  - `mask`: Contains the corresponding masks for segmentation.
- `val`: Holds validation images and masks.
- `test`: Stores the images and masks for final testing.

### Source Directory

The `src` directory contains the following files and subdirectories:

- `models`
  - `unet.py`: Implements the U-Net architecture with the pretrained ResNet-50 as the backbone. U-Net is famous for its efficiency in biomedical image segmentation. Using ResNet-50 as the backbone adds the benefit of deep residual learning, enhancing the ability to learn from complex patterns in the data. The combination of U-Net with ResNet-50 not only maintains high-resolution features throughout the network but also promotes faster convergence and potentially better performance on segmentation tasks.

- `utils`
  - `dataaugment.py`: Defines functions for data augmentation, including random cropping, rotation, color jittering, and brightness and contrast enhancement. These techniques aim to prevent overfitting and improve model generalization.
  - `dataloader.py`: Handles data loading, preprocessing into tensors, class weights computation, image normalization using previously calculated mean and standard deviation, and the transformation of masks into one-hot format. Additionally, it sets the buffer size, batch size, and shuffle parameters.
  - `help_functions.py`: Provides visualization tools for the original images, predicted results, and ground truth. These visualizations are essential for monitoring the training process and debugging.
  - `losses.py`: Specifies the loss functions, employing cross-entropy loss with a weight factor and supervision of the intermediate layer of the U-Net. It also details the combination of losses with specific weight factors to ensure model convergence.
  - `metrics.py`: Describes the metrics used for evaluation, during training process, primarily focusing on mean Intersection over Union (mIoU). A function for class-wise IoU is also included, if needed. Besides, for the evaluation phase, I calculated the metrics in assignment 2 as well.
  - `inference.py`: Contains the inference procedure to observe the model's performance on unseen data.
  - `preprocess.py`: Deals with the preprocessing steps, including the calculation of mean and standard deviation for normalization.
  - `train.py`: Manages the training process using the Adam optimizer with exponential decay learning rate. It includes visualization of the loss, training mean IoU, and validation mean IoU, as well as model saving functionality.
  - `evaluate.py`: Defines the evaluation procedures for the trained model.

## Setup and Installation

conda create -n deciphex python=3.8  
pip install -r requirements.txt

## Usage

- Download data (https://drive.google.com/file/d/1Dbu9EtPZF0XWiWGRd49zF8tukq9hOw6p/view?usp=sharing) and unzip it to ./
- Download checkpoints folder (https://drive.google.com/file/d/1CrCFWFhUqtHVHW1gNKy0l3lHl85uyIiL/view?usp=sharing) and unzip it to ./
- python train.py
- python ./src/inference.py --model_path ./saved_models/model_epoch_90_best_miou_0.9621.h5  
- python ./src/evaluation.py --model_path ./saved_models/model_epoch_90_best_miou_0.9621.h5  

