# Model Architectures

## FastAI Model Architectures

### FastAI Model Version 1 Architecture

1. **Data Preparation:**
   - **DataLoader Creation:** Utilizes FastAI methods to create a `DataLoader` object, encompassing both training and validation data loaders.
   - **Batch-Level Transformations:**
     - **Resize:** Each image is resized to 224x224 pixels.
     - **Random Scaling:** Applies a minimum scale of 75% of the original image size to each image.
     - **Data Augmentation (`aug_transforms`):** Incorporates various image augmentations, including:
       - **Rotation:** Randomly rotates images to simulate different angles.
       - **Cropping:** Randomly crops images to enhance model robustness.
       - **Color Jittering:** Adjusts brightness, contrast, saturation, and hue to improve generalization.

2. **Model Architecture:**
   - **Base Network:** Uses the ResNet34 model, a pre-defined Convolutional Neural Network (CNN) with 34 layers, known for its deep learning capabilities and residual connections to facilitate training.

### FastAI Model Version 2 Architecture

1. **Enhanced Data Augmentation:**
   - **Augmentation Details:**
     - **Rotation:** Allows random rotations up to 30 degrees.
     - **Zooming:** Applies scaling up to 10% larger than the original size.
     - **Lighting Adjustments:** Introduces random changes to lighting, including brightness and contrast adjustments.
     - **Warping:** Applies random distortions to images to simulate various transformations.

2. **Model Training:**
   - **Base Network:** Utilizes the same ResNet34 architecture.
   - **Learning Rate Finder (`lr_find`):** Employs the learning rate finder to identify the optimal learning rate. This pre-training phase helps in determining the best rate for efficient model training.

## PyTorch Model Architecture

### PyTorch Model 1 Architecture

1. **Network Architecture:**
   - **Convolutional Layers:**
     - **Layer 1:** 3x3 convolution with 32 filters, stride of 1, and padding of 1.
     - **Layer 2:** 3x3 convolution with 64 filters, stride of 1, and padding of 1.
     - **Layer 3:** 3x3 convolution with 128 filters, stride of 1, and padding of 1.
   - **Max Pooling Layer:** Applies a 2x2 max pooling operation with a stride of 2, reducing the spatial dimensions of the feature maps by half.
   - **Fully Connected Layers:**
     - **Hidden Layer:** Linear layer mapping from the flattened output of the final convolutional layer to 512 features.
     - **Output Layer:** Final linear layer producing class predictions based on the number of classes in the dataset (`len(train_dataset.classes)`).

2. **Forward Propagation:**
   - **Convolutional Processing:** Each convolutional layer is followed by ReLU activation and max pooling to extract features.
   - **Flattening:** The output from the final convolutional layer is flattened into a 1D tensor.
   - **Fully Connected Layers:** 
     - **Hidden Layer Activation:** Applies ReLU activation to the output of the first fully connected layer.
     - **Output Prediction:** The final fully connected layer generates predictions for each class based on the number of classes in the dataset.

### PyTorch Model 2 Architecture:
1. **Network Architecture:**
   - **Convolutional Layers:**
     - **Layer 1:** 3x3 convolution with 64 filters, stride of 1, and padding of 1, followed by Batch Normalization.
     - **Layer 2:** 3x3 convolution with 128 filters, stride of 1, and padding of 1, followed by Batch Normalization.
     - **Layer 3:** 3x3 convolution with 256 filters, stride of 1, and padding of 1, followed by Batch Normalization.
   - **Max Pooling Layer:** Applies a 2x2 max pooling operation with a stride of 2, reducing the spatial dimensions of the feature maps by half.
   - **Fully Connected Layers:**
     - **Hidden Layer:** Linear layer mapping from the flattened output of the final convolutional layer to 1024 features.
     - **Dropout Layer:** Applies dropout with a probability of 0.5 to prevent overfitting.
     - **Output Layer:** Final linear layer producing class predictions based on the number of classes in the dataset (`len(train_dataset.classes)`).

2. **Forward Propagation:**
   - **Convolutional Processing:** Each convolutional layer is followed by Batch Normalization, ReLU activation, and max pooling.
   - **Flattening:** The output from the final convolutional layer is flattened into a 1D tensor.
   - **Fully Connected Layers:** 
     - **Hidden Layer Activation:** Applies ReLU activation to the output of the first fully connected layer.
     - **Dropout:** Regularizes the model by applying dropout before the final output layer.
     - **Output Prediction:** The final fully connected layer generates predictions for each class based on the number of classes in the dataset.