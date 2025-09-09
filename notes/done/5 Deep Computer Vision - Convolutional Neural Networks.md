# Module 5: Deep Computer Vision - Convolutional Neural Networks


## Chunk 1 (3:43:10 - 3:53:10)

# Deep Computer Vision: Convolutional Neural Networks

## Introduction to Deep Computer Vision

- Deep computer vision is a fascinating and rapidly evolving field.
- Applications include:
  - **Self-driving cars**: Companies like Tesla utilize TensorFlow deep learning models for computer vision.
  - **Medicine**: Used for diagnostic imaging and analysis.
  - **Sports**: Employed in goal-line technology and player analysis.
- In this module, we focus on **image classification**, although the techniques can also be applied to:
  - Object detection and recognition
  - Facial detection and recognition

## Convolutional Neural Networks (CNNs)

### Overview

- CNNs are a type of deep learning model specifically designed for processing image data.
- Key components of CNNs include:
  - **Convolutional layers**
  - **Pooling layers**
- CNNs are often built using pre-trained models from companies like Google and TensorFlow for classification tasks.

### Image Data

- Image data differs from regular data as it is three-dimensional:
  - **Height**
  - **Width**
  - **Color channels** (Red, Green, Blue)
- Each pixel in an image is represented by three values, corresponding to the RGB color channels.

### Dense Neural Networks vs. Convolutional Neural Networks

#### Dense Neural Networks

- Operate on a **global scale**, learning patterns in specific areas of an image.
- Require images to be centered and similar in orientation.
- Struggle with recognizing patterns if the image is flipped or rotated.

#### Convolutional Neural Networks

- Learn **local patterns** that can be recognized anywhere in the image.
- Use convolutional layers to scan the entire image, identifying features regardless of their location.
- Output **feature maps** that indicate the presence of specific features or filters.

## How CNNs Work

### Convolutional Layers

- Convolutional layers apply **filters** over the image to detect patterns.
- These filters are small and simple, such as straight lines or edges.
- The output is a **feature map** that quantifies the presence of these patterns across the image.

### Pooling Layers

- Pooling layers reduce the spatial dimensions of the feature maps, retaining essential information while reducing computational load.

### Example: Image Recognition

- A CNN can recognize features like eyes or ears anywhere in an image, unlike a dense network that requires these features to be in specific locations.
- The CNN processes the image through multiple layers, each building upon the feature maps from the previous layer to recognize more complex patterns.

### Advantages of CNNs

- **Flexibility**: Can identify patterns regardless of their position in the image.
- **Efficiency**: By focusing on local patterns, CNNs require less data preprocessing compared to dense networks.

## Conclusion

- CNNs are powerful tools for image classification and recognition tasks.
- Understanding the difference between dense and convolutional networks is crucial for effectively applying these models.
- Further exploration of CNNs includes understanding pooling layers and advanced architectures.

For a deeper understanding, consider reviewing additional resources or the detailed descriptions provided in the accompanying notebook.


## Chunk 2 (3:53:10 - 4:03:10)

## Deep Computer Vision: Convolutional Neural Networks

### Convolutional Layers and Filters

- **Convolutional Layer Purpose**: 
  - To extract meaningful features from an image.
  - Outputs a **feature map** indicating the presence of specific patterns or **filters**.

- **Filters**:
  - Defined as patterns of pixels.
  - Applied across the image to detect specific features.
  - **Trainable Parameters**: Filters are learned and adjusted during training.
  - Commonly used filter sizes: 3x3.
  - Typical number of filters: 32, 64, or 128.

- **Input Size and Filter Size**:
  - Input size refers to the dimensions of the image being processed.
  - Filter size is the dimension of the pattern being searched for in the image.

### Feature Map Generation

- **Process**:
  1. Apply filters across the image.
  2. Calculate the **dot product** between the filter and sections of the image.
  3. Output a feature map indicating the similarity between the filter and image sections.

- **Example**:
  - For a 5x5 image with 3x3 filters:
    - The feature map size will be reduced (e.g., 3x3) due to the convolution process.
    - Each position in the feature map represents the similarity score between the filter and the corresponding image section.

### Multi-layer Feature Extraction

- **Layer Stacking**:
  - Multiple convolutional layers are stacked to detect increasingly complex features.
  - Initial layers detect simple features like edges and lines.
  - Subsequent layers detect complex structures like curves, shapes, and eventually objects (e.g., eyes, faces).

### Pooling and Padding

- **Pooling**:
  - A method to reduce the spatial dimensions of the feature map.
  - Helps in managing computational load and improving efficiency.

- **Padding**:
  - Adding extra rows and columns around the image.
  - Ensures the output feature map maintains the same dimensions as the input.
  - Helps in preserving edge information during convolution.

### Key Concepts

- **Dot Product**:
  - Element-wise multiplication of filter and image section.
  - Results in a similarity score indicating feature presence.

- **Feature Map**:
  - A reduced representation of the image highlighting detected features.
  - Depth corresponds to the number of filters used.

### Summary

Convolutional Neural Networks (CNNs) use convolutional layers to extract features from images. These layers apply filters to detect patterns, creating feature maps that highlight the presence of specific features. By stacking multiple layers, CNNs can identify complex structures in images. Techniques like pooling and padding are used to manage dimensions and computational efficiency. Understanding these processes is crucial for leveraging CNNs in deep computer vision tasks.


## Chunk 3 (4:03:10 - 4:13:10)

# Deep Computer Vision: Convolutional Neural Networks

## Key Concepts

### Padding and Stride

- **Padding**: 
  - Adding extra pixels around the border of an image.
  - Ensures that all pixels, including those on the edges, can be the center of a sample.
  - Helps generate an output map of the same size as the input.
  - Useful for detecting features at the edges of images.

- **Stride**:
  - Defines how much the sample box moves across the image.
  - A stride of 1 moves the box one pixel at a time.
  - A stride of 2 moves the box two pixels at a time, resulting in a smaller output feature map.
  - Larger strides reduce the size of the output feature map.

### Pooling Operations

- **Purpose**: Simplifies the output feature map by reducing its dimensionality.
- **Types of Pooling**:
  - **Max Pooling**: Takes the maximum value from a sample.
  - **Min Pooling**: Takes the minimum value from a sample.
  - **Average Pooling**: Takes the average value from a sample.
- **Typical Configuration**:
  - Use a 2x2 pooling size with a stride of 2.
  - Reduces the feature map size by half.

### Use Cases for Pooling

- **Max Pooling**: 
  - Identifies the presence of a feature in a local area.
  - Useful for detecting if a feature exists.

- **Average Pooling**:
  - Provides the average presence of a feature.
  - Less commonly used compared to max pooling.

- **Min Pooling**:
  - Determines if a feature does not exist in a local area.

## Building a Convolutional Neural Network (CNN)

### Dataset and Tools

- **Dataset**: CIFAR-10
  - Contains 60,000 images of 10 different classes (e.g., truck, car, ship, airplane).
  - Images are 32x32 pixels, colorful but blurry.

- **Tools**: 
  - Use **Keras** for building the CNN.
  - Import the CIFAR-10 dataset using Keras' built-in functions.

### Steps to Build a CNN

1. **Import Libraries**:
   - Import TensorFlow and Keras.
   - Load the CIFAR-10 dataset.

2. **Normalize Data**:
   - Normalize image data by dividing by 255 to ensure values are between 0 and 1.

3. **Define Class Names**:
   - Create a list of class names corresponding to the dataset labels.

4. **Visualize Data**:
   - Display sample images to understand the dataset.

### CNN Architecture

- **Structure**:
  - Stack convolutional layers with pooling layers.
  - After each convolutional layer, typically add a pooling layer to reduce dimensionality.

- **Layer Configuration**:
  - **Convolutional Layer**:
    - Define the number of filters, filter size, and activation function (e.g., ReLU).
    - Specify input shape for the first layer (e.g., 32x32x3 for CIFAR-10).

  - **Pooling Layer**:
    - Use a 2x2 pooling size with a stride of 2 to halve the feature map size.

### Example Architecture

- **First Layer**:
  - Convolutional layer with 32 filters, 3x3 filter size, ReLU activation.
  - Input shape: 32x32x3.

- **Second Layer**:
  - Max pooling layer with output shape: 15x15x32.

- **Subsequent Layers**:
  - Additional convolutional and pooling layers with increased filters (e.g., 64 filters).

### Summary

- **Output Shapes**:
  - Convolutional layers reduce spatial dimensions without padding.
  - Pooling layers further reduce dimensions by a factor of two.
- **Example Output**:
  - Initial convolutional layer output: 30x30x32.
  - After max pooling: 15x15x32.
  - Further convolution and pooling: 13x13x64, then 6x6x64.

These notes provide a structured overview of the concepts and processes involved in building a convolutional neural network, focusing on padding, stride, pooling operations, and the architecture of a CNN using the CIFAR-10 dataset.


## Chunk 4 (4:13:10 - 4:23:10)

## Deep Computer Vision: Convolutional Neural Networks (CNNs)

### Overview of CNN Architecture

- **Convolution Base**: 
  - A stack of convolution and max pooling layers.
  - Extracts features from images.
  - Example: A layer might transform input to a 4x4x64 feature map.

- **Dense Layer Classifier**:
  - Converts extracted features into class predictions.
  - Uses a dense network to map feature combinations to specific classes.

### Building a CNN Model

1. **Flattening**:
   - Convert 4x4x64 feature map into a one-dimensional array.
   - Facilitates connection to dense layers.

2. **Dense Layers**:
   - **First Dense Layer**: 
     - 64 neurons with a Rectified Linear Unit (ReLU) activation function.
   - **Output Layer**:
     - 10 neurons (corresponding to the number of classes).
     - Outputs a list of values for class prediction.

3. **Model Summary**:
   - Transition from feature map to dense layers.
   - Example: 4x4x64 becomes 1024 (4x4x64) in the flattened layer.

### Training the CNN Model

- **Epochs**:
  - Recommended: 10 epochs for better accuracy (~70%).
  - Example: Training with 4 epochs yields ~67% accuracy.

- **Optimizers and Loss Functions**:
  - **Optimizer**: Adam
  - **Loss Function**: Sparse Categorical Cross Entropy
    - Computes cross-entropy loss between labels and predictions.

- **Training Process**:
  - Use `model.fit` to train with training and validation datasets.
  - Evaluate model accuracy with test datasets.

### Working with Small Datasets

#### Challenges:
- Small datasets make it difficult to train effective CNNs.
- Large datasets (millions of images) are often needed for high accuracy.

#### Techniques for Small Datasets:

1. **Data Augmentation**:
   - Increase dataset size by creating variations of existing images.
   - Techniques include rotation, flipping, shifting, zooming, etc.
   - Example: Turn 10,000 images into 40,000 with augmentations.

2. **Using Pre-trained Models**:
   - Utilize models trained on large datasets (e.g., 1.4 million images).
   - Fine-tune the last few layers for specific tasks.
   - Example: Use Google's pre-trained models as a base.

### Data Augmentation Example

- **Image Data Generator**:
  - Use `ImageDataGenerator` from Keras for augmentation.
  - Specify parameters like rotation range, shifts, shear, zoom, etc.
  - Augment images and save them with prefixes (e.g., test1, test2).

- **Implementation**:
  - Convert images to NumPy arrays.
  - Reshape and augment images using a loop.
  - Display augmented images to visualize transformations.

### Pre-trained Models

- **Concept**:
  - Leverage existing models trained on extensive datasets.
  - Fine-tune for specific applications.

- **Benefits**:
  - Saves time and computational resources.
  - Provides a strong starting point for model development.

### Conclusion

- **CNNs**: Effective for feature extraction and classification.
- **Training**: Requires careful consideration of dataset size and augmentation techniques.
- **Pre-trained Models**: Offer a practical solution for small datasets.

These notes provide a comprehensive overview of building and training CNNs, especially when dealing with limited data. Understanding these concepts is crucial for developing efficient and accurate computer vision models.


## Chunk 5 (4:23:10 - 4:33:10)

# Deep Computer Vision: Convolutional Neural Networks

## Using Pre-trained Models

### Key Concepts

- **Pre-trained Models**: Models that have been previously trained on a large dataset and can be used as a starting point for a new model.
- **Fine-tuning**: Modifying the top layers of a pre-trained model to adapt it to a specific problem.

### Steps to Use a Pre-trained Model

1. **Utilize the Base Model**: 
   - The initial layers of a pre-trained model are adept at identifying general features such as edges.
   - These layers are not modified as they generalize well across different datasets.

2. **Modify the Top Layers**:
   - Adjust the final layers to suit the specific classification task (e.g., classifying cats vs. dogs).

3. **Fine-tuning**:
   - This involves using the pre-trained model's generalization capabilities and adding custom layers for specific tasks.

### Example: Classifying Cats vs. Dogs

- **Dataset Loading**:
  - Use TensorFlow Datasets (TFDS) to load the dataset.
  - Split the dataset into 80% training, 10% validation, and 10% testing.

- **Data Preprocessing**:
  - Images are resized to a uniform dimension (160x160) to ensure consistency.
  - Convert pixel values to `float32` and normalize them by dividing by 127.5 and subtracting 1.

### Choosing a Pre-trained Model

- **MobileNet V2**:
  - A lightweight model from Google built into TensorFlow.
  - Suitable for mobile and edge devices due to its efficiency.

- **Model Configuration**:
  - **Input Shape**: Set to 160x160x3 to match the resized images.
  - **Include Top**: Set to `False` to exclude the pre-trained classifier.
  - **Weights**: Load from ImageNet, a large dataset used for training general-purpose models.

### Model Architecture

- **Base Model**:
  - The architecture is complex and designed by experts, making it ideal for generalization.
  - Outputs a tensor of shape 32x5x5x1280.

- **Freezing the Base Model**:
  - Prevents retraining of the base model's weights by setting `trainable` to `False`.

### Adding a Custom Classifier

1. **Global Average Pooling**:
   - Reduces the spatial dimensions of the feature maps to a single vector by averaging.

2. **Prediction Layer**:
   - A dense layer with a single node for binary classification (cats vs. dogs).

### Summary

- **Pre-trained Models**: Effective for leveraging existing knowledge and reducing training time.
- **Fine-tuning**: Allows customization for specific tasks while maintaining the robustness of the base model.
- **MobileNet V2**: Chosen for its efficiency and integration with TensorFlow, making it a practical choice for this classification task.

This approach highlights the power of transfer learning in deep learning, enabling efficient and effective model development by building on existing, well-trained architectures.


## Chunk 6 (4:33:10 - 4:40:44)

# Deep Computer Vision - Convolutional Neural Networks

## Model Construction and Training

### Model Architecture
- **Base Layer**: MobileNet V2
  - Used as the foundational model.
  - Output shape is processed by the global average pooling layer.
  
- **Global Average Pooling Layer**
  - Flattens the output and computes the average.
  
- **Dense Layer**
  - Contains a single neuron for output.
  - Total parameters: 2.25 million, with only 1,281 trainable.
  - Trainable parameters include 1,280 weights and 1 bias.

### Training Process
- **Training Strategy**
  - Only the weights and biases of the top layers are trained.
  - Base layer remains untrained to leverage pre-existing features.
  
- **Learning Rate**
  - Set to a low value to avoid major changes to the weights and biases.
  
- **Loss Function**
  - **Binary Cross Entropy**: Used for two-class classification (cats vs. dogs).

### Model Evaluation
- **Initial Evaluation**
  - Conducted before training to assess the model's baseline performance.
  - Achieved an accuracy of 56% with random weights, indicating near-random guessing.

- **Training Outcome**
  - Post-training accuracy reaches approximately 92-93%.
  - Demonstrates the effectiveness of using a pre-trained base model with a custom classifier.

### Model Saving and Loading
- **Saving the Model**
  - Use `model.save('model_name.h5')` to save the trained model.
  
- **Loading the Model**
  - Use `model.load('model_name.h5')` to load the saved model.
  - Useful to avoid retraining, especially for large models.

### Making Predictions
- **Prediction Syntax**
  - Use `model.predict()` to make predictions.
  - Input should match the expected format (e.g., 160x160x3 for images).

## Object Detection with TensorFlow

### Introduction to Object Detection
- TensorFlow provides an API for object detection.
- Capable of returning confidence scores for detected objects.

### Resources
- **GitHub Resource**: TensorFlow's object detection API.
- **Facial Recognition Module**: A Python module for facial detection and recognition using CNNs.

## Conclusion

- **Convolutional Neural Networks (CNNs)**
  - Effective for image classification tasks.
  - Can be enhanced by using pre-trained models with custom classifiers.
  
- **Further Exploration**
  - Encouraged to explore additional resources and tutorials to deepen understanding.
  - Experiment with different base layers and configurations for various problems.

## Next Module: Recurrent Neural Networks (RNNs)

- Transitioning to RNNs, which offer interesting applications and challenges.
