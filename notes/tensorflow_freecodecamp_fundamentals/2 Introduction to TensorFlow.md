# Module 2: Introduction to TensorFlow


## Chunk 1 (30:08 - 40:08)

# Introduction to TensorFlow

## Overview of Module Two

In this module, we will cover the following key topics:

- General introduction to TensorFlow
- Understanding tensors, shapes, and data representation
- How TensorFlow operates at a lower level

### Importance of Lower-Level Understanding

- **Why it matters**: 
  - Enhances the ability to tweak models effectively
  - Provides a deeper understanding of TensorFlow operations

## What is TensorFlow?

- **Definition**: TensorFlow is an open-source machine learning library maintained by Google.
- **Purpose**: Facilitates the creation of machine learning models and neural networks without requiring an extensive math background.

### Mathematical Foundations

- **Basic Requirements**: 
  - Fundamental calculus
  - Basic linear algebra
- **Advanced Concepts**:
  - Gradient descent
  - Regression techniques
  - Classification

### Benefits

- Simplifies complex mathematical operations
- Provides tools to create models with a basic understanding of underlying math

## Google Collaboratory

- **Definition**: A cloud-based platform for running Jupyter Notebooks.
- **Features**:
  - Allows coding and note-taking in a single interface
  - No installation required on local machines
  - Accessible from any device, including mobile phones

### How to Use Google Collaboratory

1. **Creating a Notebook**:
   - Search for "Google Collaboratory" and create a new notebook.
   - The file format is `.ipynb` (IPython Notebook).

2. **Writing Code and Text**:
   - Supports both code and markdown text.
   - Use markdown for organizing notes and code for executing tasks.

3. **Executing Code**:
   - Run code blocks in any sequence.
   - Code blocks can access variables and functions defined in other blocks.

4. **Importing Modules**:
   - Modules like NumPy can be imported without installation.
   - TensorFlow installation instructions are available in the provided notebook.

## Key TensorFlow Concepts

### Graphs and Sessions

- **Graphs**:
  - Represent partial computations.
  - Define computations without immediate evaluation.
  - Example: A variable defined as the sum of two other variables is added to the graph but not computed until needed.

- **Sessions**:
  - Execute parts or the entire graph.
  - Start from the lowest level where no dependencies exist and proceed through computations.

### Practical Applications

- **Capabilities**:
  - Image classification
  - Data clustering
  - Regression
  - Reinforcement learning
  - Natural language processing

- **Advantage**: Automates complex math operations, allowing focus on model creation.

## Getting Started with TensorFlow

1. **Importing and Installing**:
   - Use Google Collaboratory to avoid local installation.
   - TensorFlow can be installed locally using `pip install tensorflow` or `pip install tensorflow-gpu` for GPU support.

2. **Using Collaboratory**:
   - Connects to Google servers with pre-installed modules.
   - Provides performance benefits for machines with limited resources.

3. **Runtime Management**:
   - Restart runtime to clear outputs and reset the environment.

By understanding these foundational concepts and utilizing tools like Google Collaboratory, you can effectively leverage TensorFlow for various machine learning tasks.


## Chunk 2 (40:08 - 50:08)

# Introduction to TensorFlow: Google Collaboratory and Tensors

## Google Collaboratory Basics

- **Running Code Blocks**: In Google Collaboratory, you can run specific code blocks without executing the entire script.
  - Useful for testing minor changes without rerunning everything.
  - To restart and run all code blocks sequentially, use the "Restart runtime" and "Restart and run all" options.

- **Similarity to Jupyter Notebooks**: Google Collaboratory is similar to Jupyter Notebooks, allowing easy code execution and experimentation.

## Importing TensorFlow in Google Collaboratory

### Steps to Import TensorFlow

1. **Specify TensorFlow Version**: 
   - Use `%tensorflow_version 2.x` to specify TensorFlow 2.x in Collaboratory.
   - This step is unnecessary on local machines with text editors.

2. **Import TensorFlow**:
   - Use `import tensorflow as tf` to import TensorFlow with the alias `tf`.
   - Ensure TensorFlow is installed on local machines.

3. **Check TensorFlow Version**:
   - Print the TensorFlow version using `print(tf.__version__)`.
   - Confirm that version 2.x is being used, as some features are specific to TensorFlow 2.x.

### Troubleshooting

- If TensorFlow 1.x is loaded, restart the runtime to switch to TensorFlow 2.x.

## Understanding Tensors

### Definition

- **Tensor**: A generalization of vectors and matrices to higher dimensions.
  - **Vector**: A data point with potentially multiple dimensions (e.g., x, y in 2D space).
  - **Tensor**: Can have multiple dimensions, represented as n-dimensional arrays in TensorFlow.

### Importance in TensorFlow

- Tensors are the primary objects manipulated in TensorFlow, representing computations and data flow.

### Tensor Attributes

1. **Data Type**: The type of data stored in the tensor.
   - Common types: `float32`, `int32`, `string`.

2. **Shape**: The dimensionality of the tensor.
   - Describes how data is organized within the tensor.

### Creating Tensors

- **Scalar Tensor**: A single value, e.g., `tf.Variable(5, dtype=tf.int32)`.
- **Vector Tensor**: A one-dimensional array, e.g., `tf.Variable([1, 2, 3], dtype=tf.int32)`.
- **Matrix Tensor**: A two-dimensional array, e.g., `tf.Variable([[1, 2], [3, 4]], dtype=tf.int32)`.

### Rank of Tensors

- **Rank**: The number of dimensions in a tensor.
  - **Rank 0**: Scalar (single value).
  - **Rank 1**: Vector (one-dimensional array).
  - **Rank 2**: Matrix (two-dimensional array).
  - Determined by the deepest level of nested lists.

- **Determining Rank**: Use `tf.rank(tensor)` to find the rank of a tensor.

### Examples

- **Rank 0 Tensor**: `tf.Variable(3.14, dtype=tf.float32)` results in a scalar.
- **Rank 1 Tensor**: `tf.Variable([1, 2, 3], dtype=tf.int32)` results in a vector.
- **Rank 2 Tensor**: `tf.Variable([[1, 2], [3, 4]], dtype=tf.int32)` results in a matrix.

## Summary

- Google Collaboratory allows efficient code execution and experimentation with TensorFlow.
- Tensors are fundamental to TensorFlow, representing data and computations.
- Understanding tensor attributes like data type, shape, and rank is crucial for effective TensorFlow programming.


## Chunk 3 (50:08 - 1:00:00)

# Introduction to TensorFlow: Understanding Tensors

## Tensor Shapes and Ranks

### Key Concepts
- **Shape**: Indicates the number of items in each dimension of a tensor.
  - Example: A tensor with a shape of `(2, 2)` has two elements in both the first and second dimensions.
- **Rank**: The number of dimensions in a tensor.
  - Rank 1 tensor: Single list of elements.
  - Rank 2 tensor: A list of lists (matrix).

### Examples
- **Rank 1 Tensor**: `[1, 2, 3]` has a shape of `(3,)`.
- **Rank 2 Tensor**: `[[1, 2], [3, 4]]` has a shape of `(2, 2)`.

### Modifying Shapes
- Tensors must have uniform dimensions (e.g., each list must have the same number of elements).
- Adding elements or lists changes the shape:
  - Adding a third element to each list in a rank 2 tensor changes its shape from `(2, 2)` to `(2, 3)`.
  - Adding another list changes the shape to `(3, 3)`.

## Reshaping Tensors

### Key Concepts
- **Reshape**: Changing the shape of a tensor while maintaining the same number of elements.
- **Flattening**: Converting a multi-dimensional tensor into a rank 1 tensor.

### Examples
- **Reshape a Tensor**: 
  - Original tensor with shape `(1, 2, 3)` can be reshaped to `(2, 3, 1)`.
  - Use `tf.reshape(tensor, new_shape)` to reshape tensors.
- **Negative Dimension**: Using `-1` in a shape allows TensorFlow to infer the dimension size.
  - Example: Reshaping `(3, -1)` infers the second dimension based on the total number of elements.

## Types of Tensors

### Key Types
- **Variable**: Mutable tensor whose value can change during execution.
- **Constant**: Immutable tensor with a fixed value.
- **Placeholder**: Used for input data during execution.
- **SparseTensor**: Efficient storage for tensors with a lot of zero values.

### Important Notes
- All tensors except **Variable** are immutable.
- **Variable** tensors are used when values need to change during execution.

## Evaluating Tensors

### Key Concepts
- **Session**: Required to evaluate tensors in TensorFlow.
- **Evaluation**: Necessary to compute and retrieve the value of a tensor.

### Example Code
```python
with tf.Session() as sess:
    result = tensor.eval()
```

### Usage
- Use sessions to evaluate tensors when necessary in your code.

## Practical Examples

### Creating and Reshaping Tensors
- **Creating a Tensor**: Use `tf.ones(shape)` or `tf.zeros(shape)` to create tensors filled with ones or zeros.
- **Example**: Create a tensor with shape `(5, 5, 5, 5)` and reshape it to `(625,)` to flatten it.

### Reshaping with Inference
- Reshape a tensor to `(125, -1)` to let TensorFlow infer the last dimension.

### Code Snippet
```python
t = tf.ones((5, 5, 5, 5))
t = tf.reshape(t, (625,))
```

## Conclusion

Understanding tensor shapes, ranks, and how to reshape them is crucial for working with TensorFlow. This knowledge allows for efficient manipulation and evaluation of data, which is foundational for building more complex models and algorithms. As you progress, these concepts will be applied in more advanced coding scenarios.
