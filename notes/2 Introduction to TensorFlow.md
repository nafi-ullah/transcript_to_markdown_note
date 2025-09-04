# Module 2: Introduction to TensorFlow


## Chunk 1 (30:08 - 40:08)

## Introduction to TensorFlow

### Overview
- **Module Objective**: General introduction to TensorFlow, understanding tensors, shapes, data representation, and lower-level workings of TensorFlow.
- Importance of fundamental knowledge for easier model tweaking and understanding.

### What is TensorFlow?
- **Definition**: Open-source Machine Learning Library maintained by Google.
- **Functionality**: Enables creation of machine learning models and neural networks without complex math background.
- **Math Requirements**: Basic understanding of calculus, linear algebra, gradient descent, regression techniques, and classification.
- **Tool Usage**: TensorFlow provides tools to create models without needing in-depth math knowledge.

### TensorFlow Capabilities
- **Applications**: Image classification, data clustering, regression, reinforcement learning, natural language processing, and various machine learning tasks.
- **Functionality**: Provides tools to perform complex math operations required for machine learning tasks.

### TensorFlow Working Mechanism
- **Components**: Graphs and Sessions.
- **Graphs**: Represent partial computations without evaluating them, defining relationships between computations.
- **Sessions**: Execute parts or entire graph, starting from independent computations to interlinked ones.

### Google Collaboratory
- **Definition**: Free Jupyter Notebook in the cloud for coding and note-taking.
- **Advantages**: No need to install modules, automatic connection to Google servers for execution.
- **Features**: Markdown support for text, code blocks for running code, ability to import modules without installation.
- **Usage**: Allows running code on Google servers, accessing hardware for machine learning tasks.
- **Performance Benefits**: Improved performance on lower-end machines, access to RAM and disk space information.
- **Runtime Tab**: Allows restarting runtime to clear output and restart.

### Coding in Google Collaboratory
- **Installation**: No need to install TensorFlow, modules pre-installed on Google servers.
- **Importing Modules**: Can directly import modules like NumPy without installation.
- **Performance**: Utilizes Google servers for machine learning tasks, provides performance benefits.
- **Local Runtime Connection**: Option to connect to local machine for runtime execution.

These notes provide a comprehensive overview of TensorFlow, its functionalities, working mechanism, Google Collaboratory usage, and coding capabilities.


## Chunk 2 (40:08 - 50:08)

## Introduction to TensorFlow

### Google Collaboratory
- **Advantages**:
  - Ability to run specific code blocks without executing the entire code.
  - Easy to restart runtime to clear everything and rerun all code blocks sequentially.
- **Steps**:
  1. Click "Restart runtime" to clear everything.
  2. Click "Restart and run all" to restart runtime and run every code block in order.

### Importing TensorFlow in Google Collaboratory
- **Steps**:
  1. Define TensorFlow version: `%tensorflow_version 2.x`.
  2. Import TensorFlow as an alias `TF`.
- **Version Check**:
  - Ensure TensorFlow version 2.x is loaded.
  - Print TensorFlow version to confirm correct version.

### Tensors
- **Definition**:
  - Generalization of vectors and matrices to higher dimensions.
- **Key Points**:
  - Tensors are primary objects in TensorFlow used for computations.
  - Each tensor represents a partially defined computation that produces a value.
- **Data Type and Shape**:
  - **Data Type**: Information stored in the tensor (e.g., float, string).
  - **Shape**: Representation of the tensor in terms of dimensions.
- **Creating Tensors**:
  - Examples:
    1. String tensor: `TF.strings`.
    2. Number tensor: `TF.int16`.
    3. Floating-point tensor.
- **Rank/Degree of Tensors**:
  - **Rank 0 (Scalar)**: Represents a single value with zero dimensionality.
  - **Rank 1 (Vector)**: Represents a list or array with one dimension.
  - **Rank 2 (Matrix)**: Represents a list of lists or arrays with two dimensions.
- **Determining Rank**:
  - Use `TF.rank` method to determine the rank of a tensor.
- **Examples**:
  - Rank 0 tensor: Scalar.
  - Rank 1 tensor: Vector.
  - Rank 2 tensor: Matrix.

### Conclusion
- Understanding tensors is crucial in TensorFlow as they represent computations and values.
- Tensors have data types, shapes, and ranks that define their properties.
- Practice creating and analyzing tensors to grasp their significance in TensorFlow.


## Chunk 3 (50:08 - 1:00:00)

## Introduction to TensorFlow

### Understanding Tensor Shapes

- **Shape of a Tensor**:
  - Indicates how many items are in each dimension.
  - **Rank 2 Tensor**:
    - Shape represented as 2x2 indicates 2 elements in the first dimension and 2 elements in the second dimension.
  - **Rank 1 Tensor**:
    - Shape represented as 3 indicates a single dimension with 3 elements.
  - **Reshaping Tensors**:
    - Changing the shape of tensors while maintaining the same number of elements.
    - Example: Reshaping a 2x3 tensor to a 3x2 tensor.

### Types of Tensors

- **Variable**:
  - Can change during execution.
- **Constant**:
  - Immutable, value remains constant.
- **Placeholder**:
  - Tensors used for feeding input values.
- **SparseTensor**:
  - Efficiently represent tensors with a large number of elements.

### Evaluating Tensors

- **Session**:
  - Required to evaluate tensors in TensorFlow.
  - Use `tf.Session()` to evaluate tensors and perform operations.
- **Tensor Evaluation**:
  - Use `tensor_name.eval()` within a session to evaluate a tensor and obtain its value.

### Examples of Reshaping Tensors

- **Reshaping Process**:
  - Use `tf.reshape()` to change the shape of a tensor.
- **Example**:
  - Reshaping a 5x5x5 tensor to a flattened 1D tensor with 625 elements.
- **Inference**:
  - Use `-1` in reshaping to let TensorFlow infer the shape based on the number of elements.

### Summary

- **Key Concepts**:
  - Tensor shapes, types, reshaping, and evaluation.
- **Practical Application**:
  - Understanding tensor manipulation for data processing and model building in TensorFlow.
