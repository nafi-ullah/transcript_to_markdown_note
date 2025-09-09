# Module 6: Natural Language Processing with RNNs


## Chunk 1 (4:40:44 - 4:50:44)

# Natural Language Processing with Recurrent Neural Networks

## Introduction to Natural Language Processing (NLP)

- **Natural Language Processing (NLP)**: A field within computing and machine learning focused on understanding human languages, which are referred to as "natural" because they are not computer or programming languages.
  - Computers inherently struggle with understanding textual information and human languages, necessitating the development of NLP.
- **Applications of NLP**:
  - Spellcheck
  - Autocomplete
  - Voice assistants
  - Language translation
  - Chatbots
  - Any task involving textual data (words, sentences, paragraphs)

## Recurrent Neural Networks (RNNs)

- **Recurrent Neural Networks (RNNs)**: A type of neural network particularly effective at classifying and understanding textual data.
  - RNNs are complex and involve intricate processes.
  - The focus will be on understanding the "why" and "when" of using RNNs rather than the detailed mathematical underpinnings.
  - For mathematical details, additional resources are recommended.

## Key Applications in This Module

1. **Sentiment Analysis**:
   - Task: Determine if movie reviews are positive or negative.
   - *Sentiment Analysis*: Assessing the positivity or negativity of a sentence or text.

2. **Character/Text Generation**:
   - Task: Generate the next character in a sequence to create an entire play.
   - Example: Training a model on "Romeo and Juliet" to generate text based on a given prompt.

## Textual Data vs. Numeric Data

- **Challenge**: Converting textual data into numeric data that neural networks can process.
- **Solution**: Employ various encoding and preprocessing techniques.

## Bag of Words Technique

- **Bag of Words**: A method to convert text into numeric data by creating a dictionary of unique words from the dataset.
  - Each unique word is assigned an integer.
  - The technique keeps track of word frequency but not the order of words.
  - **Limitations**:
    - Loses the order and context of words.
    - Not suitable for complex tasks where word order affects meaning.

### Example of Bag of Words

- **Vocabulary Example**: Words like "I", "am", "Tim", "day" are assigned integers.
- **Sentence Encoding**: 
  - Sentence: "I am Tim day day"
  - Encoding: [0, 1, 2, 3, 3] (based on assigned integers)
- **Flaw**: Sentences with the same words but different meanings are encoded identically.

### Illustrative Example

- **Sentences**:
  - "I thought the movie was going to be bad, but it was actually amazing."
  - "I thought the movie was going to be amazing, but it was actually bad."
- **Issue**: Both sentences are encoded the same, losing the distinct meanings due to word order.

## Conclusion

- **Bag of Words** is a foundational technique but has significant limitations for complex NLP tasks.
- Understanding these limitations is crucial for selecting appropriate methods for text processing in NLP applications.


## Chunk 2 (4:50:44 - 5:00:44)

## Natural Language Processing with RNNs

### Encoding Methods

#### Bag of Words
- **Definition**: A method of encoding text data by counting the frequency of words in a document.
- **Characteristics**:
  - Ignores the sequence or order of words.
  - Represents text as a "bag" of words with their respective counts.
- **Example**:
  - Sentence: "Tim is here"
  - Encoding: "Tim" = 3, "is" = 2, "here" = 1

#### Integer Encoding
- **Concept**: Assigns a unique integer to each word and maintains the order of words.
- **Example**:
  - Sentence: "Tim is here"
  - Encoding: "Tim" = 0, "is" = 1, "here" = 2
  - Sequence: 0, 1, 2
- **Limitations**:
  - Large vocabularies require many unique mappings.
  - Arbitrary mappings can misrepresent word similarities (e.g., "happy" = 1, "good" = 100,000).

### Word Embeddings

#### Introduction
- **Purpose**: To represent words in a way that captures their meanings and relationships.
- **Method**: Translates words into vectors in a multi-dimensional space.
- **Benefits**:
  - Similar words have similar vector representations.
  - Captures semantic relationships between words.

#### How Word Embeddings Work
- **Vector Representation**:
  - Each word is represented by a vector with multiple dimensions (e.g., 64 or 128).
  - Vectors for similar words point in similar directions.
- **Example**:
  - "Good" and "Happy" have vectors pointing in similar directions.
  - "Bad" has a vector pointing in a different direction.
- **Training**:
  - Word embeddings are learned during model training.
  - The model adjusts vectors to reflect word contexts and meanings.

#### Practical Use
- **Pre-trained Embeddings**: Can use pre-trained word embeddings for efficiency and improved performance.
- **Importance**: Proper encoding of text data is crucial for effective neural network processing.

### Recurrent Neural Networks (RNNs)

#### Overview
- **Purpose**: Designed to process sequential data, such as text.
- **Key Feature**: Contains an internal loop allowing it to maintain memory of previous inputs.

#### Differences from Other Neural Networks
- **Comparison**:
  - Unlike dense or convolutional networks, RNNs process data in time steps.
  - Maintains an internal state to remember previous inputs.

#### Functionality
- **Internal Memory**: Allows the network to consider the context of previous words when processing new inputs.
- **Sequential Processing**: Processes input data one step at a time, maintaining context throughout.

---

These notes provide a structured overview of encoding methods and the role of word embeddings and RNNs in natural language processing. Understanding these concepts is critical for effectively working with text data in machine learning models.


## Chunk 3 (5:00:44 - 5:10:44)

## Natural Language Processing with RNNs

### Feed Forward Neural Networks vs. Recurrent Neural Networks

- **Feed Forward Neural Networks (FFNNs):**
  - Data is fed all at once and processed from left to right.
  - Typically involves passing data through layers such as convolutional and dense layers.
  - All information is processed simultaneously.

- **Recurrent Neural Networks (RNNs):**
  - Data is processed one word at a time, using a loop structure.
  - Utilizes an internal memory state to keep track of information.
  - Mimics human reading by processing text sequentially, word by word.
  - Builds understanding incrementally, using previous words to inform the meaning of the current word.

### Understanding RNNs

- **Sequential Processing:**
  - RNNs read text one word at a time, developing an understanding gradually.
  - Each word's processing is informed by the words that came before it.
  - This method helps in understanding the context and meaning of the text.

- **RNN Structure:**
  - **Input (x):** The word at time step \( t \) (denoted as \( x_t \)).
  - **Output (h):** The understanding of the text at time \( t \) (denoted as \( h_t \)).
  - Example: For a text of length 4, the first input at time 0 is processed to form an initial understanding.

### Example: Sentence Processing with RNNs

- **Sentence:** "Hi, I am Tim"
  - Each word is converted into a vector (numeric representation).
  - The RNN processes each word sequentially:
    1. **Time Step 0:** Process "Hi" with no prior context.
    2. **Time Step 1:** Process "I" using the output from "Hi".
    3. **Time Step 2:** Process "am" using the combined understanding of "Hi" and "I".
    4. **Time Step 3:** Process "Tim" using the accumulated understanding.
  - The final output \( h_3 \) represents the understanding of the entire sentence.

### Challenges with Simple RNNs

- **Long Sequences:**
  - As sequence length increases, early information can become diluted.
  - The model may struggle to retain information from the beginning of long sequences.

### Advanced RNN Layer: Long Short Term Memory (LSTM)

- **Introduction to LSTM:**
  - **LSTM** stands for **Long Short Term Memory**.
  - Designed to address the limitations of simple RNNs with long sequences.
  - Adds a mechanism to retain information from any previous state, not just the immediate past.

- **Internal State Management:**
  - LSTMs maintain an internal state that can access outputs from any previous time step.
  - This allows the model to remember important information over long sequences.

### Summary

- RNNs are powerful for sequential data processing, allowing for context-aware understanding of text.
- Simple RNNs can struggle with long sequences due to information loss over time.
- LSTMs enhance RNNs by providing a more robust memory mechanism, improving performance on long sequences.

These notes provide a comprehensive overview of how RNNs function, their advantages over traditional neural networks, and the enhancements offered by LSTMs.


## Chunk 4 (5:10:44 - 5:20:44)

# Natural Language Processing with RNNs

## Long Short-Term Memory (LSTM)

- **Concept**: 
  - Unlike simple RNNs that only keep track of the previous output, LSTMs maintain a sequence of outputs using a "conveyor belt" mechanism.
  - This allows the model to access any previous state, which is beneficial for long sequences where early information might be forgotten.

- **Utility**: 
  - Enables the model to reference any part of the sequence, aiding in understanding the context from both the beginning and end of a text.
  - Particularly useful for tasks requiring long-term dependencies.

- **Mathematical Definitions**: 
  - For those interested in the mathematical underpinnings, additional resources are suggested.

## Sentiment Analysis Example

### Overview

- **Objective**: Determine if movie reviews are positive or negative.
- **Dataset**: Movie review dataset from Keras with 25,000 pre-processed and labeled reviews.

### Data Preprocessing

- **Encoding**:
  - Each word in the dataset is encoded as an integer.
  - Encoding reflects word frequency, with lower integers representing more common words.
  - Vocabulary size: 88,584 unique words.

- **Handling Variable Lengths**:
  - Reviews have variable lengths; neural networks require uniform input sizes.
  - **Padding**:
    - Reviews longer than 250 words are trimmed.
    - Reviews shorter than 250 words are padded with zeros on the left to reach 250 words.

### Model Construction

- **Architecture**:
  - **Embedding Layer**: Converts integer-encoded words into dense vectors of 32 dimensions.
  - **LSTM Layer**: Processes sequences with long-term dependencies.
  - **Dense Layer**: Uses a sigmoid activation function to output a sentiment score between 0 and 1.

- **Purpose of Layers**:
  - **Embedding Layer**: Provides meaningful vector representations of words.
  - **LSTM Layer**: Captures temporal dependencies in the sequence.
  - **Dense Layer**: Outputs a probability for sentiment classification.

### Training the Model

- **Compilation**:
  - **Loss Function**: Binary cross-entropy, suitable for binary classification tasks.
  - **Optimizer**: RMSprop, although Adam can also be used.
  - **Metrics**: Accuracy is used to evaluate model performance.

- **Training Tips**:
  - Use a GPU to accelerate training (10-20x speedup).
  - Adjust runtime settings to enable GPU usage.

### Key Points

- **Embedding Layer Parameters**: Largest number of parameters due to the complexity of transforming integers into meaningful vectors.
- **LSTM Layer**: Handles sequence processing and outputs to the dense layer.
- **Dense Layer**: Final layer that predicts sentiment based on processed input.

### Additional Notes

- **Model Summary**: Provides an overview of the model architecture and parameter count.
- **Training Considerations**: Choice of optimizer is flexible; Adam is a common default.

These notes provide a structured overview of using LSTMs for sentiment analysis, detailing the preprocessing, model construction, and training process.


## Chunk 5 (5:20:44 - 5:30:44)

## Natural Language Processing with RNNs

### Model Training and Evaluation

- **Model Training:**
  - The model is trained using a dataset with a validation split of 20% (`0.2`), meaning 20% of the training data is used for validation.
  - After training, the model achieves:
    - **Evaluation Accuracy:** Approximately 88%
    - **Overfit Accuracy:** Around 97-98%
  - **Observation:** The model is overfitting, indicating insufficient training data. After one epoch, the validation accuracy stagnates, suggesting a need for model improvement.

- **Test Evaluation:**
  - The model is evaluated on test data, resulting in an accuracy of about 85.5%. This is considered decent given the simplicity of the code implemented.

### Making Predictions

- **Preprocessing for Predictions:**
  - Preprocessing of input data must match the training data preprocessing to ensure accurate predictions.
  - A function is created to encode text into preprocessed integers, using a lookup table from the IMDb dataset.

- **Encoding Function:**
  - Uses `keras.preprocessing.text.text_to_word_sequence` to tokenize text.
  - Maps tokens to integers using a vocabulary of 88,000 words.
  - Pads sequences to ensure consistent input length.

- **Decoding Function:**
  - Reverses the word index to convert integers back to words.
  - Constructs a text string from integer sequences, excluding padding.

### Prediction Function

- **Functionality:**
  - Encodes movie reviews using the encoding function.
  - Prepares input as a NumPy array of zeros with shape `(1, 250)`, matching the model's expected input shape.
  - Uses `model.predict` to obtain predictions, focusing on the first entry for single predictions.

- **Example Predictions:**
  - Positive Review: "That movie was so awesome, I really loved it..." predicted as 72% positive.
  - Negative Review: "That movie sucked, I hated it..." predicted as 23% positive.
  - **Note:** Small changes in the review text can significantly affect prediction outcomes.

### Experimenting with Text Changes

- **Impact of Word Changes:**
  - Removing or adding specific words (e.g., "awesome," "great") can alter prediction percentages.
  - Shorter reviews are more sensitive to word changes compared to longer reviews.

### Next Steps: Recurrent Neural Network Play Generator

- **Objective:**
  - Create a model to predict the next character in a sequence.
  - Train the model on sequences from "Romeo and Juliet."
  - Use a starting prompt to generate text, allowing the model to predict subsequent characters.

This section outlines the process of training and evaluating a model for sentiment analysis, as well as preparing it for making predictions on new data. The subsequent part of the module will focus on generating text using recurrent neural networks.


## Chunk 6 (5:30:44 - 5:40:44)

# Natural Language Processing with RNNs

## Generating Text with RNNs

### Overview
- **Objective**: Use a neural network to predict and generate text, character by character.
- **Process**: 
  1. Pass a sequence to the model.
  2. Predict the next character.
  3. Use the predicted character as the next input.
  4. Repeat to generate a full text sequence.

### Implementation Steps

#### 1. Import Necessary Libraries
- **Libraries**:
  - `Keras` for neural network operations.
  - `TensorFlow` for backend support.
  - `NumPy` for numerical operations.
  - `os` for operating system interactions.

#### 2. Load Text Data
- **Dataset**: Use the text of "Romeo and Juliet" or any other text file.
- **Loading Method**:
  - Use Keras utilities to download and save the text file as `Shakespeare.txt`.
  - Alternatively, upload your own `.txt` file via Google Collaboratory.

#### 3. Read and Process Text
- **File Handling**:
  - Open the text file in read bytes mode (`rb`).
  - Read and decode the text into UTF-8 format.
  - Example: The text length is approximately 1.1 million characters.
- **Text Preview**:
  - Display the first 250 characters to understand the format (e.g., speaker names followed by lines).

#### 4. Text Encoding
- **Purpose**: Convert text characters into integers for model processing.
- **Process**:
  - **Character Set**: Identify unique characters in the text.
  - **Mapping**: Create a mapping from characters to indices.
  - **Reverse Mapping**: Create a reverse mapping from indices to characters.

#### 5. Convert Text to Integer Representation
- **Function**: `text_to_int(text)`
  - Converts each character in the text to its integer representation.
  - Uses NumPy arrays for efficient processing.
- **Example**:
  - Text "First Citizen" is encoded as `[18, 40, 75, 65, 57, 50, 81]`.

#### 6. Convert Integers Back to Text
- **Function**: `int_to_text(integers)`
  - Converts integer sequences back to text.
  - Ensures compatibility with different input types by using NumPy arrays.

### Creating Training Examples

#### 1. Define Sequence Length
- **Sequence Length**: 100 characters.
- **Training Example**:
  - **Input**: A sequence of 100 characters.
  - **Output**: The same sequence shifted by one character (e.g., input "hell" -> output "ello").

#### 2. Prepare Data for Training
- **Batching**:
  - Use TensorFlow's `tf.data.Dataset` to create batches of sequences.
  - Sequence length for batching: 101 (100 input + 1 output).
  - Drop any remainder sequences that don't fit the batch size.

#### 3. Create Input and Target Sequences
- **Function**: `split_input_target(sequence)`
  - Splits each sequence into input and target parts.
  - Input is the sequence itself, target is the sequence shifted by one character.
- **Mapping**:
  - Apply the `split_input_target` function to all sequences using `map`.

### Summary
- **Goal**: Efficiently prepare text data for training an RNN to generate text.
- **Key Steps**:
  - Import necessary libraries.
  - Load and preprocess text data.
  - Encode text into integers.
  - Create training examples with defined input-output sequences.
- **Outcome**: A dataset ready for training a character-level RNN model to predict and generate text.


## Chunk 7 (5:40:44 - 5:50:44)

# Natural Language Processing with RNNs

## Chunk 7: Training Batches and Model Building

### Key Concepts

- **Training Batches**: Groups of data used to train the model in iterations.
- **Batch Size**: Number of training examples utilized in one iteration.
- **Embedding Dimension**: Size of the vector space in which words will be embedded.
- **RNN Units**: Number of units in a Recurrent Neural Network layer.
- **Buffer Size**: Size of the buffer used for shuffling the dataset.

### Training Batches

1. **Batch Size**: 
   - Set to 64. This means each batch contains 64 sequences.
   
2. **Vocabulary Size**:
   - Determined by the number of unique characters in the text.

3. **Embedding Dimension**:
   - Set to 256. This determines the size of the embedding vectors.

4. **RNN Units**:
   - Set to 1024. This parameter is crucial for defining the complexity of the RNN layer.

5. **Buffer Size**:
   - Set to 10,000 for shuffling the dataset to ensure randomness.

### Data Preparation

- **Shuffling and Batching**:
  - The dataset is shuffled to prevent the model from learning the order of sequences.
  - Batches are created with the specified batch size, and any remainder is dropped.

### Model Building

#### Function: `build_model`

- **Purpose**: To create a model that can be trained on batches of data and later used for predictions on single sequences.

- **Parameters**:
  - **Vocabulary Size**: Number of unique characters.
  - **Embedding Dimension**: 256.
  - **Batch Size**: Initially set to 64, later adjusted to 1 for predictions.
  
- **LSTM Layer**:
  - **RNN Units**: 1024.
  - **Return Sequences**: Set to `True` to get outputs at each time step.
  - **Stateful**: Not discussed in detail, but important for sequence continuity.
  - **Recurrent Initializer**: Default values recommended by TensorFlow.

- **Dense Layer**:
  - Contains nodes equal to the vocabulary size to predict the probability distribution of the next character.

### Model Summary

- **Layers**:
  - **Embedding Layer**: Maps input sequences to a dense vector space.
  - **LSTM Layer**: Processes sequences and captures temporal dependencies.
  - **Dense Layer**: Outputs probabilities for each character in the vocabulary.

- **Output Dimensions**:
  - **Batch Size**: 64.
  - **Sequence Length**: Variable, typically 100 during training.
  - **Vocabulary Size**: 65, representing possible character outputs.

### Exploring Model Output

- **Output Shape**:
  - **Batch Size**: 64.
  - **Sequence Length**: 100.
  - **Vocabulary Size**: 65.

- **Predictions**:
  - For each batch, the model predicts the next character for each sequence.
  - Each prediction is a probability distribution over the vocabulary.

### Important Considerations

- **Batch Size Flexibility**: The model is initially trained with a batch size of 64 but can be adjusted to 1 for single-sequence predictions.
- **Return Sequences**: Ensures that the model provides outputs at each time step, not just the final output.
- **Dense Layer**: Converts RNN outputs into a probability distribution for character prediction.

These notes summarize the process of preparing data, building a model, and understanding the model's output in the context of training a Recurrent Neural Network for Natural Language Processing tasks.


## Chunk 8 (5:50:44 - 6:00:44)

## Natural Language Processing with RNNs: Understanding Model Outputs and Training

### Nested Layers and Time Steps

- **Nested Layers**: RNNs process sequences one step at a time, resulting in nested arrays for each time step.
  - Each sequence of length 100 results in 100 outputs for a single training example.
  - These outputs are predictions at each time step.

- **Output Shape**: 
  - At the first time step, the model outputs a tensor of length 65, representing the probability of each character occurring next.

### Custom Loss Function

- **Need for Custom Loss Function**:
  - TensorFlow lacks a built-in loss function for comparing 3D nested arrays of probabilities.
  - A custom loss function is necessary to evaluate model performance accurately.

- **Sampling vs. Argmax**:
  - **Sampling**: Picks a character based on the probability distribution, not necessarily the highest probability.
  - **Argmax**: Choosing the character with the highest probability can lead to repetitive outputs and potential infinite loops.

### Training the Model

1. **Compiling the Model**:
   - Use the **Adam optimizer** and the custom loss function.
   - Set up checkpoints to save model states during training.

2. **Training Process**:
   - Ensure GPU acceleration is enabled for efficient training.
   - Initial training can be done for a few epochs (e.g., 2) to observe initial results.
   - More epochs generally improve model performance without overfitting.

3. **Rebuilding the Model**:
   - After training, rebuild the model with a batch size of 1 to allow flexible input lengths.

### Generating Text

- **Loading Weights and Generating Text**:
  - Load the latest checkpoint weights.
  - Use a function to generate text based on a starting string (e.g., "Romeo").

- **Temperature Parameter**:
  - Controls text predictability:
    - *Low temperature*: More predictable text.
    - *High temperature*: More surprising text.

### Practical Example

- **Example Output**:
  - Given the input "Romeo", the model generates a sequence like:
    ```
    Romeo loose give Lady Capulet, food martone. Father gnomes come to those shell...
    ```
  - This output is pseudo-English due to limited training epochs.

### Key Concepts

- **Batch Size**: Initially set to 64 for training, reduced to 1 for flexible input during text generation.
- **Checkpointing**: Saves model state at each epoch, allowing for loading specific training states.
- **Sampling**: Essential for generating diverse text outputs, avoiding repetitive patterns.

### Summary

- RNNs process sequences step-by-step, resulting in nested outputs.
- Custom loss functions and sampling techniques are crucial for effective model training and text generation.
- Proper training setup, including GPU acceleration and checkpointing, enhances model performance and flexibility.


## Chunk 9 (6:00:44 - 6:08:00)

## Natural Language Processing with RNNs

### Text Generation with RNNs

- **Objective**: Generate sequences using a Recurrent Neural Network (RNN).
- **Process Overview**:
  1. **Initialize Input**: Start with an encoded start string.
  2. **Generate Characters**: Loop through the desired number of characters to generate (e.g., 800).
  3. **Prediction and Sampling**:
     - Use the model to predict the next character.
     - **TensorFlow Squeeze**: `TF.squeeze(predictions, axis=0)` removes extra dimensions from predictions.
     - Apply a **categorical distribution** to predict the next character.
     - Adjust predictions with a **temperature** parameter (default is 1).
     - Sample the predicted ID from the model's output.
  4. **Update Input**: Add the predicted character to the input for the next iteration.
  5. **Convert and Append**: Convert predicted integers back to strings and append to the generated text.

### Code Implementation Summary

- **Setup**:
  - Load and preprocess text data.
  - Create a vocabulary and encode text.
  - Define sequence length (e.g., 100).
  - Convert text to integer sequences.

- **Data Preparation**:
  - Use `tf.data.Dataset.from_tensor_slices` to create character datasets.
  - Batch sequences and map them to create training examples.

- **Model Training**:
  - Define model architecture and parameters.
  - Shuffle and batch data for training.
  - Compile and train the model using a loss function and optimizer.
  - Save model weights at each epoch using checkpoints.

- **Model Evaluation**:
  - Train on different datasets (e.g., B movie script vs. Romeo and Juliet).
  - **Observation**: Longer, well-structured texts like "Romeo and Juliet" yield better results.

### Practical Example: Training on B Movie Script

- **Dataset**: B movie script loaded as `B movie.txt`.
- **Comparison**: Shorter than "Romeo and Juliet", leading to less effective results.
- **Training**: Model trained for 50 epochs.
- **Results**: Demonstrated text generation with input strings like "Hello".

### Tips for Improvement

- **Epochs**: Increase the number of epochs to improve model performance.
- **Loss Monitoring**: Aim for a lower loss value; observe loss trends over epochs.
- **Data Variety**: Experiment with different texts to see varied model outputs.

### Conclusion

- **Understanding Complexity**: This section introduces complex concepts in machine learning.
- **Learning Approach**: 
  - Focus on understanding syntax and creating a working prototype.
  - Encourage further exploration and research for deeper understanding.
- **Next Steps**: Transition to learning about reinforcement learning in the upcoming module.

### Key Terms

- **RNN (Recurrent Neural Network)**: A type of neural network designed for sequence prediction.
- **TensorFlow**: A machine learning framework used for building and training models.
- **Categorical Distribution**: A probability distribution used to predict discrete outcomes.
- **Temperature**: A parameter that influences the randomness of predictions.

These notes provide a structured overview of the text generation process using RNNs, highlighting key steps, concepts, and practical applications.
