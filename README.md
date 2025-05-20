# IFumbledAnInterviewSoIMadeANeuralNetwork
Long story short, I fumbled an interview. Hard in fact. And to reaffirm that I know a little bit about what I'm talking about, I built a neural net...... from scratch using nothing but Numpy, Pandas and MatplotLib. 

## Overview

This implementation uses a simple architecture:
- Input layer: 784 neurons (28x28 pixel images)
- Hidden layer: 10 neurons with ReLU activation
- Output layer: 10 neurons with softmax activation (one for each digit 0-9)

## Features

- Two-layer neural network implementation from scratch using NumPy
- ReLU activation function for the hidden layer
- Softmax activation for the output layer
- Gradient descent optimization
- Training and validation split of the MNIST dataset
- Visualization of predictions with matplotlib

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib

## Project Structure

```
.
├── main.py           # Main implementation of the neural network
├── data/
│   └── train.csv    # MNIST training data
└── README.md
```

## Usage

1. Ensure you have the [MNIST dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) in CSV format in the `data/train.csv` file
2. Run the main script (I used Mamba/Conda to manage my Venv but use whatever tickles your fancy):
   ```bash
   python main.py
   ```

From the top:
1. Load and preprocess the MNIST data
2. Train the neural network for 500 iterations
3. Display predictions for the first 4 test images
4. Show the accuracy on the validation set

## Implementation Details

The neural network includes:
- Forward propagation with ReLU and softmax activations
- Backward propagation for gradient computation
- One-hot encoding for labels
- Mini-batch gradient descent optimization
- Learning rate of 0.10
- 500 training iterations

## Performance

I mean, for two layers, it gets above 80% which isnt bad. Could always add more layers which might be done in a future branch

## Note

This is a barebones at best and is more so a "tee-hee I did a thing" project. Hopefully you learned something from this. 