Neural Network Fundamentals

Author: KHAI TRAN NGUYEN
Course: Artificial Intelligence (CSC 475)
Assignment: Neural Network Fundamentals – Small Network + MNIST Digit Recognizer

I. Overview

This program implements a feed-forward neural network for classifying handwritten digits (0–9) from the MNIST dataset.
It uses stochastic gradient descent (SGD) and backpropagation to learn from training examples.

The program provides a menu-driven interface that allows you to:
    Train the network on MNIST data
    Load and save trained network weights
    Evaluate accuracy on training and testing sets
    Display misclassified images
    View predictions with ASCII image output

II. Neural Network Architecture

Input layer: 784 neurons
Hidden layer: 15 neurons
Output layer: 10 neurons
    (one per digit 0–9, using one-hot encoding)

Activation: Sigmoid

Training:
    Mini-batches
    Learning rate (η)
    Random initial weight

III. Required Data
- The program expects MNIST data in CSV format, training data and testing data

IV. Compile and Run
- Compile: javac BigNetwork.java
- Run:  java BigNetwork

V. Menu Option Descriptions
1. Train the network
    Reads the training data and performs SGD for multiple epochs
    Prints accuracy for each digit and overall results
    Continues until training is stopped or target accuracy is reached

2. Load a pre-trained network
    Loads weights/biases from a saved file
    Allows testing without retraining

3. Display accuracy on training data
    Runs the model on all training samples
    Prints:
        Correct counts for each digit
        Overall accuracy

4. Display accuracy on testing data
    Runs the model on the testing set
    Prints same statistics as training accuracy

5. Show images & predictions
    Iterates through testing cases and prints:
    ASCII representation of the handwritten digit
    Correct label
    Network output (predicted digit)
    Whether correct or incorrect

6. Display misclassified images
    Shows only examples where the output was wrong
    Helps diagnose the model’s weaknesses

7. Save network state   
    Writes the weights and biases to file
    Allows reloading later (Option 2)

 8. EXIT 
    Exits the application safely

VI. Menu Option Descriptions on Youtube 

https://youtu.be/K9rsMnxV6FA
    
    