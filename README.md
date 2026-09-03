# MNIST Handwritten Digit Recognizer in Java

A **feed-forward neural network built from scratch in Java** to classify handwritten digits from the **MNIST dataset**. The model uses **backpropagation** and **stochastic gradient descent (SGD)** to learn digit patterns without relying on a machine-learning framework.

## Demo

**[Watch the Othello AI Demo on YouTube](https://www.youtube.com/watch?v=K9rsMnxV6FA&t=378s)**

## Screenshots

<img width="1150" height="463" alt="Screenshot 2026-08-25 at 3 06 04 PM" src="https://github.com/user-attachments/assets/494ce253-dfe8-4720-b58c-ae21e569fdf7" />
<img width="1155" height="471" alt="Screenshot 2026-08-25 at 3 06 30 PM" src="https://github.com/user-attachments/assets/57404d90-d35d-4cd2-84d8-9a01d1b7ac4a" />
<img width="1333" height="872" alt="Screenshot 2026-08-25 at 3 06 54 PM" src="https://github.com/user-attachments/assets/0440e369-453c-48fa-af34-e746bfc740e5" />
<img width="845" height="459" alt="Screenshot 2026-08-25 at 3 07 23 PM" src="https://github.com/user-attachments/assets/63b68f62-f611-42b7-bfcc-c54549ad5133" />

## Features

* Classifies handwritten digits **0–9** from MNIST images
* Implements **forward propagation and backpropagation**
* Trains using **mini-batch stochastic gradient descent**
* Evaluates accuracy on both **training and testing datasets**
* Saves and reloads learned **weights and biases**
* Displays predictions using **ASCII-rendered digit images**
* Shows **misclassified samples** for model error analysis
* Reports per-digit and overall classification accuracy

## Neural Network Architecture

```text
Input Layer      784 neurons (28 × 28 pixels)
      ↓
Hidden Layer      15 neurons
      ↓
Output Layer      10 neurons (digits 0–9)
```

* **Activation:** Sigmoid
* **Output Representation:** One-hot encoding
* **Optimization:** Mini-batch SGD
* **Learning Algorithm:** Backpropagation
* **Initialization:** Random weights and biases

## How It Works

Each MNIST image is represented by **784 pixel values** and passed through the neural network.

During **forward propagation**, the network computes activations through the hidden and output layers.

During training, the predicted output is compared with the correct digit label. **Backpropagation** computes how each weight and bias contributed to the prediction error, and **SGD** updates those parameters over repeated mini-batches and epochs.

The output layer contains **10 neurons**, corresponding to digits `0–9`. The neuron with the highest activation becomes the predicted digit.

## Model Evaluation

The application can evaluate the trained network on both training and testing data and report:

* Correct predictions for each digit
* Overall classification accuracy
* Predicted vs. actual labels
* Misclassified samples for debugging and analysis

## Interactive Menu

| Option | Action                               |
| ------ | ------------------------------------ |
| **1**  | Train the neural network             |
| **2**  | Load a pre-trained network           |
| **3**  | Evaluate accuracy on training data   |
| **4**  | Evaluate accuracy on testing data    |
| **5**  | Display digit images and predictions |
| **6**  | Display misclassified images         |
| **7**  | Save network weights and biases      |
| **8**  | Exit                                 |

## Getting Started

### Requirements

* Java Development Kit (JDK)
* MNIST training and testing datasets in CSV format

### Compile

```bash
javac BigNetwork.java
```

### Run

```bash
java BigNetwork
```

## Tech Stack

**Java • Neural Networks • Backpropagation • Stochastic Gradient Descent • MNIST**

## Project Context

Developed for **Artificial Intelligence (CSC 475)** as an implementation of neural-network fundamentals and handwritten-digit classification.

## Author

**Khai Tran Nguyen**
