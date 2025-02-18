# Image Classification using Spiking Neural Networks (SNNs)

This project implements a **Spiking Neural Network (SNN)** for image classification. The SNN is based on the Leaky Integrate-and-Fire (LIF) neuron model and is designed to classify images from datasets like MNIST.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Training and Testing](#training-and-testing)
6. [Dataset](#dataset)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview

The goal of this project is to demonstrate how Spiking Neural Networks (SNNs) can be used for image classification tasks. The SNN is implemented using Python and NumPy, and it includes:

- **LIF Neuron Model**: A basic Leaky Integrate-and-Fire neuron.
- **Layers**: A layer of LIF neurons for feedforward processing.
- **SNN Architecture**: A simple feedforward SNN with input, hidden, and output layers.
- **MNIST Dataset**: Preprocessing and spike encoding for the MNIST dataset.

---

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/my_project.git
   cd my_project
   ```
2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
3. Verify the installation

   ```bash
   python image_classification_using_spiking_neural_networks/main.py
   ```

<h2>Project Structure

The project is organized as follows:

```bash
my_project/                  # Root folder
│
├── my_project/              # Main package folder
│   ├── __init__.py          # Makes the folder a Python package
│   ├── snn.py               # Spiking Neural Network implementation
│   ├── main.py              # Main script for running the SNN
│   └── utils/               # Utility functions
│       ├── __init__.py
│       └── helper.py        # Dataset loading and preprocessing
│
├── scripts/                 # Scripts for training and testing
│   ├── train.py             # Training script
│   └── test.py              # Testing script
│
├── data/                    # Folder for datasets
│   └── mnist/               # MNIST dataset
│
├── tests/                   # Unit tests
│   ├── __init__.py
│   └── test_snn.py          # Tests for the SNN
│
├── requirements.txt         # List of dependencies
├── README.md                # Project overview
└── setup.py                 # For packaging and distribution
```

<h2>Usage

To use the Spiking Neural Network, follow these steps:

1. **Load the dataset:**
   The MNIST dataset is automatically downloaded and preprocessed using the `helper.py` module.
2. **Train the SNN:**
   Run the training script to train the SNN on the MNIST dataset:

   ```bash
   python scripts/train.py
   ```
3. Test the SNN

   After training, test the SNN on the test dataset: 

```bash
python scripts/test.py
```

<h2>Training and Testing

### Training

The training script (`scripts/train.py`) trains the SNN on the MNIST dataset. It performs the following steps:

* Loads and preprocesses the dataset.
* Encodes images into spike trains.
* Trains the SNN using the spike-encoded data.

### Testing

The testing script (`scripts/test.py`) evaluates the trained SNN on the test dataset. It calculates the accuracy of the model.


## Dataset

The project currently supports the  **MNIST dataset** ,
 which consists of 28x28 grayscale images of handwritten digits (0-9).
The dataset is automatically downloaded and preprocessed using the `helper.py` module.

To add more datasets (e.g., CIFAR-10), extend the `helper.py` file with additional dataset loading functions.


## Contributing

Contributions to this project are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your fork.
4. Submit a pull request.


## Acknowledgments

* This project is inspired by research on Spiking Neural Networks and their applications in image classification.
* Special thanks to the creators of the MNIST dataset for providing a benchmark for testing.

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/yourusername/my_project).
