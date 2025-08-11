# my_project/utils/helper.py
"""
Utility functions for Image Classification using Spiking Neural Networks.
"""

import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist, cifar10

def preprocess_images(image_paths, target_size=(28, 28)):
    """
    Preprocess images by resizing and normalizing pixel values.

    Args:
        image_paths (list): List of file paths to images.
        target_size (tuple): Target size for resizing images.

    Returns:
        np.ndarray: Preprocessed images as a numpy array.
    """
    images = []
    for path in image_paths:
        img = Image.open(path).convert("L")  # Convert to grayscale
        img = img.resize(target_size)       # Resize to target size
        img = np.array(img) / 255.0         # Normalize pixel values to [0, 1]
        images.append(img)
    return np.array(images)

def encode_spikes(images, timesteps=15):
    """
    Encode images into spike trains using a threshold.

    Args:
        images (np.ndarray): Preprocessed images.
        threshold (float): Threshold for spike encoding.

    Returns:
        np.ndarray: Spike-encoded images.
    """
    # Pre-compute probabilities
    probs = np.tile(images, (timesteps, 1, 1)).transpose(1, 0, 2)
    # Generate all spikes at once
    spikes = np.random.random(probs.shape) < probs
    return spikes.astype(np.float32)  # Use float32 to save memory

def load_dataset(dataset_path):
    """
    Load and preprocess a dataset of images.

    Args:
        dataset_path (str): Path to the dataset directory.

    Returns:
        np.ndarray: Preprocessed images.
    """
    # Example: Load images from the dataset directory
    import os
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".png")]
    return preprocess_images(image_paths)

def split_dataset(images, labels, train_ratio=0.8):
    """
    Split a dataset into training and testing sets.

    Args:
        images (np.ndarray): Preprocessed images.
        labels (np.ndarray): Corresponding labels.
        train_ratio (float): Ratio of training data.

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    num_train = int(len(images) * train_ratio)
    train_images, test_images = images[:num_train], images[num_train:]
    train_labels, test_labels = labels[:num_train], labels[num_train:]
    return train_images, train_labels, test_images, test_labels

def load_mnist():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)
    return train_images, train_labels, test_images, test_labels


def load_cifar10():
    """
    Load and preprocess the CIFAR-10 dataset.

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize images to [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Flatten images to 1D arrays
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    return train_images, train_labels, test_images, test_labels