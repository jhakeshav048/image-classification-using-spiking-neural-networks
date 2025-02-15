# my_project/main.py
import numpy as np
from image_classification_using_spiking_neural_networks.snn import SpikingNeuralNetwork
from image_classification_using_spiking_neural_networks.utils.helper import load_mnist, encode_spikes, split_dataset

def main():
    # Load MNIST dataset
    train_images, train_labels, test_images, test_labels = load_mnist()

    # Encode images into spikes
    train_spikes = encode_spikes(train_images)
    test_spikes = encode_spikes(test_images)

    # Initialize SNN
    input_size = train_spikes.shape[1]
    hidden_size = 100  # Adjust as needed
    output_size = 10   # MNIST has 10 classes
    snn = SpikingNeuralNetwork(input_size, hidden_size, output_size)

    # Train the SNN
    from scripts.train import train
    train(snn, train_spikes, train_labels)

    # Test the SNN
    from scripts.test import test
    test(snn, test_spikes, test_labels)

if __name__ == "__main__":
    main()