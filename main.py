# my_project/main.py
import numpy as np
from image_classification_using_spiking_neural_networks.snn import SpikingNeuralNetwork
from image_classification_using_spiking_neural_networks.utils.helper import load_mnist, encode_spikes, split_dataset

def main():
    # Load full MNIST dataset
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # Encode spikes (using more efficient encoding)
    print("Encoding training spikes...")
    train_spikes = encode_spikes(train_images)
    print("Encoding test spikes...")
    test_spikes = encode_spikes(test_images)
    
    # Initialize SNN with optimized architecture
    input_size = train_images.shape[1]
    hidden_size = 256  # Increased hidden size for better learning
    output_size = 10
    snn = SpikingNeuralNetwork(input_size, hidden_size, output_size)
    
    # Train and test
    from scripts.train import train
    from scripts.test import test
    
    print("Training...")
    train(snn, train_spikes, train_labels, epochs=3)  # Reduced epochs for speed
    
    print("Testing...")
    test(snn, test_spikes, test_labels)

if __name__ == "__main__":
    main()