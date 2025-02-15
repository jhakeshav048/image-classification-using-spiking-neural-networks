# scripts/train.py
import numpy as np

def train(snn, train_spikes, train_labels, epochs=10):
    """
    Train the Spiking Neural Network.

    Args:
        snn (SpikingNeuralNetwork): The SNN to train.
        train_spikes (np.ndarray): Spike-encoded training images.
        train_labels (np.ndarray): Training labels.
        epochs (int): Number of training epochs.
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, (spikes, label) in enumerate(zip(train_spikes, train_labels)):
            output = snn.forward(spikes)
            print(f"Sample {i + 1}/{len(train_spikes)}: Output={output}, Label={label}")