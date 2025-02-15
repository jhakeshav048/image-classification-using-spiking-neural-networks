# scripts/test.py
import numpy as np

def test(snn, test_spikes, test_labels):
    """
    Test the Spiking Neural Network.

    Args:
        snn (SpikingNeuralNetwork): The SNN to test.
        test_spikes (np.ndarray): Spike-encoded test images.
        test_labels (np.ndarray): Test labels.
    """
    correct = 0
    for spikes, label in zip(test_spikes, test_labels):
        output = snn.forward(spikes)
        predicted_label = np.argmax(output)
        if predicted_label == label:
            correct += 1
    accuracy = correct / len(test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")