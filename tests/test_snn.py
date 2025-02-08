import unittest
import numpy as np
from image_classification_using_spiking_neural_networks.snn import LIFNeuron, Layer, SpikingNeuralNetwork

class TestLIFNeuron(unittest.TestCase):
    """Tests for the LIFNeuron class."""
    
    def test_initial_state(self):
        """Test if neuron initializes with correct parameters."""
        neuron = LIFNeuron()
        self.assertEqual(neuron.potential, 0.0)
        self.assertFalse(neuron.spike)
    
    def test_spiking_behavior(self):
        """Test if neuron spikes correctly when threshold is reached."""
        neuron = LIFNeuron(threshold=1.0)
        self.assertFalse(neuron.update(0.5))  # Should not spike
        self.assertTrue(neuron.update(0.6))   # Should spike
    
    def test_leakage(self):
        """Test if neuron potential decays over time."""
        neuron = LIFNeuron(decay=0.5)
        neuron.update(1.0)
        self.assertLess(neuron.potential, 1.0)

class TestLayer(unittest.TestCase):
    """Tests for the Layer class."""
    
    def test_layer_size(self):
        """Test if layer initializes with correct number of neurons."""
        layer = Layer(5)
        self.assertEqual(len(layer.neurons), 5)
    
    def test_layer_forward(self):
        """Test if layer correctly processes input currents."""
        layer = Layer(3)
        inputs = np.array([1.0, 0.5, 0.3])
        spikes = layer.forward(inputs)
        self.assertEqual(len(spikes), 3)
        self.assertTrue(all(isinstance(spike, bool) for spike in spikes))

class TestSpikingNeuralNetwork(unittest.TestCase):
    """Tests for the SpikingNeuralNetwork class."""
    
    def test_network_initialization(self):
        """Test if network initializes with correct layer sizes."""
        snn = SpikingNeuralNetwork(4, 3, 2)
        self.assertEqual(len(snn.input_layer.neurons), 4)
        self.assertEqual(len(snn.hidden_layer.neurons), 3)
        self.assertEqual(len(snn.output_layer.neurons), 2)
    
    def test_forward_pass(self):
        """Test if forward pass returns correct output shape."""
        snn = SpikingNeuralNetwork(5, 3, 2)
        inputs = np.random.rand(5)
        output_spikes = snn.forward(inputs)
        self.assertEqual(len(output_spikes), 2)
        self.assertTrue(all(isinstance(spike, bool) for spike in output_spikes))

if __name__ == "__main__":
    unittest.main()
