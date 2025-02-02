import numpy as np

class LIFNeuron:
    """
    Implements a basic Leaky Integrate-and-Fire (LIF) neuron.
    """
    def __init__(self, threshold=1.0, decay=0.9, reset_potential=0.0):
        """
        Initializes the neuron with given threshold, decay factor, and reset potential.
        """
        self.threshold = threshold
        self.decay = decay
        self.reset_potential = reset_potential
        self.potential = 0.0
        self.spike = False

    def update(self, input_current):
        """
        Updates the membrane potential based on input current and checks for spiking.
        """
        self.potential = self.potential * self.decay + input_current
        if self.potential >= self.threshold:
            self.spike = True
            self.potential = self.reset_potential
        else:
            self.spike = False
        return self.spike

class Layer:
    """
    Represents a layer of LIF neurons.
    """
    def __init__(self, num_neurons):
        """
        Initializes a layer with a specified number of LIF neurons.
        """
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]
    
    def forward(self, inputs):
        """
        Processes input currents and returns spikes from all neurons.
        """
        return np.array([neuron.update(inp) for neuron, inp in zip(self.neurons, inputs)])

class SpikingNeuralNetwork:
    """
    A simple feedforward Spiking Neural Network.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the SNN with input, hidden, and output layers.
        """
        self.input_layer = Layer(input_size)
        self.hidden_layer = Layer(hidden_size)
        self.output_layer = Layer(output_size)
        
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
    
    def forward(self, inputs):
        """
        Forward pass through the network.
        """
        hidden_input = np.dot(inputs, self.weights_input_hidden)
        hidden_output = self.hidden_layer.forward(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output)
        return self.output_layer.forward(output_input)
