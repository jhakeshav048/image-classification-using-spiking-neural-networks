import numpy as np

class SpikingNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with proper scaling
        scale = 0.1
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * scale
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * scale
        
        # Neuron parameters
        self.hidden_threshold = 1.0
        self.output_threshold = 1.0
        self.tau = 10.0  # Membrane time constant
        
        # Initialize activation trackers
        self.last_input_spikes = np.zeros(input_size)
        self.last_hidden_spikes = np.zeros(hidden_size)
        self.last_output = np.zeros(output_size)
    
    def forward(self, input_spikes, timesteps=15):
        # Initialize potentials
        hidden_potential = np.zeros(self.hidden_size)
        output_potential = np.zeros(self.output_size)
        
        # Initialize spike counters
        output_spike_counts = np.zeros(self.output_size)
        hidden_spike_counts = np.zeros(self.hidden_size)
        
        for t in range(timesteps):
            # Store current input spikes
            current_input = input_spikes[t]
            self.last_input_spikes = current_input
            
            # Input to hidden layer
            hidden_input = np.dot(current_input, self.weights_input_hidden)
            hidden_potential += (hidden_input - hidden_potential)/self.tau
            hidden_spikes = (hidden_potential >= self.hidden_threshold).astype(float)
            hidden_potential[hidden_spikes.astype(bool)] = 0  # Reset
            hidden_spike_counts += hidden_spikes
            self.last_hidden_spikes = hidden_spikes
            
            # Hidden to output layer
            output_input = np.dot(hidden_spikes, self.weights_hidden_output)
            output_potential += (output_input - output_potential)/self.tau
            output_spikes = (output_potential >= self.output_threshold).astype(float)
            output_potential[output_spikes.astype(bool)] = 0  # Reset
            output_spike_counts += output_spikes
            
            # Store last output (average over timesteps)
            self.last_output = output_spike_counts / (t + 1)
        
        # Return normalized spike counts
        total = np.sum(output_spike_counts)
        return output_spike_counts / (total + 1e-8)  # Avoid division by zero
    
    def update_weights(self, target, learning_rate=0.001):
        # Create target vector
        target_vector = np.zeros(self.output_size)
        target_vector[target] = 1.0
        
        # Calculate output error
        output_error = target_vector - self.last_output
        
        # Update hidden-output weights (STDP-like rule)
        delta_output = np.outer(self.last_hidden_spikes, output_error)
        self.weights_hidden_output += learning_rate * delta_output
        
        # Calculate hidden error
        hidden_error = np.dot(self.weights_hidden_output, output_error)
        
        # Update input-hidden weights
        delta_input = np.outer(self.last_input_spikes, hidden_error)
        self.weights_input_hidden += learning_rate * delta_input
        
        # Apply weight constraints (keep weights positive)
        self.weights_input_hidden = np.maximum(0, self.weights_input_hidden)
        self.weights_hidden_output = np.maximum(0, self.weights_hidden_output)