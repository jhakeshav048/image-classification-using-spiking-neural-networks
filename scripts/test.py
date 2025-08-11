import numpy as np
from tqdm import tqdm

def test(snn, test_spikes, test_labels):
    correct = 0
    confidences = []
    
    for spikes, label in zip(test_spikes, test_labels):
        output = snn.forward(spikes)
        predicted = np.argmax(output)
        confidence = output[predicted]
        
        if predicted == label:
            correct += 1
        confidences.append(confidence)
    
    accuracy = correct / len(test_labels)
    avg_confidence = np.mean(confidences)
    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"Average Confidence: {avg_confidence:.2%}")