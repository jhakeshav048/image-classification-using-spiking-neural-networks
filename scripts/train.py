import numpy as np


def train(snn, train_spikes, train_labels, epochs=5, batch_size=100, patience=2):
    best_accuracy = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        correct = 0
        for i in range(0, len(train_labels), batch_size):
            batch_spikes = train_spikes[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            
            batch_correct = 0
            for spikes, label in zip(batch_spikes, batch_labels):
                output = snn.forward(spikes)
                predicted = np.argmax(output)
                snn.update_weights(label)
                if predicted == label:
                    batch_correct += 1
            
            correct += batch_correct
            batch_acc = batch_correct / len(batch_labels)
            print(f"Batch {i//batch_size}: Accuracy = {batch_acc:.2%}")
        
        epoch_acc = correct / len(train_labels)
        print(f"Epoch {epoch+1} Accuracy: {epoch_acc:.2%}")
        
        # Early stopping
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break