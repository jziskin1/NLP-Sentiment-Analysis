import os
import matplotlib.pyplot as plt


def create_accuracy_plot(history, directory):
    """
    Takes model history and plots training and validation accuracy.
    Saves results in directory.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], '-o', label="Training Accuracy")
    plt.plot(history.history['val_accuracy'], '-o', label="Validation Accuracy")
    plt.title("Accuracy of Yelp Review Sentiment using LSTM")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(directory, "accuracy.png"))