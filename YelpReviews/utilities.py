import os
from contextlib import redirect_stdout
import numpy as np


def word_counts(text):
    """
    Takes a list of strings of words and counts
    how many times each word appears in the dataset.

    Returns a list of tuples where each tuple is
    in the form: (word, count).
    """
    # Initiate counts dictionary
    counts = {}

    # Loop through every word in every review
    for review in text:
        for word in review.split():

            # Update counts
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

    # Order the dictionary in descending order
    counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    return counts


def create_dir(w2vmodel):
    """
    Returns new directory for saving models
    """
    root = "YelpReviews/results"
    if not w2vmodel:
        model = "keras"
    else:
        model = w2vmodel.split(".")[0]

    i = 0
    path = os.path.join(root, model, str(i))
    while os.path.exists(path):
        i += 1
        path = os.path.join(root, model, str(i))

    os.mkdir(path)

    return path


def save_to_file(data, path):
    """
    Takes data, saves it as a given file type to a given path
    User has the option to adjust path. Otherwise, default path
    will be taken. Does not return anything.
    """
    # Get directory, filepath, and extension information
    directory, filepath = os.path.split(path)
    filename, extension = filepath.split(".")

    # Loop through files in directory to name next file by default
    i = 0
    full_path = f"{directory}/{filename}{i}.{extension}"
    while os.path.exists(full_path):
        i += 1
        full_path = f"{directory}/{filename}{i}.{extension}"
    filename = filename + str(i)

    # Give user option to change filename
    decision = False
    while not decision:
        print(f"Data will be saved to {full_path}. ")
        print("Do you want to change the path? Answer 'y' or 'n'.")
        choice = input()

        if choice == "y":
            print(f"\nWhat would you like to change the filename '{filename}' to?")
            filename = input()
            print("")
            full_path = f"{directory}/{filename}.{extension}"

        elif choice == "n":
            decision = True

        else:
            print("Response unclear.")
            print("")

    # Save stoplist to txt file
    if extension == "txt":
        with open(full_path, "w") as output:
            for word in data[:-1]:
                output.write(str(word) + '\n')
            output.write(str(data[-1]))

        print(f"Stoplist saved to {full_path}.")

    elif extension == "npz":
        np.savez(full_path, vectors=data[0], sentiments=data[1])
        print(f"Vectors saved to {full_path}.")


def record_args(directory, train_size, model, history, **kwargs):
    """
    Takes the directory and information about the model and
    command line arguments to put into readable .txt file.
    """
    with open(os.path.join(directory, "info.txt"), "w") as f:
        f.write(f"LSTM model trained on {train_size} samples.\n")
        max_val_acc = max(history.history['val_accuracy'])
        epoch = history.history['val_accuracy'].index(max_val_acc)+1
        f.write(f"Model hit maximum validation accuracy of {round(100*max_val_acc, 2)}% on epoch {epoch}.\n\n\n")

        f.write("Command line args:\n\n")
        for k, v in kwargs.items():
            if v:
                f.write(f"  {k}: {v}\n")

        f.write("\n\n")

        with redirect_stdout(f):
            model.summary()
