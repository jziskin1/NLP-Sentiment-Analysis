import os
import pickle
from keras.models import load_model
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from .processing import pad_vectors


def predict_sentiment(w2vmodel, directory, model=None, modelpath=None):
    """
    Interactive predictive function that takes a model and predicts sentiment of user's review
    directory is the directory where saved files can be found
    model is the LSTM model
    modelpath is a path to a saved LSTM model
    w2v is a words2vec model used during training. Will be autofilled if model is entered.
    """
    print("Would you like to enter a review and see if I can detect its sentiment?")
    print("Enter 'y' to continue.")

    predict_flag = input() == "y"

    # Return if "y" is not selected
    if not predict_flag:
        return

    # Load model path if one is given and show model summary
    if modelpath:
        model = load_model(os.path.join(directory, modelpath))

    print("\nWe will be using this model to predict")
    model.summary()
    print("Please wait while the model loads")

    # Load word tokenizer or embedder to predict reviews
    if w2vmodel:
        w2v = KeyedVectors.load_word2vec_format(f"./YelpReviews/w2v_models/{w2vmodel}", binary=True)

    else:
        with open(os.path.join(directory, "keras_tokenizer.pickle"), "rb") as f:
            tokenizer = pickle.load(f)

    while predict_flag:
        print("\nOkay. Enter your review.")
        review = input()

        if w2vmodel:
            vector = []
            for word in review.split():

                # If word is in dictionary, add word vector to review. Otherwise, continue.
                try:
                    vector.append(w2v[word])

                except KeyError:
                    continue

            vector = np.array(vector)

            with open(os.path.join(directory, "dims.txt"), "r") as f:
                dims = f.read()
                pad_length = int(dims.split()[1][:-1])
            vector, _ = pad_vectors([vector], max_word_len=pad_length)

        else:
            sequences = tokenizer.texts_to_sequences([review])
            vector = pad_sequences(sequences, maxlen=350)

        prediction = float(model.predict(vector)[0])

        if prediction > 0.5:
            print(f"\nThere is a {round(prediction*100, 4)}% this review is positive.")
        else:
            print(f"\nThere is a {round((1-prediction)*100, 4)}% this review is negative.")

        print("\nWould you like to try another review? Press 'y'.")
        predict_flag = input() == "y"