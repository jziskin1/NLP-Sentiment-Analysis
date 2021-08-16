import os
import re
import json
import pickle
import numpy as np
from num2words import num2words
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors

from .utilities import word_counts, save_to_file


def load_data(filepath, max_samples=None):
    """
    Takes a .json filepath with yelp reviews and appends the sentiment
    (1 meaning positive for 4-5 star ratings and 0 meaning negative for 1-3 star ratings)
    to the sentiment list and the text to the text list.

    Function will load same number of positive and negative reviews

    Return the text list and sentiments numpy array as a tuple
    """
    stars_to_sentiment = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    sentiments = []
    text = []

    # Open file and read lines
    with open(filepath) as f:
        lines = f.readlines()

        # If given max_samples, set a limit for each review type and count to make sure they're equal
        if max_samples is not None:
            limit = max_samples//2
            counts = {0: 0, 1: 0}

        # Loop through every sample
        for line in lines:
            data = json.loads(line)
            sentiment = stars_to_sentiment[data["stars"]]

            # Add sentiments and text
            if counts[sentiment] < limit:
                text.append(data['text'])
                sentiments.append(sentiment)
                counts[sentiment] += 1

            # Break out of loop if positive and negative reviews hit limit
            if counts[0] == limit and counts[1] == limit:
                break

        return text, sentiments


def clean_text(text, stop_list=[], stop_only=False):
    """
    Takes a list of strings and "cleans" the text.

    We define cleaning as...

    replacing any dashes with spaces (ie: "long-term" -> "long term")
    removing punctuation/special characters (ie: "great!" -> "great",   "wish," -> "wish")
    changing all words to lowercase (ie: "Great" -> "great",   "FANTASTIC," -> "fantastic")
    removing words that appear on stop list if one is provided
    turn reviews into lists of words

    if stop_only flag set to true, it will the ignore previous cleaning and only check to see
    that word is not on stoplist before adding to string. This is used when when have to get
    the stop list from the cleaned text to begin with.

    Returns a list of strings containing the words used in each review.
    """
    clean = []

    if not stop_only:

        # Loop through every review
        for review in text:
            cleaned_review = ""

            # Loop through every word in the review splitting along spaces, dashes, or periods
            for word in re.split(r'[.\s-]+', review):
                # If word is number, change to word
                if word.isdecimal():
                    word = num2words(word)

                # Remove special characters from word and make lowercase
                word = re.sub('[^A-Za-z0-9]+', '', word).lower()

                # Add any words to cleaned_review
                if word != "" and word not in stop_list:
                    cleaned_review = cleaned_review + word + " "

            # Add cleaned review to clean
            clean.append(cleaned_review[:-1])

    else:
        # Only check to see that word isn't in stoplist
        for review in text:
            cleaned_review = ""
            for word in review.split():
                if word not in stop_list:
                    cleaned_review = cleaned_review + word + " "
            clean.append(cleaned_review[:-1])

    return clean


def create_stop_list(counts):
    """
    Takes a dictionary of words counts and has the user determine
    which words should be on the stop list.
    """
    print("Let's create a stoplist from our cleaned yelp review data.")
    stop_list_count = 0
    seen_words_count = 0
    total_words = sum(counts.values())
    stop_list = []

    for word in counts:
        print(f"The word '{word}' appears {counts[word]} times.")
        print(f"It makes up {round(counts[word]/total_words*100,4)}% of all words used.")
        seen_words_count += counts[word]
        decision = False

        while not decision:
            print(f"Would you like to add '{word}' to the stop list? Answer 'y' or 'n'.")
            choice = input()
            if choice == "y":
                stop_list.append(word)
                stop_list_count += counts[word]
                print(f"\nThe word '{word}' has been added to the stop list.\n")
                decision = True

            elif choice == "n":
                print(f"\nThe word '{word}' has not been added to the stop list.\n")
                decision = True

            else:
                print("\nResponse unclear.")

        print(f"The stop list has {len(stop_list)} words which accounts for",
              f"{round(stop_list_count/total_words*100,4)}% of the words in the dataset.")
        print(f"The words you've seen so far account for {round(seen_words_count/total_words*100,4)}% of the dataset.")
        decision = False

        while not decision:
            print("Would you like to keep adding words to the stop list? Answer 'y' or 'n'.")
            choice = input()
            if choice == "y":
                decision = True

            elif choice == "n":
                print("\nThe final list of words on the stop list are:")
                print(stop_list)
                print("Enter y to save stoplist to file.")
                if input() == "y":
                    save_to_file(stop_list, "./YelpReviews/stoplists/stoplist.txt")
                return stop_list

            else:
                print("Response unclear.")
            print("")

    print("There are no more words to consider.")
    return stop_list


def words2vecs(text, sentiments, modelpath):
    """
    Takes a list of string of words and a trained
    words2vec model to change the words to vectors.

    Returns a numpy array where each value is
    vector representation of the word.

    Shorter reviews are padded with 0 vectors

    Gives option to save text vectors and sentiment
    vectors to file.
    """
    # Load model
    model = KeyedVectors.load_word2vec_format(f"./YelpReviews/w2v_models/{modelpath}", binary=True)
    print(f"Words2vec model: {modelpath} loaded.")

    # Initiate empty vectors list
    vectors = []
    max_word_len = 0

    # Loop through each review
    for review in text:

        # Initiate vectorized review list
        vectorized_review = []

        # Loop through each word in review
        for word in review.split():

            # If word is in dictionary, add word vector to vectorized_review. Otherwise, continue
            try:
                vectorized_review.append(model[word])

            except KeyError:
                continue

        # Append each vectorized review to vectors
        if len(vectorized_review) > max_word_len:
            max_word_len = len(vectorized_review)
        vectors.append(np.array(vectorized_review))

    # Pad vectors of shorter length for modeling
    vectors, sentiments = pad_vectors(vectors, sentiments, max_word_len)

    # Ask user if they would like to save vectors
    print("Would you like to save this list of vectors?",
          "Enter 'y' to do so. Enter anything else to continue.")

    # If so, save to vectors repo
    if input() == "y":
        save_to_file([vectors, sentiments], "./YelpReviews/vectors/vectors.npz")

    dims = vectors.shape

    return vectors, sentiments, dims


def pad_vectors(vectors, sentiments=None, max_word_len=0):
    """
    Used by words2vecs to pad reviews so they are all
    of equal dimensions
    """
    # Pad vectors of shorter length for modeling
    padded = []
    removed_count = 0
    for i in range(len(vectors)):
        vector = vectors[i]
        if len(vector) < max_word_len:
            try:
                padding = np.zeros((max_word_len-vector.shape[0], vector.shape[1]))
                padded_vec = np.concatenate((padding, vector))
                padded.append(padded_vec)

            # This is used to catch empty vectors
            except IndexError:
                if len(vector) == 0:
                    removed_count += 1
                    if sentiments:
                        sentiments.pop(i)
                    continue
                raise IndexError

        else:
            padded.append(vector)

    # Change vectors to a numpy array
    padded = np.array(padded)
    if sentiments:
        print(f"Vectorization Complete. {removed_count} reviews were removed.")

    return padded, sentiments


def preprocess(datapath, max_samples, stoplistpath=None, w2vmodelpath=None, directory=None):
    """
    Do all the preprocessing and returns text vectors (either embedded or not) and sentiments as a tuple.

    datapath is the path the the yelp reviews dataset
    max_samples are the number of samples we'll take from the dataset of over 8 million reviews
    stoplistpath is a path to the stoplist. If none is provided, we'll create one.
    w2vmodelpath is a path to a saved words2vec model. If none is provided, word embedding will be done in keras NN.
    directory is the directory path to save output files to.
    """
    # Load data
    text, sentiments = load_data(filepath=datapath, max_samples=max_samples)
    print("Data loaded.")

    # If path to stoplist not included, create stoplist
    if stoplistpath is None:
        # Clean text
        text = clean_text(text)
        print("Data cleaned.")

        # Create word counts
        counts = word_counts(text)

        # Create stop list
        stoplist = create_stop_list(counts)

        # Remove stoplist words
        text = clean_text(text, stop_list=stoplist, stop_only=True)
        print("Stop list words removed.")

    # Otherwise load stoplist as a python list
    else:
        with open(f"./YelpReviews/stoplists/{stoplistpath}") as f:
            # Turn into python list
            stoplist = f.read().splitlines()

            # Clean text
            text = clean_text(text, stop_list=stoplist)
            print("Data cleaned and stop list words removed.")

    # If there is no w2v model, tokenize text for keras embedding
    if w2vmodelpath is None:
        tokenizer = Tokenizer(num_words=25000)
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)
        vectors = pad_sequences(sequences, maxlen=350)

        # Save keras tokenizer
        tokenpath = os.path.join(directory, "keras_tokenizer.pickle")
        with open(tokenpath, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"Words Tokenized and saved to {tokenpath}.")

    # If saved w2v model, load it and apply words2vec
    else:
        vectors, sentiments, dims = words2vecs(text, sentiments, modelpath=w2vmodelpath)
        # Record dimensions in case model gets reloaded
        with open(os.path.join(directory, "dims.txt"), "w") as f:
            f.write(str(dims))

    print("Preprocessing Complete.")
    return vectors, sentiments
