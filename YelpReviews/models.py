import os
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


def train_model(vectors, sentiments, embedded=False, conv=False, directory=None):
    """
    Trains model given transformed yelp review vectors and sentiments list

    If embedded flag set to False, it will performs word embedding in the neural network
    If embedded and conv flag are set to True, model will add a 1D convolutional layer to embedding
    directory is the directory to save model output to

    Returns model and history
    """
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(vectors, sentiments, test_size=0.2)

    # Initiate model
    print("Building Model")
    model = Sequential()

    # Add embedding layer if necessary
    if not embedded:
        model.add(Embedding(25000, 150, input_length=350))

        # Use 1D convolutional layer if specified
        if conv:
            model.add(Dropout(0.2))
            model.add(Conv1D(75, 5, activation='relu'))
            model.add(MaxPooling1D(pool_size=4))

    # Use LSTM with 150 units, dropout and recurrent dropout of 0.2
    model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Create model checkpoint to save best model
    checkpoint = ModelCheckpoint(os.path.join(directory, "best_model.hdf5"), monitor='val_accuracy',
                                 verbose=1, save_best_only=True, mode='max', save_weights_only=False)

    # Fit model
    history = model.fit(X_train, np.array(y_train), validation_data=(X_test, np.array(y_test)),
                        epochs=10, callbacks=[checkpoint])

    return model, history