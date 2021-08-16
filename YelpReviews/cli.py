#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import numpy as np

from .processing import preprocess
from .models import train_model
from .visualization import create_accuracy_plot
from .predict import predict_sentiment
from .utilities import create_dir, record_args


def get_args():
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--samplesize", type=int, nargs="?", default=10000)
    parser.add_argument("-s", "--stoplist", type=str, nargs="?")
    parser.add_argument("-v", "--vectors", type=str, nargs="?")
    parser.add_argument("-w", "--w2vmodel", type=str, nargs="?")
    parser.add_argument("-M", "--model", type=str, nargs="?")
    parser.add_argument("-c", "--conv", action='store_true')
    args = parser.parse_args()

    if args.model and not args.w2vmodel and args.model[:5] != "keras":
        args.w2vmodel = args.model.split("/")[0] + ".bin.gz"

    return args


def main(args):
    args = get_args()
    path = "./YelpReviews/data/yelp_academic_dataset_review.json"

    # Predict sentiment of user generated review based on loaded LSTM model
    if args.model:
        directory = os.path.join("./YelpReviews/results/", args.model)
        predict_sentiment(args.w2vmodel, directory, modelpath="best_model.hdf5")
        print("Program Terminated")
        return

    # Create directory to save models and visualizations to
    directory = create_dir(args.w2vmodel)

    # If no saved vectors or w2v models are provided (start from scratch)
    if args.vectors is None and args.w2vmodel is None:
        X, y = preprocess(path, max_samples=args.samplesize, stoplistpath=args.stoplist,
                          w2vmodelpath=args.w2vmodel, directory=directory)
        embedded = False

    # Load saved vectors if provided
    elif args.vectors is not None:
        saved = np.load(f"./YelpReviews/vectors/{args.vectors}")
        X, y = saved["vectors"], saved["sentiments"]
        embedded = True

    # If there aren't saved vectors, but we're given a w2v model, use it to get vectors
    else:
        X, y = preprocess(path, max_samples=args.samplesize, stoplistpath=args.stoplist,
                          w2vmodelpath=args.w2vmodel, directory=directory)
        embedded = True

    # Train model
    model, history = train_model(X, y, embedded=embedded, conv=args.conv, directory=directory)

    # Plot accuracy
    create_accuracy_plot(history, directory)

    # Record model args in a txt
    record_args(directory, int(0.8*len(X)), model, history, samplesize=args.samplesize, stoplist=args.stoplist,
                vectors=args.vectors, w2vmodel=args.w2vmodel, conv=args.conv)

    # Predict sentiment of user generated review
    predict_sentiment(args.w2vmodel, directory, model=model)
    print("Program Terminated")


if __name__ == "__main__":
    main()
