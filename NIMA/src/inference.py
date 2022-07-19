import argparse

import numpy as np

import utils
from model_builder import Nima

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='image to inference', required=True)
    parser.add_argument('-bs', '--base-model', help='base model name', required=True)
    parser.add_argument('-ew', '--existing-weights', help='path of weights file', required=True)
    args = parser.parse_args()

    # Load model
    nima = Nima(
        args.base_model, 3)
    nima.build()
    print(f"Logging model from {args.existing_weights}")
    nima.nima_model.load_weights(args.existing_weights)

    # Preprocessing
    X = np.empty((1, 224, 224, 3))
    X[0, ] = utils.load_image(args.image, (224, 224))
    X = nima.preprocessing_function()(X)

    # Serve model
    prediction = nima.nima_model(X)
    print(prediction)


