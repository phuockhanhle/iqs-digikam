import os
import glob
import argparse
import json
import logging
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2 as cvtc2
import tensorflow as tf
import numpy
from evaluate import evaluate
from utils import save_json, set_logger
from model_builder import Nima
from data_generator import TestDataGenerator
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger()


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.' + img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': int(img_id)})

    return samples


def ref_file_to_json(ref_file):
    samples = []
    with open(ref_file) as json_file:
        data = json.load(json_file)
    for ex in data:
        if "image_path" not in ex.keys():
            samples.append({'image_id': int(ex['image_id']), 'digikam_label': int(ex['digikam_label'])})
        else:
            samples.append({'image_path': ex['image_path'], 'digikam_label': int(ex['digikam_label'])})
    return samples


def predict(model, data_generator):
    return model.predict(data_generator, verbose=1)


def evaluate_core(model, image_source, predictions_file, reference_file, img_format='jpg'):
    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source

    samples = ref_file_to_json(reference_file)

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 4, 3, model.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    predictions = predict(model.nima_model, data_generator)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        # sample['mean_score_prediction'] = calc_mean_score(predictions[i])
        sample['mean_score_prediction'] = int(numpy.argmax(predictions[i]))

    if predictions_file is not None:
        save_json(samples, predictions_file)

    evaluate(reference_file, samples)


def main(base_model_name, weights_file, image_source, predictions_file, reference_file, img_format='jpg'):
    # build model and load weights
    nima = Nima(base_model_name, n_classes=3, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    evaluate_core(nima, image_source, predictions_file, reference_file, img_format)

    full_model = tf.function(lambda inputs: nima.nima_model(inputs))
    full_model = full_model.get_concrete_function(
        [tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in nima.nima_model.inputs])
    frozen_func = cvtc2(full_model)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="./frozen_models",
                      name="simple_frozen_graph.pb", as_text=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)
    parser.add_argument('-rf', '--reference-file', help='file with reference', required=True, default=None)

    args = parser.parse_args()
    set_logger(os.path.join("./", 'logs'))

    main(**args.__dict__)
