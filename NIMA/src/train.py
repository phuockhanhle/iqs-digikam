import os
import argparse
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from data_generator import TrainDataGenerator, TestDataGenerator
from model_builder import Nima
from samples_loader import load_samples
from config_loader import load_config
from utils import ensure_dir_exists, set_logger
from predict import evaluate_core
import logging
from PIL import ImageFile

logger = logging.getLogger()
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(
        base_model_name,
        n_classes,
        samples,
        image_dir,
        batch_size,
        epochs_train_dense,
        epochs_train_all,
        learning_rate_dense,
        learning_rate_all,
        dropout_rate,
        job_dir,
        img_format='jpg',
        existing_weights=None,
        multiprocessing_data_load=False,
        num_workers_data_load=2,
        decay_dense=0,
        decay_all=0,
        **kwargs
):
    # build NIMA model and load existing weights if they were provided in config
    nima = Nima(
        base_model_name, n_classes, learning_rate_dense, dropout_rate, decay=decay_dense
    )
    nima.build()

    if existing_weights != '':
        logger.info(f"Logging model from {existing_weights}")
        nima.nima_model.load_weights(existing_weights)

    # split samples in train and validation set, and initialize data generators
    samples_train, samples_test = train_test_split(
        samples, test_size=0.05, shuffle=True, random_state=10207
    )

    training_generator = TrainDataGenerator(
        samples_train,
        image_dir,
        batch_size,
        n_classes,
        nima.preprocessing_function(),
        img_format=img_format,
    )

    validation_generator = TestDataGenerator(
        samples_test,
        image_dir,
        batch_size,
        n_classes,
        nima.preprocessing_function(),
        img_format=img_format,
    )

    # initialize callbacks TensorBoard and ModelCheckpoint
    tensorboard = TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'), update_freq='batch'
    )

    model_save_name = (
            'weights_' + base_model_name.lower() + '_{epoch:02d}_{val_loss:.3f}.hdf5'
    )
    model_file_path = os.path.join(job_dir, 'weights', model_save_name)
    model_checkpointer = ModelCheckpoint(
        filepath=model_file_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
    )

    # start training only dense layers
    for layer in nima.base_model.layers:
        layer.trainable = False

    nima.compile()

    nima.nima_model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        epochs=epochs_train_dense,
        verbose=1,
        use_multiprocessing=multiprocessing_data_load,
        workers=num_workers_data_load,
        max_queue_size=30,
        callbacks=[tensorboard],
    )

    # start training all layers
    for layer in nima.base_model.layers:
        layer.trainable = True

    nima.learning_rate = learning_rate_all
    nima.decay = decay_all
    nima.compile()

    nima.nima_model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        epochs=epochs_train_dense + epochs_train_all,
        initial_epoch=epochs_train_dense,
        verbose=1,
        use_multiprocessing=multiprocessing_data_load,
        workers=num_workers_data_load,
        max_queue_size=30,
        callbacks=[tensorboard, model_checkpointer],
    )

    return nima

    # K.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job-dir', help='train job directory with samples and config file', required=True)
    parser.add_argument('-it', '--image-dir', help='directory with image files for train')
    parser.add_argument('-ew', '--existing-weights', help='path of weights file', default='')
    parser.add_argument('-ie', '--image-source', help='image directory or file', required=True)
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)
    parser.add_argument('-rf', '--reference-file', help='file with reference', required=True, default=None)

    args = parser.parse_args()

    image_dir = args.__dict__['image_dir']
    job_dir = args.__dict__['job_dir']

    ensure_dir_exists(os.path.join(job_dir, 'weights'))
    ensure_dir_exists(os.path.join(job_dir, 'logs'))
    set_logger(os.path.join(job_dir, 'logs'))

    config_file = os.path.join(job_dir, 'config.json')
    config = load_config(config_file)
    logger.info(config)

    samples_file = os.path.join(job_dir, 'samples_train.json')
    samples_ = load_samples(samples_file)

    config["existing_weights"] = args.existing_weights
    model_nima = train(samples=samples_, job_dir=job_dir, image_dir=image_dir, **config)
    evaluate_core(model_nima, args.image_source, args.predictions_file, args.reference_file)
    K.clear_session()
