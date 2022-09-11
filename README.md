# iqs-digikam

The main idea of Image quality sorter in digiKam is to determine the quality of an image and convert it into a score. This score is based on four criteria which are factors sabotaging image: blur, noise, exposure, and compression. The current approach is useful to determine images distorted by one of these reasons. However, the current algorithm presents also some drawbacks : It demands a lots of fine tuning from user's side and it's cannot work on aesthetic image. So, I propose the solution of deep learning algorithm. While the dataset and the paper for aesthetic image quality assessment are free to use, we are capable of constructing a mathematical model that can learn the pattern of a dataset, hence, predict the score of quality. As deep learning is an end-to-end solution, it doesn’t require the setting for the hyperparameters. Hence, we can reduce most of the fine-tuning parts to make this feature easier to use
## Approach
At the end, the main purpose of Image quality sorter in digiKam is to label a image as Accepted Image, Pending Image and Rejected Image. Hence, to adapt the context of digiKam , I consider the problem as classification of 3 classes based on 3 labels of digiKam.

The main issue of this project is to define what is a good aesthetic image. Although there are various of dataset for image quality assessment, the definition of a good image is very different. However, by my research, there are two types of dataset. The first one contains natural images that are captured in different situations of distortion (blur, noise, exposure). The second one contains aesthetic image. Hence, I define each class as :
- *Rejected image* is the normal image with distortion
- *Pending image* is the normal image without distortion or artistic image but badly captured
- *Accepted image* is a good artistic image.

I used 4 dataset for training and evaluation : [Koniq10k][Koniq10k] (normal image), [SPAQ][SPAQ] (normal image), [EVA][EVA] (aesthetic image), [AVA][AVA] (aesthetic image). After labeling each image by above definition, I applied the next structure to train the model.

## Data processing
**EVA dataset** :
- Download data from https://github.com/kang-gnak/eva-dataset
- Change the path of [line 38](https://github.com/phuockhanhle/iqs-digikam/blob/b8f75e1b93f5fdd8cd0aedbd0626f1a00553b7f9/data/EVA_dataset/NIMA_config/convert_label_nima.py#L38) in data/EVA_dataset/NIMA_config/convert_label_nima.py to adapt the place of the dataset
- Run `python ./data/EVA/NIMA_config/convert_label_nima.py`

**AVA dataset** :
- Download data from https://github.com/imfing/ava_downloader
- Change the path of [line 52](https://github.com/phuockhanhle/iqs-digikam/blob/b8f75e1b93f5fdd8cd0aedbd0626f1a00553b7f9/data/AVA_dataset/NIMA_config/convert_label_nima.py#L52) to [line 55](https://github.com/phuockhanhle/iqs-digikam/blob/b8f75e1b93f5fdd8cd0aedbd0626f1a00553b7f9/data/AVA_dataset/NIMA_config/convert_label_nima.py#L55) in data/AVA_dataset/NIMA_config/convert_label_nima.py to adapt the place of the dataset
- Run `python ./data/AVA/NIMA_config/convert_label_nima.py`

**Combine datasets**:
- Run `python ./data/combined_dataset/NIMA_config/combine_data.py`

## Training and evaluation
To train and evaluate the model with the parameter of current report:

`source NIMA/train.sh`

To change the configuration for training, change the hyper-parameter in [here](https://github.com/phuockhanhle/iqs-digikam/blob/main/data/combined_dataset/NIMA_config/config.json)


The model checkpoint is saved after each epoch





[EVA]: https://github.com/kang-gnak/eva-dataset
[AVA]: http://refbase.cvc.uab.es/files/MMP2012a.pdf
[Koniq10k]: http://database.mmsp-kn.de/koniq-10k-database.html
[project-proposal]: https://summerofcode.withgoogle.com/media/user/3bea17365af2/proposal/znmmTvwbY9aBIkA7.pdf
[iqs-digikam-repo]: https://github.com/phuockhanhle/iqs-digikam
[NIMA-repo]: https://github.com/idealo/image-quality-assessment
[musiq-repo]: https://github.com/google-research/google-research/tree/master/musiq
[SRCC]: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
[F1_score]: https://medium.com/synthesio-engineering/precision-accuracy-and-f1-score-for-multi-label-classification-34ac6bdfb404
[SPAQ]: https://github.com/h4nwei/SPAQ
[freeze-model]: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
[inference-c++]: https://github.com/phuockhanhle/iqs-digikam/blob/inference_cpp/inference/inference_iqs/inference_iqs.cpp
[softmax]: https://en.wikipedia.org/wiki/Softmax_function
[InceptionResNetV2]: https://keras.io/api/applications/inceptionresnetv2/
[Model-path]: https://drive.google.com/file/d/1c4uVuyLp_eqE1vLCEVgU0LXv1ik5YgA1/view?usp=sharing
[AVA-pred-file]: https://raw.githubusercontent.com/phuockhanhle/iqs-digikam/main/data/AVA_dataset/NIMA_config/pred.json
[JAX]: https://github.com/google/jax