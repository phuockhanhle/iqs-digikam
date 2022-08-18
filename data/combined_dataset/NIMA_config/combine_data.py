import json
from typing import List, Dict, Any
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def add_ava_dataset(sample_file: str, image_dir: str) -> List[Dict[str, Any]]:
    res = []
    nb_class0, nb_class1, nb_class2 = 0, 0, 0
    with open(sample_file) as f:
        samples = json.load(f)
    for idx, sample in enumerate(samples):
        image_path = os.path.join(image_dir, "%s.jpg" % sample["image_id"])
        if sample["score"] <= 4:
            digikam_label = 0
            nb_class0 += 1
        elif sample["score"] <= 5:
            digikam_label = 1
            nb_class1 += 1
        elif sample["score"] >= 6:
            digikam_label = 2
            nb_class2 += 1
        res.append({"image_path": image_path, "digikam_label": digikam_label})
    print(
        f"data AVA contains {nb_class0} samples of class 0,{nb_class1} samples of class 1, {nb_class2} samples of class 2")
    return res


def add_eva_dataset(sample_file: str, image_dir: str) -> List[Dict[str, Any]]:
    res = []
    nb_class1, nb_class2 = 0, 0
    with open(sample_file) as f:
        samples = json.load(f)
    for idx, sample in enumerate(samples):
        image_path = os.path.join(image_dir, "%s.jpg" % sample["image_id"])
        if sample["score"] <= 6:
            digikam_label = 1
            nb_class1 += 1
        else:
            digikam_label = 2
            nb_class2 += 1
        res.append({"image_path": image_path, "digikam_label": digikam_label})
    print(
        f"data EVA contains {nb_class1} samples of class 1, {nb_class2} samples of class 2")
    return res


def add_koniq10k_dataset(sample_file: str, image_dir: str) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    train_data, test_data = [], []
    nb_class0 = 0
    df = pd.read_csv(sample_file)
    df_train, df_test = train_test_split(df, test_size=0.2)
    for index, sample in df_train.iterrows():
        image_path = os.path.join(image_dir, sample["image_name"])
        if sample["MOS_zscore"] <= 60:
            digikam_label = 0
            nb_class0 += 1
        else:
            continue
        train_data.append({"image_path": image_path, "digikam_label": digikam_label})
    print(f"data Koniq10k contains {len(train_data)} samples of class 0 for training")
    for index, sample in df_test.iterrows():
        image_path = os.path.join(image_dir, sample["image_name"])
        if sample["MOS_zscore"] <= 60:
            digikam_label = 0
        else:
            continue
        test_data.append({"image_path": image_path, "digikam_label": digikam_label})
    print(f"data Koniq10k contains {len(test_data)} samples of class 0 for testing")
    return train_data, test_data


def add_spaq_dataset(sample_file: str, image_dir) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    train_data, test_data = [], []
    nb_class0 = 0
    df = pd.read_csv(sample_file)
    df_train, df_test = train_test_split(df, test_size=0.2)
    for index, sample in df_train.iterrows():
        image_path = os.path.join(image_dir, sample["Image name"])
        if sample["MOS"] <= 60:
            digikam_label = 0
            nb_class0 += 1
        else:
            continue
        train_data.append({"image_path": image_path, "digikam_label": digikam_label})
    print(f"data SPAQ contains {len(train_data)} samples of class 0 for training")
    for index, sample in df_test.iterrows():
        image_path = os.path.join(image_dir, sample["Image name"])
        if sample["MOS"] <= 60:
            digikam_label = 0
        else:
            continue
        test_data.append({"image_path": image_path, "digikam_label": digikam_label})
    print(f"data SPAQ contains {len(test_data)} samples of class 0 for testing")
    return train_data, test_data


if __name__ == '__main__':
    ava_train = add_ava_dataset("./data/AVA_dataset/NIMA_config/samples_train.json", "D:/AVA_dataset/images/images")
    eva_train = add_eva_dataset("./data/EVA_dataset/NIMA_config/samples_train.json", "./data/EVA_dataset/image")

    ava_test = add_ava_dataset("./data/AVA_dataset/NIMA_config/samples_test.json", "D:/AVA_dataset/images/images")
    eva_test = add_eva_dataset("./data/EVA_dataset/NIMA_config/samples_test.json", "./data/EVA_dataset/image")

    koniq_train, koniq_test = add_koniq10k_dataset("./data/koniq10k_dataset/koniq10k_scores_and_distributions.csv", "D:/koniq10k")
    spaq_train, spaq_test = add_spaq_dataset("./data/SPAQ/SPAQ_annotations.csv", "D:/SPAQ_dataset")
    data_train = ava_train + eva_train + koniq_train + spaq_train
    data_test = ava_test + eva_test + koniq_test + spaq_test
    with open("./data/combined_dataset/NIMA_config/samples_train.json", 'w') as writer:
        json.dump(data_train, writer, indent=2, sort_keys=True)

    with open("./data/combined_dataset/NIMA_config/samples_test.json", 'w') as writer:
        json.dump(data_test, writer, indent=2, sort_keys=True)
