import math
import os
import random
import pandas as pd
import json
from tqdm import tqdm
random.seed(42)


def get_dataframe(mean_raw_file):
    df_data = pd.read_csv(mean_raw_file, skiprows=1, sep='\s+', names=[i for i in range(15)])
    return df_data[[i for i in range(1, 12)]]


def parse_raw_data(df_data, source, list_id_image):
    samples = []
    for i, row in tqdm(df_data.iterrows()):
        image_id = int(row[1])
        if image_id in list_id_image and check_file_exist(source, image_id):
            label = [int(row[j]) for j in range(2, 12)]
            score = sum([label[k] * (k + 1) for k in range(len(label))]) / sum(label)
            cls = math.trunc(score) + 1
            if cls <= 5:
                digikam_label = 0
            elif cls <= 6:
                digikam_label = 1
            else:
                digikam_label = 2
            samples.append(
                {'image_id': image_id, 'label': label, 'score': score, "class": cls, "digikam_label": digikam_label})
    return samples


def get_id(*, path, mode):
    res = []
    list_file = os.listdir(path)
    for file_name in list_file:
        file_path = os.path.join(path, file_name)
        if mode in file_name:
            with open(file_path) as reader:
                raw = reader.readlines()
            ids = [int(item.split(r"\n")[0]) for item in raw]
            res.extend(ids)
    return res


def check_file_exist(base_path, file_name):
    return os.path.exists(os.path.join(base_path, "%s.jpg" % file_name))


if __name__ == '__main__':
    source_path = r'D:/AVA_dataset/images/images/'
    df = get_dataframe('D:/AVA_dataset/AVA.txt')
    list_id_train = get_id(path=r'D:/AVA_dataset/aesthetics_image_lists/', mode="train")
    list_id_test = get_id(path=r'D:/AVA_dataset/aesthetics_image_lists/', mode="test")
    list_id = list_id_train + list_id_test
    random.shuffle(list_id)
    list_id_train = list_id[:int(len(list_id) * 0.8)]
    list_id_test = list_id[int(len(list_id) * 0.8):]

    data_train = parse_raw_data(df, source_path, list_id_train)
    with open('./data/AVA_dataset/NIMA_config/samples_train.json', 'w') as f:
        json.dump(data_train, f, indent=2, sort_keys=True)

    data_test = parse_raw_data(df, source_path, list_id_test)
    with open('./data/AVA_dataset/NIMA_config/samples_test.json', 'w') as f:
        json.dump(data_test, f, indent=2, sort_keys=True)
