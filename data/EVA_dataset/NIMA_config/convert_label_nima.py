import math
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from tqdm import tqdm


def get_dataframe(mean_raw_file):
    df_data = pd.read_csv(mean_raw_file)
    return df_data[['image_id', 'score']]


def parse_raw_data(df_data):
    samples = []
    list_image_id = []
    for i, row in tqdm(df_data.iterrows()):
        image_id = row['image_id']
        if not math.isnan(image_id):
            if image_id in list_image_id:
                continue
            list_score = df_data[df_data['image_id'] == image_id]['score'].tolist()
            label = []
            for j in range(10):
                label.append(list_score.count(j+1))

            samples.append({'image_id': int(image_id), 'label': label})
            list_image_id.append(image_id)
    return samples


def save_json(data_frame, saved_path):
    samples = parse_raw_data(data_frame)
    with open(saved_path, 'w') as writer:
        json.dump(samples, writer, indent=2, sort_keys=True)


if __name__ == '__main__':
    df = get_dataframe('./data/EVA_dataset/votes_filtered.csv')
    train, test = train_test_split(df, test_size=0.2)
    save_json(train, './data/EVA_dataset/NIMA_config/samples_train.json')
    save_json(test, './data/EVA_dataset/NIMA_config/samples_test.json')
