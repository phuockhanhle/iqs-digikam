import math

import pandas as pd
import json


def get_dataframe(mean_raw_file):
    df_data = pd.read_csv(mean_raw_file)
    return df_data[['image_id', 'score']]


def parse_raw_data(df_data):
    samples = []
    list_image_id = []
    for i, row in df_data.iterrows():
        image_id = row['image_id']
        if not math.isnan(image_id):
            if image_id in list_image_id:
                continue
            list_score = df_data[df_data['image_id'] == image_id]['score'].tolist()[:10]
            samples.append({'image_id': int(image_id), 'label': list_score})
            list_image_id.append(image_id)
    return samples


if __name__ == '__main__':
    df = get_dataframe('./votes_filtered.csv')
    data = parse_raw_data(df)
    with open('./samples.json', 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
