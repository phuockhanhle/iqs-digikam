from typing import List, Dict, Any
import json


def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def analyse_data(list_example: List[Dict[str, Any]]):
    count_dict = {}
    count_dict_cls_digikam = {}
    for ex in list_example:
        if ex["class"] not in count_dict.keys():
            count_dict[ex["class"]] = 1
        else:
            count_dict[ex["class"]] += 1

        if ex["digikam_label"] not in count_dict_cls_digikam.keys():
            count_dict_cls_digikam[ex["digikam_label"]] = 1
        else:
            count_dict_cls_digikam[ex["digikam_label"]] += 1
    print(f"number sample of each class {count_dict}")
    print(f"number sample of each class digikam {count_dict_cls_digikam}")


def balancing_data(list_example, limit_each_class=9000):
    saved_data = []
    count_dict_cls_digikam = {}
    for ex in list_example:
        if ex["digikam_label"] not in count_dict_cls_digikam.keys():
            count_dict_cls_digikam[ex["digikam_label"]] = 1
            saved_data.append(ex)
        else:
            if count_dict_cls_digikam[ex["digikam_label"]] < limit_each_class:
                count_dict_cls_digikam[ex["digikam_label"]] += 1
                saved_data.append(ex)
    print(f"number sample of each class digikam {count_dict_cls_digikam}")
    return saved_data


def save_json(data_frame: List[Dict[str, Any]], saved_path):
    with open(saved_path, 'w') as writer:
        json.dump(data_frame, writer, indent=2, sort_keys=True)


if __name__ == '__main__':
    data = load_json(r'./data/AVA_dataset/NIMA_config/samples_train.json')
    analyse_data(data)
    saved_data = balancing_data(data)
    save_json(saved_data, r'./data/AVA_dataset/NIMA_config/samples_train.json')
