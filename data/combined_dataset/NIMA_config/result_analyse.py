import json
from typing import List, Dict, Any
from sklearn.metrics import f1_score


def extract_result_by_dataset(dataset_name: str, prediction_file: str) -> List[Dict[str, Any]]:
    res = []
    with open(prediction_file) as f:
        samples = json.load(f)
    for idx, sample in enumerate(samples):
        if dataset_name.lower() in sample["image_path"].lower():
            res.append(sample)
    print(f"dataset {dataset_name} contains {len(res)} samples.")
    return res


def evaluation(prediction: List[Dict[str, Any]]):
    label, predict = [], []
    for sample in prediction:
        label.append(sample["digikam_label"])
        predict.append(sample["mean_score_prediction"])
    print(f"accuracy : {sum(1 for x, y in zip(label, predict) if x == y) / float(len(label))}")
    print(f"F1 score : {f1_score(label, predict, average=None)}")


if __name__ == '__main__':
    for dataset in ["AVA", "EVA", "koniq10k", "spaq"]:
        print("----------------------------------")
        print(f"evaluation on dataset {dataset}")
        prediction_result = extract_result_by_dataset(dataset, "./data/combined_dataset/NIMA_config/pred.json")
        evaluation(prediction_result)


