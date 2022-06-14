from typing import Dict
import json
import math
from sklearn.metrics import mean_squared_error
import logging
logger = logging.getLogger()


def get_score_ref(file_path: str) -> Dict[str, float]:
    result = {}
    with open(file_path) as json_file:
        data = json.load(json_file)
    for ex in data:
        result[ex['image_id']] = sum([ex['label'][i] * i for i in range(len(ex['label']))]) / sum(ex['label'])
    return result


def get_score_candidate(file_path: str) -> Dict[str, float]:
    result = {}
    with open(file_path) as json_file:
        data = json.load(json_file)
    for ex in data:
        result[ex['image_id']] = ex['mean_score_prediction']
    return result


def calculate_score(reference: Dict[str, float], candidate: Dict[str, float]):
    y_pred = []
    y_test = []
    list_id_absent = []
    for idx in reference.keys():
        if idx in candidate.keys():
            y_test.append(reference[idx])
            y_pred.append(candidate[idx])
        else:
            list_id_absent.append(idx)
    logger.info(f"list id absent : {list_id_absent}")
    logger.info(f"***** MSE: {mean_squared_error(y_test, y_pred)} ****")
    logger.info(f"***** MSE: {math.sqrt(mean_squared_error(y_test, y_pred))} ****")


def evaluate(file_reference, file_candidate):
    ref = get_score_ref(file_reference)
    can = get_score_candidate(file_candidate)
    calculate_score(ref, can)

