from typing import Dict, List, Any
import json
import math
from sklearn.metrics import f1_score
import logging

logger = logging.getLogger()


def get_score_ref(file_path: str) -> Dict[str, float]:
    result = {}
    with open(file_path) as json_file:
        data = json.load(json_file)
    for ex in data:
        # result[ex['image_id']] = sum([ex['label'][i] * (i + 1) for i in range(len(ex['label']))]) / sum(ex['label'])
        result[ex['image_id']] = ex['digikam_label']
    return result


def get_score_candidate(prediction: List[Any]) -> Dict[str, float]:
    result = {}
    for ex in prediction:
        result[ex['image_id']] = ex['mean_score_prediction']
    return result


def calculate_metric_regression(reference: Dict[str, float], candidate: Dict[str, float]):
    y_pred = []
    y_test = []
    y_pred_cls = []
    y_test_cls = []
    y_pred_cls_digikam = []
    y_test_cls_digikam = []
    list_common_id = [i for i in reference.keys() if i in candidate.keys()]
    list_id_absent = list(set(reference.keys()) - set(list_common_id)) + list(
        set(candidate.keys()) - set(list_common_id))
    for id_image in list_common_id:
        y_pred.append(candidate[id_image])
        y_test.append(reference[id_image])
        y_pred_cls.append(math.trunc(candidate[id_image]) + 1)
        y_test_cls.append(math.trunc(reference[id_image]) + 1)
        y_pred_cls_digikam.append(candidate[id_image])
        y_test_cls_digikam.append(reference[id_image])

    logger.info(f"list id absent : {list_id_absent}")
    # logger.info(f"***** MSE: {mean_squared_error(y_test, y_pred)} ****")
    # logger.info(f"***** RMSE: {math.sqrt(mean_squared_error(y_test, y_pred))} ****")
    # logger.info(f"***** SRCC: {stats.spearmanr(y_test, y_pred)} ****")
    # logger.info(f"***** F1 score: {f1_score(y_test_cls, y_pred_cls, average='weighted')} ****")
    logger.info(f"***** F1 score digikam: {f1_score(y_test_cls_digikam, y_pred_cls_digikam, average=None)} ****")
    logger.info(f"***** accuracy digikam: {multi_class_accuracy(y_test_cls_digikam, y_pred_cls_digikam)} ****")


def evaluate(file_reference, prediction):
    ref = get_score_ref(file_reference)
    can = get_score_candidate(prediction)
    calculate_metric_regression(ref, can)


def multi_class_accuracy(y_test, y_pred):
    true_score = 0
    for idx, y in enumerate(y_test):
        if y == y_pred[idx]:
            true_score += 1
    return true_score / len(y_test)
