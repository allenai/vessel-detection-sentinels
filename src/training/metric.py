import typing as t

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix


def compute_loc_performance(
    gt_array: np.ndarray, pred_array: np.ndarray, distance_tolerance: int = 20
) -> t.Tuple[float, float, float]:
    """Compute confusion matrix components for gt/prediction locations."""
    # distance_matrix below doesn't work when preds is empty, so handle that first
    if len(pred_array) == 0:
        return [], [], gt_array.tolist()

    # Building distance matrix using Euclidean distance pixel space
    # multiplied by the UTM resolution (10 m per pixel)
    dist_mat = distance_matrix(pred_array, gt_array, p=2)
    dist_mat[dist_mat > distance_tolerance] = 99999

    # Using Hungarian matching algorithm to assign lowest-cost gt-pred pairs
    rows, cols = linear_sum_assignment(dist_mat)

    tp_inds = [
        {"pred_idx": rows[ii], "gt_idx": cols[ii]}
        for ii in range(len(rows))
        if dist_mat[rows[ii], cols[ii]] < distance_tolerance
    ]
    tp = [
        {
            "pred": pred_array[a["pred_idx"]].tolist(),
            "gt": gt_array[a["gt_idx"]].tolist(),
        }
        for a in tp_inds
    ]
    tp_inds_pred = set([a["pred_idx"] for a in tp_inds])
    tp_inds_gt = set([a["gt_idx"] for a in tp_inds])
    fp = [
        pred_array[i].tolist() for i in range(len(pred_array)) if i not in tp_inds_pred
    ]
    fn = [gt_array[i].tolist() for i in range(len(gt_array)) if i not in tp_inds_gt]

    return tp, fp, fn


def score(
    gt: dict, pred: dict, distance_tolerance: int = 20
) -> t.Tuple[float, float, float, dict]:
    """Compute confusion matrix based metrics for ground truth/preds."""
    tp, fp, fn = [], [], []

    for scene_id in gt.keys():
        cur_tp, cur_fp, cur_fn = compute_loc_performance(
            gt[scene_id], pred[scene_id], distance_tolerance=distance_tolerance
        )
        tp += [{"scene_id": scene_id, "pred": a["pred"], "gt": a["gt"]} for a in cur_tp]
        fp += [{"scene_id": scene_id, "point": a} for a in cur_fp]
        fn += [{"scene_id": scene_id, "point": a} for a in cur_fn]

    if len(tp) == 0:
        return 0, 0, 0, None

    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))
    fscore = 2 * precision * recall / (precision + recall)
    return precision, recall, fscore, {"tp": tp, "fp": fp, "fn": fn}
