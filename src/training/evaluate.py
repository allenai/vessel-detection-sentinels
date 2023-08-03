import logging

import numpy as np
import scipy.optimize
import torch
import wandb
from sklearn.metrics import confusion_matrix

from src.data.dataset import NEGATIVE_VESSEL_TYPE_FACTOR
from src.training.utils import compute_f1


class Evaluator:
    def __init__(self, options, tuner_data=None):
        pass

    def update(self, targets: list, outputs: list, loss: float):
        """
        Update Evaluator given the provided targets and outputs.
        Does not return anything.
        """
        pass

    def score(self) -> dict:
        """
        Returns a dict from str -> float of computed scores.
        """
        pass


class Processor:
    def __init__(self, options, windows, tuner_data=None):
        pass

    def process(self, images: list, targets: list, outputs: list):
        """
        Returns a list of siv.Label representing the outputs.
        """
        pass


class Tuner:
    def __init__(self, options):
        pass

    def update(self, targets: list, outputs: list, loss: float):
        """
        Update Tuner given the provided targets and outputs.
        Does not return anything.
        """
        pass

    def tune(self) -> dict:
        """
        Returns a dict, usually from str -> float, of tuner_data.
        """
        pass


class LossEvaluator(Evaluator):
    def __init__(self, options, tuner_data=None):
        self.losses = []

    def update(self, targets: list, outputs: list, loss: float):
        self.losses.append(loss)

    def score(self) -> dict:
        return {"score": -np.mean(self.losses)}


def accuracy(preds, gt):
    """Compute ratio of correct predictions to total predictions
    for classification task, given 1-d torch tensor of predicted classes,
    and 1-d torch tensor of ground truth classes.
    """
    correct = (preds == gt).sum()
    total = len(preds)
    if total > 0:
        return correct / total
    else:
        return 0


class AttributeEvaluator(LossEvaluator):
    def __init__(self, options, tuner_data=None):
        super().__init__(options, tuner_data)
        self._empty_labels()
        self.thresholds = [
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.98,
            0.99,
        ]

    def _empty_labels(self):
        self.losses = []
        self.outputs = []
        self.length_outputs = []
        self.length_targets = []
        self.width_outputs = []
        self.width_targets = []
        self.heading_outputs = []
        self.heading_targets = []
        self.speed_outputs = []
        self.speed_targets = []
        self.type_outputs = []
        self.type_targets = []

    def update(self, targets: list, outputs: list, loss: float):
        self.losses.append(loss)
        targets = torch.stack([target["target"] for target in targets], dim=0)
        self.length_outputs.append(outputs[:, 0])
        self.length_targets.append(targets[:, 0])
        self.width_outputs.append(outputs[:, 1])
        self.width_targets.append(targets[:, 1])
        self.heading_outputs.append(outputs[:, 2:18])
        self.heading_targets.append(targets[:, 2:18])
        self.speed_outputs.append(outputs[:, 18])
        self.speed_targets.append(targets[:, 18])
        self.type_outputs.append(outputs[:, 19:21])
        self.type_targets.append(targets[:, 19:21])

    def score(self) -> dict:
        return {"score": -np.mean(self.losses)}

    def log_metrics(self, logger: logging.Logger, use_wandb=True) -> None:
        """Get MAE and cross entropy metrics for length, width and heading."""
        length_outputs = torch.cat(self.length_outputs, dim=0)
        length_targets = torch.cat(self.length_targets, dim=0)
        width_outputs = torch.cat(self.width_outputs, dim=0)
        width_targets = torch.cat(self.width_targets, dim=0)
        heading_outputs = torch.cat(self.heading_outputs, dim=0)
        heading_targets = torch.cat(self.heading_targets, dim=0)
        speed_outputs = torch.cat(self.speed_outputs, dim=0)
        speed_targets = torch.cat(self.speed_targets, dim=0)
        type_outputs = torch.cat(self.type_outputs, dim=0)
        type_targets = torch.cat(self.type_targets, dim=0)
        width_mae = torch.mean(torch.abs(width_outputs - width_targets))
        length_mae = torch.mean(torch.abs(length_outputs - length_targets))
        speed_mae = torch.mean(torch.abs(speed_outputs - speed_targets))

        # Cross entropies
        heading_ce = torch.nn.functional.cross_entropy(
            input=heading_outputs, target=heading_targets
        )
        type_ce = torch.nn.functional.cross_entropy(
            input=type_outputs,
            target=type_targets,
            weight=torch.tensor([1, NEGATIVE_VESSEL_TYPE_FACTOR]).to(
                type_outputs.device
            ),
        )

        # Accuracy stats
        heading_preds = heading_outputs.argmax(dim=1)
        heading_gt = heading_targets.argmax(dim=1)
        heading_acc = accuracy(heading_preds, heading_gt)

        # Activity type preds conf mat based metrics
        best_f1_score = None
        best_threshold = None
        type_f1s = []
        type_precisions = []
        type_recalls = []
        tns = []
        fps = []
        fns = []
        tps = []
        normalized_type_outputs = torch.nn.functional.softmax(type_outputs, dim=1)

        for _, threshold in enumerate(self.thresholds):
            type_preds = torch.where(normalized_type_outputs[:, 1] > threshold, 1, 0)
            type_gt = type_targets.argmax(dim=1)
            type_acc = accuracy(type_preds, type_gt)

            # Full confusion matrix
            tn, fp, fn, tp = confusion_matrix(type_gt.cpu(), type_preds.cpu()).ravel()
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
            tps.append(tp)
            f1, precision, recall = compute_f1(tp, fp, fn)
            type_f1s.append(f1)
            type_precisions.append(precision)
            type_recalls.append(recall)

            if best_f1_score is None or f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = threshold
                best_threshold_idx = self.thresholds.index(best_threshold)
                best_type_acc = type_acc

        best_precision = type_precisions[best_threshold_idx]
        best_recall = type_recalls[best_threshold_idx]

        # Full confusion matrix for heading
        # heading_conf_mat = confusion_matrix(heading_gt.cpu(), heading_preds.cpu())

        # Log metrics to stdout
        logger.info("New best model save. Reporting val metrics.")
        logger.info(f"Length MAE: {length_mae.item()}")
        logger.info(f"Width MAE: {width_mae.item()}")
        logger.info(f"Speed MAE: {speed_mae.item()}")
        logger.info(f"Heading CE: {heading_ce.item()}")
        logger.info(f"Heading Accuracy: {heading_acc}")
        logger.info(f"Type CE: {type_ce.item()}")
        logger.info(f"Type Accuracy (@best): {best_type_acc}")
        logger.info(
            f"Type (tn, fp, fn, tp) (@best)={tns[best_threshold_idx],fps[best_threshold_idx],fns[best_threshold_idx],tps[best_threshold_idx]}"
        )
        logger.info(f"Type Thresholds: {self.thresholds}")
        logger.info(f"Type True negatives: {tns}")
        logger.info(f"Type True positives: {tps}")
        logger.info(f"Type False positives: {fps}")
        logger.info(f"Type False negatives: {fns}")

        # WANDB Log fishing type confusion
        columns = ["Threshold", "TP", "FP", "FN", "TN", "Recall", "Precision", "F1"]
        data = [
            [
                thresh,
                tps[idx],
                fps[idx],
                fns[idx],
                tns[idx],
                type_recalls[idx],
                type_precisions[idx],
                type_f1s[idx],
            ]
            for idx, thresh in enumerate(self.thresholds)
        ]
        if use_wandb:
            fishing_type_table = wandb.Table(columns=columns, data=data)
            wandb.log({"Vessel-Type Is-Fishing Metrics": fishing_type_table})

        # WANDB Log heading preds confusion matrix
        class_names = [f"cls_{i}" for i in range(16)]
        if use_wandb:
            wandb.log(
                {
                    "conf_mat": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=heading_gt.cpu().tolist(),
                        preds=heading_preds.cpu().tolist(),
                        class_names=class_names,
                    )
                }
            )

            wandb.log(
                {
                    "Val Length MAE": length_mae.item(),
                    "Val Width MAE": width_mae.item(),
                    "Val Speed MAE": speed_mae.item(),
                    "Val Heading CE": heading_ce.item(),
                    "Val Heading Accuracy": heading_acc,
                    "Val Type CEs": type_ce.item(),
                    "Val Type Accuracy (@best)": best_type_acc,
                    "Val Type F1 (@best)": best_f1_score,
                    "Val Type Recall (@best)": best_recall,
                    "Val Type Precision (@best)": best_precision,
                }
            )
        return None


class DetectPerClassF1Evaluator(Evaluator, Tuner):
    """
    Evaluates F1 scores and tunes confidence thresholds for box and point detection tasks.
    Compares each class independently.
    """

    def __init__(self, options, tuner_data=None):
        num_classes = 1  # TODO
        if tuner_data:
            self.thresholds = []
            for cls_idx in range(num_classes):
                threshold = tuner_data["class{}".format(cls_idx)]
                self.thresholds.append([threshold])
        else:
            self.thresholds = [
                [
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    0.95,
                    0.98,
                    0.99,
                ]
                for _ in range(num_classes)
            ]
        self._zero_confusion_matrix()

    def _zero_confusion_matrix(self):
        self.true_positives = [
            [0] * len(self.thresholds[i]) for i in range(len(self.thresholds))
        ]
        self.false_positives = [
            [0] * len(self.thresholds[i]) for i in range(len(self.thresholds))
        ]
        self.false_negatives = [
            [0] * len(self.thresholds[i]) for i in range(len(self.thresholds))
        ]

    def update(self, targets: list, outputs: list, loss: float):
        for img_idx in range(len(targets)):
            for cls_idx, cls_thresholds in enumerate(self.thresholds):
                gt_valid = targets[img_idx]["labels"] == cls_idx
                gt_boxes = targets[img_idx]["boxes"].cpu().numpy()

                for threshold_idx, threshold in enumerate(cls_thresholds):
                    pred_valid = (outputs[img_idx]["scores"] >= threshold) & (
                        outputs[img_idx]["labels"] == cls_idx
                    )
                    pred_boxes = outputs[img_idx]["boxes"][pred_valid, :].cpu().numpy()

                    if len(gt_boxes) == 0:
                        self.false_positives[cls_idx][threshold_idx] += len(pred_boxes)
                        continue
                    elif len(pred_boxes) == 0:
                        self.false_negatives[cls_idx][threshold_idx] += len(gt_boxes)
                        continue

                    # Create binary association matrix of overlapping boxes.
                    gt_tiled = gt_boxes[:, None, :].repeat(
                        repeats=len(pred_boxes), axis=1
                    )
                    pred_tiled = pred_boxes[None, :, :].repeat(
                        repeats=len(gt_boxes), axis=0
                    )
                    assoc = (
                        (gt_tiled[:, :, 0] <= pred_tiled[:, :, 2])
                        & (gt_tiled[:, :, 1] <= pred_tiled[:, :, 3])
                        & (gt_tiled[:, :, 2] >= pred_tiled[:, :, 0])
                        & (gt_tiled[:, :, 3] >= pred_tiled[:, :, 1])
                    )

                    # Optimize the assignment on the association matrix.
                    rows, cols = scipy.optimize.linear_sum_assignment(
                        1 - assoc.astype("float32")
                    )
                    tp = len(
                        [ii for ii in range(len(rows)) if assoc[rows[ii], cols[ii]]]
                    )
                    fp = len(pred_boxes) - tp
                    fn = len(gt_boxes) - tp
                    self.true_positives[cls_idx][threshold_idx] += tp
                    self.false_positives[cls_idx][threshold_idx] += fp
                    self.false_negatives[cls_idx][threshold_idx] += fn

    def _score(self):
        best_scores = {}
        best_thresholds = {}

        for cls_idx, cls_thresholds in enumerate(self.thresholds):
            best_score = None
            best_threshold = None

            for threshold_idx, threshold in enumerate(cls_thresholds):
                tp = self.true_positives[cls_idx][threshold_idx]
                fp = self.false_positives[cls_idx][threshold_idx]
                fn = self.false_negatives[cls_idx][threshold_idx]

                f1, _, _ = compute_f1(tp, fp, fn)

                if best_score is None or f1 > best_score:
                    best_score = f1
                    best_threshold = threshold

            best_scores["class{}".format(cls_idx)] = best_score
            best_thresholds["class{}".format(cls_idx)] = best_threshold

        best_scores["score"] = np.mean(list(best_scores.values()))
        return best_scores, best_thresholds

    def score(self):
        scores, _ = self._score()
        return scores

    def tune(self):
        _, tuner_data = self._score()
        return tuner_data

    def log_metrics(
        self, class_name: str, logger: logging.Logger, use_wandb=True
    ) -> None:
        """Get point/box detection metrics for specified class_name (at best threshold)."""
        best_thresholds = self.tune()
        classes = list(best_thresholds.keys())
        class_idx = classes.index(class_name)
        logger.info(f'New best model save. Getting metrics for class "{class_name}"')

        thresholds = self.thresholds[class_idx]
        best_threshold = self.tune()[class_name]
        best_threshold_idx = thresholds.index(best_threshold)

        tps = self.true_positives[class_idx]
        fps = self.false_positives[class_idx]
        fns = self.false_negatives[class_idx]
        logger.info(f"Thresholds: {thresholds}")
        logger.info(f"True positives: {tps}")
        logger.info(f"False positives: {fps}")
        logger.info(f"False negatives: {fns}")

        precisions = []
        recalls = []
        f1s = []

        for (tp, fp, fn) in zip(tps, fps, fns):
            f1, precision, recall = compute_f1(tp, fp, fn)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        logger.info(f"F1s: {f1s}")
        logger.info(f"Precisions: {precisions}")
        logger.info(f"Recalls: {recalls}")

        logger.info(f"Best F1 occurs at threshold {best_threshold}")
        logger.info(
            f"At best threshold:\nPrecision: {precisions[best_threshold_idx]} \nRecall: {recalls[best_threshold_idx]} \nF1: {f1s[best_threshold_idx]}\n"
        )
        if use_wandb:
            wandb.log(
                {
                    "Val Precision (@best)": precisions[best_threshold_idx],
                    "Val Recall (@best)": recalls[best_threshold_idx],
                    "val F1 (@best)": f1s[best_threshold_idx],
                }
            )

        columns = ["Threshold", "TP", "FP", "FN", "Recall", "Precision", "F1"]
        data = [
            [
                thresh,
                tps[idx],
                fps[idx],
                fns[idx],
                recalls[idx],
                precisions[idx],
                f1s[idx],
            ]
            for idx, thresh in enumerate(thresholds)
        ]
        if use_wandb:
            detection_metrics_table = wandb.Table(columns=columns, data=data)
            wandb.log({"Detection Metrics": detection_metrics_table})
        return None


class MSEEvaluator(Evaluator):
    def __init__(self, options, tuner_data=None):
        self.scores = []

    def update(self, targets: list, outputs: list, loss: float):
        for img_idx in range(len(targets)):
            score = torch.square(targets[img_idx]["target"] - outputs[img_idx]).mean()
            self.scores.append(score.item())

    def score(self) -> dict:
        if self.scores:
            return {"score": -np.mean(self.scores)}
        else:
            return 0


class DetectProcessor(Processor):
    def __init__(self, task: str, options: dict, windows: list, tuner_data=None):
        self.task = task
        self.windows = windows
        num_classes = 1
        if tuner_data:
            self.thresholds = tuner_data
        else:
            self.thresholds = [0] * num_classes

    def process(self, images: list, targets: list, outputs: list):
        labels = []

        for img_idx, output in enumerate(outputs):
            window_idx = targets[img_idx]["image_id"].item()
            window = self.windows[window_idx]

            for pred_idx, box in enumerate(output["boxes"].tolist()):
                score = output["scores"][pred_idx].item()
                cls_idx = output["labels"][pred_idx].item()

                if score < self.thresholds[cls_idx]:
                    continue

                d = {
                    "WindowID": window["ID"],
                    "Score": score,
                    "CategoryID": cls_idx,
                }

                if self.task == "point":
                    column = (box[0] + box[2]) / 2
                    row = (box[1] + box[3]) / 2
                    column *= window["Width"] / images[img_idx].shape[2]
                    row *= window["Height"] / images[img_idx].shape[1]

                    d["Column"] = window["Column"] + int(column)
                    d["Row"] = window["Row"] + int(row)

                elif self.task == "box":
                    box[0] *= window["Width"] / images[img_idx].shape[2]
                    box[1] *= window["Height"] / images[img_idx].shape[1]
                    box[2] *= window["Width"] / images[img_idx].shape[2]
                    box[3] *= window["Height"] / images[img_idx].shape[1]

                    d["Column"] = window["Column"] + int(box[0])
                    d["Row"] = window["Row"] + int(box[1])
                    d["Width"] = int(box[2])
                    d["Height"] = int(box[3])

                labels.append(d)

        return labels


class ValueProcessor(Processor):
    def __init__(self, options: dict, windows: list, tuner_data=None):
        self.windows = windows

    def process(self, images: list, targets: list, outputs: list):
        labels = []

        for img_idx, output in enumerate(outputs):
            window_idx = targets[img_idx]["image_id"].item()
            window = self.windows[window_idx]

            labels.append(
                {
                    "WindowID": window["ID"],
                    "Value": output.item(),
                }
            )

        return labels


def get_evaluator(task, options, tuner_data=None):
    name = options.get("Evaluator", "accuracy")

    if name == "loss":
        if task == "custom":
            return AttributeEvaluator(options, tuner_data=tuner_data)
        else:
            return LossEvaluator(options, tuner_data=tuner_data)
    if task in ["point", "box"]:
        if name == "per_class_f1":
            return DetectPerClassF1Evaluator(options, tuner_data=tuner_data)
    elif task in ["regression", "segmentation"]:
        if name == "mse":
            return MSEEvaluator(options, tuner_data=tuner_data)

    return None


def get_tuner(task, options):
    name = options.get("Tuner", "f1")

    if task in ["point", "box"]:
        if name == "per_class_f1":
            return DetectPerClassF1Evaluator(options)

    return None


def get_processor(task, options, windows, tuner_data=None):

    if task in ["point", "box"]:
        return DetectProcessor(task, options, windows, tuner_data=tuner_data)
    elif task == "regression":
        return ValueProcessor(options, windows, tuner_data=tuner_data)

    return None


def evaluate(
    model,
    device,
    loader,
    half_enabled=False,
    tuner_data=None,
    evaluator=None,
    processor=None,
):
    eval_losses = []
    processed_outputs = []

    if isinstance(evaluator, DetectPerClassF1Evaluator):
        evaluator._zero_confusion_matrix()
    if isinstance(evaluator, AttributeEvaluator):
        evaluator._empty_labels()

    with torch.no_grad():
        for images, targets in loader:
            images = list(image.to(device).float() / 255 for image in images)
            gpu_targets = [
                {
                    k: v.to(device)
                    for k, v in t.items()
                    if not isinstance(v, str) and not isinstance(v, tuple)
                }
                for t in targets
            ]

            with torch.autocast(device.type, enabled=half_enabled):
                outputs, loss = model(images, gpu_targets)

            eval_losses.append(loss.item())

            if evaluator:
                evaluator.update(gpu_targets, outputs, loss.item())
            if processor:
                processed_outputs.extend(
                    processor.process(images, gpu_targets, outputs)
                )

    return np.mean(eval_losses), processed_outputs
