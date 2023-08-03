import datetime
import json
import logging
import math
import os
import sys
import time
import typing as t

import numpy as np
import torch
import torch.cuda.amp
import torch.utils.data
import wandb

from src.data.dataset import Dataset
from src.data.image import Channels
from src.data.transforms import get_transform
from src.models import models
from src.training.ema import EMA
from src.training.evaluate import evaluate, get_evaluator
from src.training.utils import collate_fn

num_loader_workers = 4

# Acquire logger
logger = logging.getLogger("training")


def train_loop(
    model_cfg: dict,
    dataset: dict,
    windows: t.List[dict],
    save_dir: str,
    training_data_dir: str,
) -> None:
    """Run training loop for detection or attribute prediction model.

    Parameters
    ----------
    model_cfg: dict
        Dictionary of model config.

    dataset: dict
        Dictionary specifying dataset from sqlite DB.

    windows: list[dict]
        List of window dictionaries corresponding to all sqlite records
        from relevant database. Will get filtered by model_cfg.
        TODO: Only retrieve relevant windows to begin with, rather
        than getting all and filtering.

    save_dir: str
        Local directoy in which trained artifacts will be saved.

    training_data_dir: str
        Local directory in which training data (preprocess folder) lives.

    Returns
    -------
    : None
    """
    options = model_cfg["Options"]

    channels = Channels(model_cfg["Channels"])
    task = dataset["task"]
    model_cfg["Data"] = {}
    if dataset.get("task"):
        model_cfg["Data"]["task"] = dataset["task"]
    if dataset.get("categories"):
        model_cfg["Data"]["categories"] = dataset["categories"]
    model_name = model_cfg["Architecture"]

    run_tag = os.environ.get("RUN_TAG", "test_run")
    os.environ["WANDB_DIR"] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    experiment_name = "-".join(
        [task, run_tag, datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%p")]
    )
    run = wandb.init(
        save_code=False,
        name=experiment_name,
        mode=os.environ.get("WANDB_MODE", "offline"),
    )

    train_splits = options.get("TrainSplits", ["train"])
    val_splits = options.get("ValSplits", ["valid"])
    batch_size = options.get("BatchSize", 4)
    effective_batch_size = options.get("EffectiveBatchSize", batch_size)
    num_epochs = options.get("NumberEpochs", 10)
    chip_size = options.get("ChipSize", 0)
    image_size = options.get("ImageSize", 0)

    half_enabled = options.get("Half", True)
    summary_frequency = options.get("SummaryFrequency", 8192)
    restore_path = options.get("RestorePath", None)
    ema_factor = options.get("EMA", 0)
    save_path = os.path.join(save_dir, str(model_cfg["Name"]), run_tag)
    os.makedirs(save_path, exist_ok=True)

    # Prepare transforms.
    train_transforms = get_transform(
        model_cfg, options, options.get("TrainTransforms", [])
    )
    val_transforms = get_transform(model_cfg, options, options.get("ValTransforms", []))

    train_data = Dataset(
        dataset=dataset,
        windows=windows,
        channels=channels,
        splits=train_splits,
        transforms=train_transforms,
        image_size=image_size,
        chip_size=chip_size,
        preprocess_dir=os.path.join(training_data_dir, "preprocess"),
    )

    val_data = Dataset(
        dataset=dataset,
        windows=windows,
        channels=channels,
        splits=val_splits,
        transforms=val_transforms,
        image_size=image_size,
        chip_size=chip_size,
        valid=True,
        preprocess_dir=os.path.join(training_data_dir, "preprocess"),
    )

    # Export model and training config
    logger.info(f"Writing training artifact outputs to {save_path}.")
    with open(os.path.join(save_path, "cfg.json"), "w") as f:
        f.write(json.dumps(model_cfg))

    logger.info(
        "Loaded {} train image references, and {} validation image references from DB.".format(
            len(train_data), len(val_data)
        )
    )
    device = torch.device("cuda")

    train_sampler_cfg = options.get("TrainSampler", {"Name": "random"})
    if train_sampler_cfg["Name"] == "random":
        train_sampler = torch.utils.data.RandomSampler(train_data)
    elif train_sampler_cfg["Name"] == "bg_balanced":
        train_sampler = train_data.get_bg_balanced_sampler()
    else:
        raise Exception("invalid sampler config {}".format(train_sampler_cfg))

    val_sampler = torch.utils.data.SequentialSampler(val_data)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_loader_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_loader_workers,
        collate_fn=collate_fn,
    )

    # instantiate model with a number of classes
    model_cls = models[model_name]
    example = train_data[0]
    logger.debug("The shape of training samples is {}.".format(example[0].shape))
    model = model_cls(
        {
            "Channels": channels,
            "Device": device,
            "Model": model_cfg,
            "Options": options,
            "Data": model_cfg["Data"],
            "Example": example,
        }
    )

    # Restore saved model if requested.
    if restore_path:
        logger.info(f"Restoring model from {restore_path}")
        state_dict = torch.load(restore_path)
        model.load_state_dict(state_dict)

    if ema_factor:
        logger.info("creating EMA model")
        model = EMA(model, decay=ema_factor)

    # move model to the correct device
    model.to(device)

    # construct an optimizer
    optimizer_config = options.get("Optimizer", {})
    optimizer_name = optimizer_config.get("Name", "sgd")
    initial_lr = optimizer_config.get("InitialLR", 0.001)
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=initial_lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=initial_lr)
    else:
        raise Exception("unknown optimizer name {}".format(optimizer_name))

    lr_scheduler = None

    if "Scheduler" in options:
        scheduler_config = options["Scheduler"]
        if scheduler_config["Name"] == "plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_config.get("Factor", 0.1),
                patience=scheduler_config.get("Patience", 2),
                min_lr=scheduler_config.get("MinLR", 1e-5),
                cooldown=scheduler_config.get("Cooldown", 5),
            )
        else:
            raise Exception("invalid scheduler config {}".format(scheduler_config))

    warmup_iters = 0
    warmup_lr_scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=half_enabled)
    best_score = None

    # TensorBoard logging
    cur_iterations = 0
    summary_iters = summary_frequency // batch_size
    summary_epoch = 0
    summary_prev_time = time.time()
    train_losses = []

    if effective_batch_size:
        accumulate_freq = effective_batch_size // batch_size
    else:
        accumulate_freq = 1

    logger.info(f"Beginning training for {num_epochs} epochs.")
    model.train()
    for epoch in range(num_epochs):
        logger.info("Starting epoch {}".format(epoch))

        model.train()
        optimizer.zero_grad()

        for images, targets in train_loader:
            cur_iterations += 1

            images = [image.to(device).float() / 255 for image in images]
            targets = [
                {
                    k: v.to(device)
                    for k, v in t.items()
                    if not isinstance(v, str) and not isinstance(v, tuple)
                }
                for t in targets
            ]

            with torch.cuda.amp.autocast(enabled=half_enabled):
                _, loss = model(images, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                logger.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            scaler.scale(loss).backward()

            if cur_iterations == 1 or cur_iterations % accumulate_freq == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema_factor:
                    model.update(summary_epoch)

            train_losses.append(loss_value)

            if warmup_lr_scheduler:
                warmup_lr_scheduler.step()
                if cur_iterations > warmup_iters + 1:
                    logger.info("removing warmup_lr_scheduler")
                    warmup_lr_scheduler = None

            if cur_iterations % summary_iters == 0:
                train_loss = np.mean(train_losses)

                eval_time = time.time()
                model.eval()
                evaluator = get_evaluator(task, options)
                val_loss, _ = evaluate(
                    model,
                    device,
                    val_loader,
                    half_enabled=half_enabled,
                    evaluator=evaluator,
                )
                val_scores = evaluator.score()
                model.train()

                val_score = val_scores["score"]

                if task == "point":
                    # Note: due to current implementation, constant 0 val-loss is expected for
                    # point detection task.
                    wandb.log({"train_loss": train_loss, "val_score": val_score})
                    logger.info(
                        "summary_epoch {}: train_loss={} val_score={} best_val_score={} elapsed={} lr={}".format(
                            summary_epoch,
                            train_loss,
                            val_score,
                            best_score,
                            int(eval_time - summary_prev_time),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                else:
                    wandb.log({"train_loss": train_loss, "val_loss": val_loss})
                    logger.info(
                        "summary_epoch {}: train_loss={} val_loss = {} val_score={} best_val_score={} elapsed={} lr={}".format(
                            summary_epoch,
                            train_loss,
                            val_loss,
                            val_score,
                            best_score,
                            int(eval_time - summary_prev_time),
                            optimizer.param_groups[0]["lr"],
                        )
                    )

                del train_losses[:]
                summary_epoch += 1
                summary_prev_time = time.time()

                # update the learning rate
                if lr_scheduler and warmup_lr_scheduler is None:
                    lr_scheduler.step(train_loss)

                # Model saving.
                if ema_factor:
                    state_dict = model.shadow.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(save_path, "last.pth"))

                if best_score is None or val_score > best_score:
                    torch.save(state_dict, os.path.join(save_path, "best.pth"))
                    best_score = val_score
                    if task == "point":
                        # Log full val set confusion matrix
                        evaluator.log_metrics("class0", logger)
                    if task == "custom":
                        # Log full val set MAEs by attribute
                        evaluator.log_metrics(logger)
    run.finish()
    return None
