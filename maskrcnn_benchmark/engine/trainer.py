# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import random
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, is_main_process


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def get_vis_image(image_list, index):
    h, w = image_list.image_sizes[index]
    image = image_list.tensors[index, :h, :w]
    image = np.array(image.cpu()).transpose((1, 2, 0)).copy()
    return (image - image.min()) / (image.max() - image.min())


def overlay_box(image, box, color=(0, 1, 0)):
    box = box.to(torch.int64)
    top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
    # Draw a green rectangle
    visualized = cv2.rectangle(image, tuple(top_left), tuple(bottom_right),
                               (0, 1, 0), 3)
    return visualized


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    meters
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(
            loss=losses_reduced,
            time=batch_time,
            data=data_time,
            lr=optimizer.param_groups[0]["lr"],
            **loss_dict_reduced)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            # Log only certain keys to stdout + file, and rest only to file.
            log_to_stdout = set([
                'loss', 'time', 'loss_box_reg', 'loss_mask', 'loss_objectness'
            ])
            logging.info(
                meters.delimiter.join([
                    "eta: {eta}",
                    "iter: {iter}",
                    "{meters}",
                    "lr: {lr:.6f}",
                    "max mem: {memory:.0f}",
                ]).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=meters.to_log_str(log_keys=log_to_stdout),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                ))
            logging.debug(
                'Other meters: %s', meters.to_log_str(log_keys=log_to_stdout))
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        if (iteration < 5000 and (iteration == 1 or iteration % 1000 == 0)
                and is_main_process() and hasattr(meters, 'writer')):
            index = random.randrange(len(targets))
            image = get_vis_image(images, index)
            box = targets[index].convert('xyxy').bbox[0]
            meters.writer.add_image('groundtruth',
                                    overlay_box(image, box.cpu()),
                                    meters.iteration)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
