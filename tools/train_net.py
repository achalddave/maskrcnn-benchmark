# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import logging
import os
from pathlib import Path

from script_utils.common import common_setup

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def _safe_int(x):
    if isinstance(x, float):
        assert x.is_integer(), ('%s is not an integer.' % x)
        return int(x)
    elif isinstance(x, int):
        return int(x)
    else:
        raise ValueError('Unknown type: %s (%s)' % (x, type(x)))


def train(cfg, local_rank, distributed):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        '--reduce-batch',
        type=int,
        help=('Divide IMS_PER_BATCH by this amount. This appropriately '
              'updates the learning rate, number of iterations, and so on.'))
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)

    # We want to get the OUTPUT_DIR from the args immediately, if it exists, so
    # we can setup logging. We will merge the rest of the config in a few
    # lines.
    try:
        dir_index = args.opts[0::2].index('OUTPUT_DIR')
        cfg.OUTPUT_DIR = args.opts[dir_index * 2 + 1]
    except ValueError:
        pass

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    if get_rank() == 0:
        file_logger = common_setup(__file__, output_dir, args)
    else:
        file_logger = common_setup(
            __file__ + '-worker%s' % get_rank(),
            output_dir,
            args,
            log_console_level=logging.CRITICAL,
            save_git_state=False)

    # Automatically handle config changes as required by
    # https://github.com/facebookresearch/maskrcnn-benchmark/tree/327bc29bcc4924e35bd61c59877d5a1d25bb75af#single-gpu-training
    if args.reduce_batch:
        assert num_gpus in (1, 2, 4)
        scale = args.reduce_batch
        logging.info('Updating config for # GPUs = %s', num_gpus)

        def update_config(key, new_value):
            key_list = key.split('.')
            d = cfg
            for subkey in key_list[:-1]:
                d = cfg[subkey]
            subkey = key_list[-1]
            old_value = d[subkey]
            logging.info('Updating cfg.%s: %s -> %s', key, old_value,
                         new_value)
            d[subkey] = new_value
        update_config('SOLVER.IMS_PER_BATCH',
                      _safe_int(cfg.SOLVER.IMS_PER_BATCH / scale))
        update_config('SOLVER.BASE_LR',
                      cfg.SOLVER.BASE_LR / scale)
        update_config('SOLVER.MAX_ITER',
                      _safe_int(cfg.SOLVER.MAX_ITER * scale))
        update_config('SOLVER.STEPS',
                      tuple(_safe_int(x * scale) for x in cfg.SOLVER.STEPS))

    logging.info('Updating config from arguments')
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logging.info('Config: %s', cfg)
    with open(Path(output_dir) / 'config.yaml', 'wb') as f:
        f.write(cfg.dump())

    logger = logging.getLogger("maskrcnn_benchmark")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        file_logger.info(config_str)
    file_logger.info("Running with config:\n{}".format(cfg))
    if get_rank() == 0:
        with open(Path(output_dir) / 'config.yaml', 'w') as f:
            f.write(cfg.dump())

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
