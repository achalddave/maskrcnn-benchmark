# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import logging
import os
from pathlib import Path

import torch
from script_utils.common import common_setup
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        help=('Path to model pickle file. If not specified, the latest '
              'checkpoint, if it exists, or cfg.MODEL.WEIGHT is loaded.'))
    parser.add_argument(
        '--output-dir',
        default='{cfg_OUTPUT_DIR}/inference-{model_stem}',
        help=('Output directory. Can use variables {cfg_OUTPUT_DIR}, which is '
              'replaced by cfg.OUTPUT_DIR, and {model_stem}, which is '
              'replaced by the stem of the file used to load weights.'))
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    assert cfg.OUTPUT_DIR, 'cfg.OUTPUT_DIR must not be empty.'
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    if args.model_path:
        load_path = str(args.model_path.resolve())
        load_msg = 'Loading model from --model-path: %s' % load_path
    else:
        if checkpointer.has_checkpoint():
            load_path = checkpointer.get_checkpoint_file()
            load_msg = 'Loading model from latest checkpoint: %s' % load_path
        else:
            load_path = cfg.MODEL.WEIGHT
            load_msg = 'Loading model from cfg.MODEL.WEIGHT: %s' % load_path

    output_dir = Path(args.output_dir.format(
        cfg_OUTPUT_DIR=cfg.OUTPUT_DIR, model_stem=Path(load_path).stem))
    output_dir.mkdir(exist_ok=True, parents=True)
    file_logger = common_setup(__file__, output_dir, args)
    # We can't log the load_msg until we setup the output directory, but we
    # can't get the output directory until we figure out which model to load.
    # So we save load_msg and log it here.
    logging.info(load_msg)
    logging.info('Output inference results to: %s' % output_dir)

    logger = logging.getLogger("maskrcnn_benchmark")
    logger.info("Using {} GPUs".format(num_gpus))
    file_logger.info('Config:')
    file_logger.info(cfg)

    file_logger.info("Collecting env info (might take some time)")
    file_logger.info("\n" + collect_env_info())

    checkpointer.load(load_path, allow_override=False)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    for idx, dataset_name in enumerate(dataset_names):
        output_folder = output_dir / dataset_name
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


if __name__ == "__main__":
    main()
