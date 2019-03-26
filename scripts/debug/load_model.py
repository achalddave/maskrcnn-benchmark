import logging
import torch

import argparse
from pathlib import Path

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.demo.predictor import COCODemo
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer


def load_model(config, model_path):
    model = build_detection_model(config)
    model.eval()
    device = torch.device(config.MODEL.DEVICE)
    model.to(device)
    checkpointer = DetectronCheckpointer(
        config, model, save_dir=config.OUTPUT_DIR)
    logging.info('Loading model from model-path: %s', model_path)
    load_path = model_path
    checkpointer.load(load_path, allow_override=False)
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--config-file',
        required=True,
        help='optional config file')
    parser.add_argument(
        '--model-path',
        type=Path,
        help=('Path to model pickle file. If not specified, the latest '
              'checkpoint, if it exists, or cfg.MODEL.WEIGHT is loaded.'))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # update the config options with the config file
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Avoid lots of logging when loading model.
    model = load_model(cfg, str(args.model_path.resolve()))
    __import__('ipdb').set_trace()

if __name__ == "__main__":
    main()
