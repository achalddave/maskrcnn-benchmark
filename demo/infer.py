import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pickle
from script_utils.common import common_setup
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.build import build_dataset
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.utils.imports import import_file
from predictor import COCODemo

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image(path):
    return path.is_file and any(path.suffix == extension
                                for extension in IMG_EXTENSIONS)


def convert_segmentation_coco_format(prediction):
    """Modified from
    data.datasets.evaluation.coco.coco_eval.prepare_for_coco_segmentation"""
    import pycocotools.mask as mask_util
    masks = prediction.get_field('mask')
    rles = [
        mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
        for mask in masks
    ]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")
    return rles


def main():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(
        description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        required=True,
        help='optional config file')
    parser.add_argument(
        '--model-path',
        help=('Path to model pickle file. If not specified, the latest '
              'checkpoint, if it exists, or cfg.MODEL.WEIGHT is loaded.'))

    parser.add_argument(
        '--image-dir',
        type=Path,
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        type=Path,
        help=('images to infer. Must not use with --image_dirs. If the model '
              'requires multiple input datasets, use --image_dirs instead.'))
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--dataset',
        default='coco_2017_train')
    parser.add_argument(
        '--recursive',
        help='Whether to search recursively in --image-dir for images.',
        action='store_true')

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    # update the config options with the config file
    cfg.merge_from_file(args.config_file)

    # Avoid lots of logging when loading model.
    logging.root.setLevel(logging.WARN)
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7)
    if args.model_path:
        logging.info('Loading model from: %s', args.model_path)
        coco_demo.load_model(args.model_path)
    logging.root.setLevel(logging.INFO)

    if args.image_dir:
        logging.info('Collecting images')
        if args.recursive:
            images = [x for x in args.image_dir.rglob('*') if is_image(x)]
        else:
            images = [x for x in args.image_dir.iterdir() if is_image(x)]
        outputs = [(args.output_dir / x.relative_to(
            args.image_dir)).with_suffix('.pickle') for x in images]
    else:
        images = args.images
        outputs = [(args.output_dir / (x.stem + '.pickle')) for x in images]

    if not images:
        raise ValueError('No images found!')
    logging.info('Inferring on %s images', len(images))
    for image_path, output_path in zip(tqdm(images), outputs):
        if output_path.exists():
            continue
        image = cv2.imread(str(image_path))
        predictions = coco_demo.compute_prediction(image)

        # Output dictionary containing keys 'boxes', 'segmentations',
        # 'keypoints':
        # - 'boxes': List of length num_classes+1, where each element is a
        #   (num_boxes, 5) array containing (x0, y0, x1, y1, score) for each
        #   box.
        # - 'segmentations': List of length num_classes+1, where each element
        #   is a list of length num_boxes, containing an RLE encoded mask.
        # - 'keypoints': List of length num_classes+1, each containing
        #   keypoints for each box. (Unused right now)
        #
        # This is for "legacy" reasons - I (@achald) used to save the
        # cls_boxes, cls_segms, and cls_keyps variables from the Caffe2
        # detectron in a pickle file, and wrote a number of scripts that
        # operate on that format, so I save in the same format here.
        # 1. <https://github.com/facebookresearch/Detectron/blob/7c0ad88fc0d33cf0f698a3554ee842262d27babf/tools/infer.py#L146>  # noqa: E501
        # 2. <https://bitbucket.org/devalab/detectron-track-everything/src/7cf00d20900cc03eb485cd8924403bc4c18ccbf8/tools/infer_simple.py#lines-432>  # noqa: E501
        output = {
            'boxes': [[] for _ in coco_demo.CATEGORIES],
            'segmentations': [[] for _ in coco_demo.CATEGORIES],
            'keypoints': [[] for _ in coco_demo.CATEGORIES]
        }
        rle_masks = convert_segmentation_coco_format(predictions)

        for i, label in enumerate(predictions.get_field('labels')):
            box = np.zeros(5)
            box[:4] = predictions.bbox[i]
            box[4] = predictions.get_field('scores')[i]
            output['boxes'][label].append(box)
            output['segmentations'][label].append(rle_masks[i])
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)

        top_predictions = coco_demo.select_top_predictions(predictions)
        result = image.copy()
        result = coco_demo.overlay_boxes(result, top_predictions)
        if coco_demo.cfg.MODEL.MASK_ON:
            result = coco_demo.overlay_mask(result, top_predictions)
        if coco_demo.cfg.MODEL.KEYPOINT_ON:
            result = coco_demo.overlay_keypoints(result, top_predictions)
        result = coco_demo.overlay_class_names(result, top_predictions)
        cv2.imwrite(str(args.output_dir / image_path.name), result)


if __name__ == "__main__":
    main()
