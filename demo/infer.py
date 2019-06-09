import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import pickle
from script_utils.common import common_setup
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.demo.predictor import COCODemo
from maskrcnn_benchmark.utils.parallel.fixed_gpu_pool import FixedGpuPool

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image(path):
    return path.is_file and any(path.suffix.lower() == extension
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


def init_model(init_args, context):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(context['gpu'])
    context['predictor'] = COCODemo(
        init_args['config'],
        model_path=init_args['model_path'],
        confidence_threshold=init_args['score_threshold'])


def infer(kwargs, context):
    coco_demo = context['predictor']
    image_path = kwargs['image_path']
    output_path = kwargs['output_path']
    visualize = kwargs['visualize']
    if output_path.exists():
        return
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
    # This specific format is for "legacy" reasons - I (@achald) used to
    # save the cls_boxes, cls_segms, and cls_keyps variables from the
    # Caffe2 detectron in a pickle file, and wrote a number of scripts that
    # operate on that format, so I save in the same format here.
    # 1. <https://github.com/facebookresearch/Detectron/blob/7c0ad88fc0d33cf0f698a3554ee842262d27babf/tools/infer.py#L146>  # noqa: E501
    # 2. <https://bitbucket.org/devalab/detectron-track-everything/src/7cf00d20900cc03eb485cd8924403bc4c18ccbf8/tools/infer_simple.py#lines-432>  # noqa: E501
    output = {
        'boxes': [[] for _ in coco_demo.CATEGORIES],
        'segmentations': [[] for _ in coco_demo.CATEGORIES],
        'keypoints': [[] for _ in coco_demo.CATEGORIES]
    }
    if predictions.has_field('mask'):
        rle_masks = convert_segmentation_coco_format(predictions)
    else:
        rle_masks = [None for _ in predictions.get_field('labels')]

    for i, label in enumerate(predictions.get_field('labels')):
        box = np.zeros(5)
        box[:4] = predictions.bbox[i]
        box[4] = predictions.get_field('scores')[i]
        output['boxes'][label].append(box)
        output['segmentations'][label].append(rle_masks[i])
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    if visualize:
        top_predictions = coco_demo.select_top_predictions(predictions)
        result = image.copy()
        result = coco_demo.overlay_boxes(result, top_predictions)
        if coco_demo.cfg.MODEL.MASK_ON:
            result = coco_demo.overlay_mask(result, top_predictions)
        if coco_demo.cfg.MODEL.KEYPOINT_ON:
            result = coco_demo.overlay_keypoints(result, top_predictions)
        result = coco_demo.overlay_class_names(result, top_predictions)
        cv2.imwrite(str(output_path.with_suffix('.jpg')), result)


def main():
    """Parse in command line arguments"""
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
    parser.add_argument('--gpus', default=[0], nargs='*', type=int)

    parser.add_argument('--score-thresh', default=0.7, type=float)
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
        '--recursive',
        help='Whether to search recursively in --image-dir for images.',
        action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument(
        '--threads-per-worker',
        type=int,
        help='Defaults to 1 if using multiple gpus, otherwise unrestricted.')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    if len(args.gpus) > 1 and args.threads_per_worker is None:
        args.threads_per_worker = 1
    if args.threads_per_worker is not None:
        torch.set_num_threads(args.threads_per_worker)
        os.environ['OMP_NUM_THREADS'] = str(args.threads_per_worker)

    # update the config options with the config file
    cfg.merge_from_file(args.config_file)
    if len(args.gpus) > 1 and 'DATALOADER.NUM_WORKERS' not in args.opts:
        args.opts += ['DATALOADER.NUM_WORKERS', '1']
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.image_dir:
        logging.info('Collecting images')
        images = [x for x in args.image_dir.iterdir() if is_image(x)]
        if args.recursive:
            # We could just use args.images_dir.rglob('*'), but this ignores
            # symlinks. For convenience, we handle immediate child directories
            # being symlinks.
            for x in args.image_dir.iterdir():
                if x.is_dir():
                    images.extend([y for y in x.rglob('*') if is_image(y)])
        outputs = [(args.output_dir / x.relative_to(
            args.image_dir)).with_suffix('.pickle') for x in images]
    else:
        images = args.images
        outputs = [(args.output_dir / (x.stem + '.pickle')) for x in images]

    if not images:
        raise ValueError('No images found!')
    logging.info('Inferring on %s images', len(images))

    init_args = {
        'model_path': str(args.model_path.resolve()),
        'config': cfg,
        'score_threshold': args.score_thresh
    }
    infer_tasks = [{
        'image_path': image,
        'output_path': output,
        'visualize': args.visualize
    } for image, output in zip(images, outputs)]
    if len(args.gpus) == 1:
        context = {'gpu': args.gpus[0]}
        init_model(init_args, context)
        for task in tqdm(infer_tasks):
            infer(task, context)
    else:
        pool = FixedGpuPool(
            args.gpus, initializer=init_model, initargs=init_args)
        list(
            tqdm(
                pool.imap_unordered(infer, infer_tasks),
                total=len(infer_tasks)))


if __name__ == "__main__":
    main()
