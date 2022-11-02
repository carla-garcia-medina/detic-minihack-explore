# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp
from xxlimited import Str
import numpy as np
import os
import sys
import shutil

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo

# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    else:
        cfg.MODEL.DEVICE = "cuda"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    input_dir = 'minihack_datasets/MiniHack-River-Monster-v0/dataset_0/'
    output_dir = 'outputs/'

    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=input_dir,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default=output_dir,
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.03,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def lvis_experiment(args):
    out_dir = os.path.join(args.output, "lvis_experiment/")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    args.confidence_threshold = 0.5
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args, name="lvis_experiment")
    
    for counter in range(len(os.listdir(args.input+'pixels/'))):
        img_path = '{}.jpg'.format(counter)
        
        img = read_image(args.input+'pixels/' + img_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        out_filename = os.path.join(out_dir, os.path.basename(img_path))
        visualized_output.save(out_filename)

def screen_description_experiment_all_items(args):
    out_dir = os.path.join(args.output, "screen_description_expts_all_items/")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    vocab = set([])

    for counter in range(len(os.listdir(args.input+'pixels/'))):
        for row in np.load('{}screen_descriptions/{}.npy'.format(args.input, counter)).reshape(1659, 80):
            description = ''.join([chr(hex) for hex in row if hex>0])
            if len(description) > 0:
                vocab.add(description)
    
    args.vocabulary = 'custom'
    args.custom_vocabulary = ','.join(vocab)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args, name="screen_description_experiment_all_items")
    
    for counter in range(len(os.listdir(args.input+'pixels/'))):
        img_path = '{}.jpg'.format(counter)
        
        img = read_image(args.input+'pixels/' + img_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        out_filename = os.path.join(out_dir, os.path.basename(img_path))
        visualized_output.save(out_filename)

def screen_description_experiment_items_in_image(args):
    out_dir = os.path.join(args.output, "screen_description_expts_items_in_image/")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
        
    for counter in range(len(os.listdir(args.input+'pixels/'))):
        vocab = set([])
        for row in np.load('{}screen_descriptions/{}.npy'.format(args.input, counter)).reshape(1659, 80):
            description = ''.join([chr(hex) for hex in row if hex>0])
            if len(description) > 0:
                vocab.add(description)

        args.vocabulary = 'custom'
        args.custom_vocabulary = ','.join(vocab)
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg, args, name="screen_description_experiment_items_in_image_" + str(counter))

        img_path = '{}.jpg'.format(counter)
        
        img = read_image(args.input+'pixels/' + img_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        out_filename = os.path.join(out_dir, os.path.basename(img_path))
        visualized_output.save(out_filename)

def all_message_experiment(args):
    out_dir = os.path.join(args.output, "all_message_experiment/")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    vocab = set([])

    for counter in range(len(os.listdir(args.input+'pixels/'))):
        message = ''.join([chr(hex) for hex in np.load('{}messages/{}.npy'.format(args.input, counter)) if hex>0])
        if len(message) > 0:
            vocab.add(message)

    args.vocabulary = 'custom'
    args.custom_vocabulary = ','.join(vocab)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args, name="all_message_experiment")

    for counter in range(len(os.listdir(args.input+'pixels/'))):
        img_path = '{}.jpg'.format(counter)
        img = read_image(args.input+'pixels/' + img_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        print(predictions)
        out_filename = os.path.join(out_dir, os.path.basename(img_path))
        visualized_output.save(out_filename)

def corresponding_message_experiment(args):
    out_dir = os.path.join(args.output, "corresponding_message_experiment/")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    args.vocabulary = 'custom'
    args.confidence_threshold = 0.01
    
    file = open(out_dir + "confidence_scores.txt", "w")
    for counter in range(len(os.listdir(args.input+'pixels/'))):
        message = ''.join([chr(hex) for hex in np.load('{}messages/{}.npy'.format(args.input, counter)) if hex>0])
        args.custom_vocabulary = message
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg, args, name="corresponding_message_experiment" + str(counter))

        img_path = '{}.jpg'.format(counter)
        img = read_image(args.input+'pixels/' + img_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        file.writelines("{0} {1} {2}\n".format(counter, predictions['instances'].scores.cpu().numpy()[0], args.custom_vocabulary))
        out_filename = os.path.join(out_dir, os.path.basename(img_path))
        visualized_output.save(out_filename)
    file.close()

def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    
    screen_description_experiment_all_items(args)
    #screen_description_experiment_items_in_image(args)
    #all_message_experiment(args)
    #corresponding_message_experiment(args)
    #lvis_experiment(args)


if __name__ == '__main__':
    main()