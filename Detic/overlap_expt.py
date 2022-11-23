import argparse
import multiprocessing as mp
from xxlimited import Str
import numpy as np
import os
import sys
import shutil
import matplotlib.pyplot as plt
import clip
import torch

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

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from PIL import Image

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


def convert_hex_to_words(in_dir, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for file in os.scandir(in_dir):
        matrix = np.load(file)
        word_matrix = np.empty((matrix.shape[0], matrix.shape[1]), dtype='object')
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                pixel = matrix[i, j]
                message = ''.join([chr(hex) for hex in pixel if hex>0])
                word_matrix[i, j] = message

        np.save(out_dir+file.name, word_matrix)


def get_descriptions_vocab(args):
    vocab = set([])
        
    for file in os.scandir(args.input+'word_screen_descriptions/'):
        for word in np.load(file, allow_pickle=True).flatten():
            vocab.add(word)
    
    return vocab


def save_bboxes_ground_truth_labels(args, out_dir, labels_dir):

    if os.path.exists('{}bbox_ground_truth_labels'.format(out_dir)):
        shutil.rmtree('{}bbox_ground_truth_labels'.format(out_dir))
    os.makedirs('{}bbox_ground_truth_labels'.format(out_dir))

    if os.path.exists('{}bbox_pred_classes'.format(out_dir)):
        shutil.rmtree('{}bbox_pred_classes'.format(out_dir))
    os.makedirs('{}bbox_pred_classes'.format(out_dir))

    label_to_i_dict = {label:i for i, label in enumerate(args.custom_vocabulary.split(','))}
    label_to_i_dict[''] = len(args.custom_vocabulary.split(','))
    print(label_to_i_dict)

    for filename in os.scandir('{}bboxes'.format(out_dir)):
        img_bboxes = np.load('{}bboxes/{}'.format(out_dir, filename.name))
        img_pred_classes = np.load('{}pred_classes/{}'.format(out_dir, filename.name))
        labels_mtrx = np.load('{}{}/{}'.format(args.input, labels_dir, filename.name))
        for counter in range(len(img_bboxes)):
            bbox = img_bboxes[counter]
            pred_class = img_pred_classes[counter]
            if labels_dir == 'screen_descriptions':
                x1, y1, x2, y2 = bbox
                x = int((x1+x2)/2)
                y = int((y1+y2)/2)
                label_arr = labels_mtrx[int(y/16), int(x/16)]
            else:
                label_arr = labels_mtrx
            string = ''.join([chr(hex) for hex in label_arr if hex>0])
            ground_truth_label = label_to_i_dict[string.split(',')[0]]
            ground_truth_label = np.array(ground_truth_label)
        
            np.save('{}bbox_ground_truth_labels/{}_{}.npy'.format(out_dir, filename.name.split('/')[-1][:-4], counter), ground_truth_label)
            np.save('{}bbox_pred_classes/{}_{}.npy'.format(out_dir, filename.name.split('/')[-1][:-4], counter), pred_class)

def get_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    height = y2-y1
    width = x2-x1
    return height * width

def get_overlap_area(pred_bbox, gt_bbox):  # returns None if rectangles don't intersect
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
    width = min(pred_x2, gt_x2) - max(pred_x1, gt_x1)
    height = min(pred_y2, gt_y2) - max(pred_y1, gt_y1)
    if (width>=0) and (height>=0):
        return width*height
    else:
        return 0

def main():
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    np.set_printoptions(threshold=sys.maxsize)

    screen_descriptions_dir = 'minihack_datasets/numerical_expts/screen_descriptions/'
    word_screen_descriptions_dir = 'minihack_datasets/numerical_expts/word_screen_descriptions/'

    continuous_areas = ['', 'dark part of a room', 'floor of a room', 'water']

    #convert_hex_to_words(screen_descriptions_dir, word_screen_descriptions_dir)
    #vocab = get_descriptions_vocab(args)
    #file = open('vocab.txt', 'w')
    #file.write(str(vocab))
    #file.close()
    vocab = {'', 'grid bug', 'lichen', 'a boulder', 'newt', 'jackal', 'a goblin corpse', 'sewer rat', 'dark part of a room', 'water', 'a newt corpse', 'kobold zombie', 'fox', 'a scroll labeled LOREM IPSUM', 'a lichen corpse', 'floor of a room', 'a kobold corpse', 'goblin', 'human rogue called Agent', 'staircase up'}
    for item in continuous_areas:
        if item in vocab:
            vocab.remove(item)
    args.vocabulary = 'custom'
    args.custom_vocabulary = ','.join(vocab)
    
    label_to_i_dict = {label:i for i, label in enumerate(args.custom_vocabulary.split(','))}
    i_to_label_dict = {i:label for label, i in label_to_i_dict.items()}

    item_gt_area = 16*16
    threshs = [0.2, 0.4, 0.6, 1]
    for thresh in threshs:
        print(thresh)
        print()
        args.confidence_threshold = thresh/100
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg, args, name="screen_description_experiment_all_items")
        for file in os.scandir(args.input+'pixels/'):
            print(file.name)
            file_num = str(file.name[:-4])
            img = read_image(file, format="RGB")
            gt_matrix = np.load(word_screen_descriptions_dir + file_num + '.npy', allow_pickle=True)

            predictions, visualized_output = demo.run_on_image(img)
            #visualized_output.save('out.jpg')
            bboxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
            print(len(bboxes))
            gt_lst = gt_matrix.flatten()
            gt_lst = gt_lst[np.isin(gt_lst, continuous_areas, invert=True)]
            unique, counts = np.unique(gt_lst, return_counts=True)
            cost_matrix = np.zeros((len(bboxes), np.sum(counts)))

            gt_bboxes = []

            for i in range(gt_matrix.shape[0]):
                for j in range(gt_matrix.shape[1]):
                    if gt_matrix[i, j] not in continuous_areas:
                        x1 = j * 16
                        x2 = x1 + 16
                        y1 = i * 16
                        y2 = y1 + 16
                        gt_bboxes.append((x1, y1, x2, y2))

            for i, pred_bbox in enumerate(bboxes):
                for j, gt_bbox in enumerate(gt_bboxes):
                    pred_bbox_area = get_bbox_area(pred_bbox)
                    overlap_area = get_overlap_area(pred_bbox, gt_bbox)
                    cost = overlap_area/max(pred_bbox_area, item_gt_area)
                    cost_matrix[i, j] = cost
            print(cost_matrix.shape)
            print(np.amax(cost_matrix, axis=1).shape)
            print(np.amax(cost_matrix, axis=1))

    '''
    plt.figure()
    plt.plot([thresh for thresh in range(0, 31, 2)], count_diff_avgs)
    plt.xlabel("confidence threshold %")
    plt.ylabel("num of bboxes returned by detic - num of items in gt image")
    plt.savefig('plot_diff_avgs_3.jpg')
    '''


    #single_items_count_accuracy(word_screen_descriptions_dir, pred_classes_dir, continuous_areas, i_to_label_dict)

    # For continuous surfaces
    '''
    for phrase in monsters_vocab_dicti:
        monsters_vocab_dicti[phrase] = numIslands(matrix.copy(), phrase)
    print(monsters_vocab_dicti)
    '''
        

main()