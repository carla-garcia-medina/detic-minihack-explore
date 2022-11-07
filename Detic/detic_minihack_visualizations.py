# Copyright (c) Facebook, Inc. and its affiliates.
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

def get_vocab(args):
    vocab = set([])
        
    for counter in range(len(os.listdir(args.input+'pixels/'))):
        for row in np.load('{}screen_descriptions/{}.npy'.format(args.input, counter)).reshape(1659, 80):
            description = ''.join([chr(hex) for hex in row if hex>0])
            if len(description) > 0:
                vocab.add(description)
    
    return vocab

def save_bboxes_ground_truth_labels(args, out_dir):
    if os.path.exists('{}bbox_ground_truth_labels'.format(out_dir)):
        shutil.rmtree('{}bbox_ground_truth_labels'.format(out_dir))
    os.makedirs('{}bbox_ground_truth_labels'.format(out_dir))
    if os.path.exists('{}bbox_pred_classes'.format(out_dir)):
        shutil.rmtree('{}bbox_pred_classes'.format(out_dir))
    os.makedirs('{}bbox_pred_classes'.format(out_dir))

    label_to_i_dict = {label:i for i, label in enumerate(args.custom_vocabulary.split(','))}
    label_to_i_dict[''] = len(args.custom_vocabulary.split(','))

    for filename in os.scandir('{}bboxes'.format(out_dir)):
        img_bboxes = np.load('{}bboxes/{}'.format(out_dir, filename.name))
        img_pred_classes = np.load('{}pred_classes/{}'.format(out_dir, filename.name))
        labels_mtrx = np.load('{}screen_descriptions/{}'.format(args.input, filename.name))
        for counter in range(len(img_bboxes)):
            bbox = img_bboxes[counter]
            pred_class = img_pred_classes[counter]
            x1, y1, x2, y2 = bbox
            x = int((x1+x2)/2)
            y = int((y1+y2)/2)
            label_arr = labels_mtrx[int(y/16), int(x/16)]
            ground_truth_label = label_to_i_dict[''.join([chr(hex) for hex in label_arr if hex>0])]
            ground_truth_label = np.array(ground_truth_label)
        
            np.save('{}bbox_ground_truth_labels/{}_{}.npy'.format(out_dir, filename.name.split('/')[-1][:-4], counter), ground_truth_label)
            np.save('{}bbox_pred_classes/{}_{}.npy'.format(out_dir, filename.name.split('/')[-1][:-4], counter), pred_class)

def save_predictions(args, out_dir):
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args, name="screen_description_experiment_all_items")

    os.makedirs('{}bboxes/'.format(out_dir))
    os.makedirs('{}pred_classes/'.format(out_dir))

    for counter in range(len(os.listdir(args.input+'pixels/'))):
        img_path = '{}.jpg'.format(counter)
        img = read_image(args.input+'pixels/' + img_path, format="RGB")

        predictions, _ = demo.run_on_image(img)
        bboxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
        pred_classes = predictions['instances'].pred_classes.cpu().numpy()

        np.save('{}bboxes/{}.npy'.format(out_dir, counter), bboxes)
        np.save('{}pred_classes/{}.npy'.format(out_dir, counter), pred_classes)

def save_preprocessed_imgs(args, out_dir):
    if args.cpu:
        device="cpu"
    else:
        device = "cuda"
    _, preprocess = clip.load('ViT-B/32', device)

    if os.path.exists('{}preprocessed_imgs/'.format(out_dir)):
        shutil.rmtree('{}preprocessed_imgs/'.format(out_dir))
    os.makedirs('{}preprocessed_imgs/'.format(out_dir))

    for img_counter in range(len(os.listdir(args.input+'pixels/'))):
        print(img_counter)
        img_path = '{}.jpg'.format(img_counter)
        img = read_image(args.input+'pixels/' + img_path, format="RGB")

        bboxes = np.load('{}bboxes/{}.npy'.format(out_dir, img_counter))

        for bbox_counter in range(len(bboxes)):
            bbox = bboxes[bbox_counter]
            x1, y1, x2, y2 = bbox.astype(int)
            
            # crop boxes from images
            cropped = img[y1:y2, x1:x2]

            # encode images
            cropped = Image.fromarray(np.uint8(cropped)).convert('RGB')
            # preprocess imgs
            preprocessed_img = preprocess(cropped).unsqueeze(0).to(device).cpu().numpy()
            
            np.save('{}preprocessed_imgs/{}_{}.npy'.format(out_dir, img_counter, bbox_counter), preprocessed_img)

def save_img_features(args, out_dir):
    if args.cpu:
        device="cpu"
    else:
        device = "cuda"

    if os.path.exists('{}img_features'.format(out_dir)):
        shutil.rmtree('{}img_features'.format(out_dir))
    os.makedirs('{}img_features'.format(out_dir))

    model, _ = clip.load('ViT-B/32', device)

    for filename in os.scandir('{}preprocessed_imgs/'.format(out_dir)):
        preprocessed_img = np.load(filename)
        img_features = model.encode_image(torch.tensor(preprocessed_img).to(device))
        img_features = torch.nn.functional.normalize(img_features, p=2.0, dim = 1)
        img_features = img_features.detach().cpu().numpy().flatten()
        np.save('{}img_features/{}'.format(out_dir, filename.name), img_features)

def get_correct_and_total(pred_classes, labels):
    correct = 0
    for i in range(len(pred_classes)):
        if labels[i] == int(pred_classes[i]):
            correct += 1
        print(pred_classes[i], labels[i])

    return correct, len(pred_classes)

def screen_description_experiment_all_items(args, dim_red = PCA(2), dim_red_name = 'PCA'):
    #IncrementalPCA(n_components=2, batch_size = 10000)

    out_dir = os.path.join(args.output, "screen_description_expts_all_items/")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    #vocab = get_vocab(args)

    vocab = {'grid bug', 'a kobold corpse', 'a scroll labeled LOREM IPSUM', 'water', 'lichen', 'floor of a room', 'goblin', 'a boulder', 'dark part of a room', 'kobold zombie', 'sewer rat', 'newt', 'a goblin corpse', 'human rogue called Agent', 'staircase up', 'fox', 'jackal', 'a newt corpse', 'a lichen corpse'}
    print(len(vocab))
    print(vocab)

    args.vocabulary = 'custom'
    args.custom_vocabulary = ','.join(vocab)
    
    # plot text embeddings

    # tokenize experiment prompts
    if args.cpu:
        device="cpu"
    else:
        device = "cuda"

    model, _ = clip.load('ViT-B/32', device)

    prompts = list(vocab)
    
    tokenized_prompts = [clip.tokenize(prompt).to(device) for prompt in prompts]

    prompt_features_lst = []
    for tp in tokenized_prompts:
        with torch.no_grad():
            prompt_features = model.encode_text(tp) # pass tokens through CLIP encoder to get prompt features
            prompt_features /= prompt_features.norm(dim=-1, keepdim=True) # normalize features
            for pf in prompt_features:
                prompt_features_lst.append(pf.cpu().data.numpy())

    #save_predictions(args, out_dir)
    #save_preprocessed_imgs(args, out_dir)
    #save_img_features(args, out_dir)
    #save_bboxes_ground_truth_labels(args, out_dir)
            
    for counter, filename in enumerate(os.scandir('{}img_features/'.format(out_dir))):
        img_features = np.load(filename)
        gt_label = np.load('{}bbox_ground_truth_labels/{}'.format(out_dir, filename.name))
        pred_class = np.load('{}bbox_pred_classes/{}'.format(out_dir, filename.name))
        if counter > 0:
            img_features_lst = np.vstack([img_features_lst, img_features])
            gt_labels_lst = np.vstack([gt_labels_lst, gt_label])
            pred_classes_lst = np.vstack([pred_classes_lst, pred_class])
        else:
            img_features_lst = img_features
            gt_labels_lst = np.array([gt_label])
            pred_classes_lst = np.array([pred_class])
    
    # perform dimensionality reduction
    reduced_prompt_features_lst = dim_red.fit_transform(prompt_features_lst)
    reduced_img_features_lst = dim_red.transform(img_features_lst)

    if os.path.exists('{}embedding_plots'.format(out_dir)):
        shutil.rmtree('{}embedding_plots'.format(out_dir))
    os.makedirs('{}embedding_plots'.format(out_dir))

    # plot embedding spaces color-coded by features
    plt.figure(figsize=(50, 50), dpi=1000)
    plt.ylim([-0.5, 0.5])
    plt.xlim([-0.5, 1])
    for x, y, prompt in zip(reduced_prompt_features_lst[:,0], reduced_prompt_features_lst[:,1], prompts):
        plt.text(x, y, prompt)
    plt.scatter(reduced_img_features_lst[:,0], reduced_img_features_lst[:,1], c = pred_classes_lst, s = 10)
    plt.colorbar()
    plt.savefig('{}embedding_plots/pred_classes.png'.format(out_dir, dim_red_name))

    # calculate acuracy
    correct, total = 0, 0

    new_correct, new_total = get_correct_and_total(pred_classes_lst.flatten(), gt_labels_lst.flatten())
    correct += new_correct
    total += new_total
    print(correct)
    print(total)
    print('Accuracy:', correct/total)

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

    print(len(vocab))

    args.vocabulary = 'custom'
    args.custom_vocabulary = ','.join(vocab)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args, name="all_message_experiment")

    for counter in range(len(os.listdir(args.input+'pixels/'))):
        img_path = '{}.jpg'.format(counter)
        img = read_image(args.input+'pixels/' + img_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        #print(predictions)
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
    os.makedirs('embedding_plot', exist_ok=True)
    screen_description_experiment_all_items(args)
    #screen_description_experiment_items_in_image(args)
    #all_message_experiment(args)
    #corresponding_message_experiment(args)
    #lvis_experiment(args)


if __name__ == '__main__':
    main()