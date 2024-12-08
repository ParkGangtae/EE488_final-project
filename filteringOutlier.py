#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import time
import os
import argparse
import pdb
import glob
import datetime
import numpy as np
import logging
from EmbedNet import *
from DatasetLoader import get_data_loader
from sklearn import metrics
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "Face Recognition Training")

## Data loader
parser.add_argument('--batch_size',         type=int, default=64,	help='Batch size, defined as the number of classes per batch')
parser.add_argument('--max_img_per_cls',    type=int, default=500,	help='Maximum number of images per class per epoch')
parser.add_argument('--nDataLoaderThread',  type=int, default=5, 	help='Number of data loader threads')

## Training details
parser.add_argument('--test_interval',  type=int,   default=5,      help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=15,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="softmax",  help='Loss function to use')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='Optimizer')
parser.add_argument('--scheduler',      type=str,   default="cosine", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Initial learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.90,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')

## Loss functions
parser.add_argument('--margin',         type=float, default=0.2,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerClass',      type=int,   default=1,      help='Number of images per class per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=1250,   help='Number of classes in the softmax layer, only for softmax-based losses')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights, otherwise initialise with random weights')
parser.add_argument('--save_path',      type=str,   default="data/outlier_dict", help='Path for model and logs')

## Training and evaluation data
parser.add_argument('--train_path',     type=str,   default="data/ee488_24_data/train2",   help='Absolute path to the train set')
parser.add_argument('--train_ext',      type=str,   default="jpg",  help='Training files extension')

## Model definition
parser.add_argument('--model',          type=str,   default="ResNet18", help='Name of model definition')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')

## Training
parser.add_argument('--gpu',            type=int,   default=1,      help='GPU index')

args = parser.parse_args()

## ===== ===== ===== ===== ===== ===== ===== =====
## Sort by similarity
## ===== ===== ===== ===== ===== ===== ===== =====

def sort_images_by_similarity(train_path, model, transform, batch_size=64):

    dataset = datasets.ImageFolder(root=train_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.cuda()
            embeds = model(images)
            embeddings.append(embeds.cpu().numpy())
            labels.append(targets.cpu().numpy())

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    class_indices = {i: np.where(labels == i)[0] for i in np.unique(labels)}

    class_similarities = {}
    
    for class_idx, indices in class_indices.items():
        class_embeddings = embeddings[indices]
        mean_embedding = np.mean(class_embeddings, axis=0)

        similarities = np.dot(class_embeddings, mean_embedding) / (
            np.linalg.norm(class_embeddings, axis=1) * np.linalg.norm(mean_embedding)
        )

        sorted_indices = np.argsort(similarities)

        sorted_files = [dataset.samples[idx][0] for idx in indices[sorted_indices]]

        class_similarities[class_idx] = sorted_files

    return class_similarities, embeddings, labels

## ===== ===== ===== ===== ===== ===== ===== =====
## Filter low similarity
## ===== ===== ===== ===== ===== ===== ===== =====

def filter_low_similarity(sorted_image_dict, percentile=10):

    filtered_image_dict = {}
    
    for class_idx, image_list in sorted_image_dict.items():
        cutoff_index = int(len(image_list) * (percentile / 100))
        filtered_image_dict[class_idx] = image_list[:cutoff_index]

    return filtered_image_dict

## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(args):

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.save_path+"/scores.txt", mode="a+"),
        ],
        level=logging.DEBUG,
        format='[%(levelname)s] :: %(asctime)s :: %(message)s', 
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ## Load models
    model = EmbedNet(**vars(args)).cuda()

    ep          = 1

    ## Input transformations for training (you can change if you like)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(256),
         transforms.RandomCrop([224,224]),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Initialise trainer and data loader
    trainLoader = get_data_loader(transform=transform, **vars(args))
    trainer     = ModelTrainer(model, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('{}/epoch0*.model'.format(args.save_path))
    modelfiles.sort()

    ## If the target directory already exists, start from the existing file
    if len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        ep = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
    elif(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))

    ## If the current iteration is not 1, update the scheduler
    for ii in range(1,ep):
        trainer.__scheduler__.step()

    ## Print total number of model parameters
    pytorch_total_params = sum(p.numel() for p in model.__E__.parameters())
    print('Total model parameters: {:,}'.format(pytorch_total_params))

    ## Log arguments
    logger.info('{}'.format(args))

    ## Core training script
    for ep in range(ep,args.max_epoch+1):

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        logger.info("Epoch {:04d} started with LR {:.5f} ".format(ep,max(clr)))
        loss = trainer.train_network(trainLoader)
        logger.info("Epoch {:04d} completed with TLOSS {:.5f}".format(ep,loss))

        if ep == 15:

            sorted_image_dict, embeddings, labels = sort_images_by_similarity(
                args.train_path, 
                model.__E__,
                transform,
                batch_size=args.batch_size
            )

            filtered_image_dict = filter_low_similarity(
                sorted_image_dict, 
                percentile=15
            )
            
            with open(f"{args.save_path}/filtered_images.txt", "w") as f:
                for class_idx, image_list in filtered_image_dict.items():
                    f.write(f"Class {class_idx}:\n")
                    for image_path in image_list:
                        f.write("./" + image_path + "\n")
                    f.write("\n")
            
            logger.info(f"Filtered image list saved for epoch {ep} at {args.save_path}/filtered_images.txt")

    

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====

def main():

    os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(args.gpu)
            
    if not(os.path.exists(args.save_path)):
        os.makedirs(args.save_path)

    main_worker(args)


if __name__ == '__main__':
    main()