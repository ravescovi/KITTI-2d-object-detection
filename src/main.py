# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:33:09 2019

@author: user
"""

from __future__ import division

import csv
import os
import warnings

import torch
from torch.utils.data import DataLoader

from src.dataset import KITTI2D
from src.dataset_allegro import AllegroDataset
from src.model import Darknet
from src.train_model import train_model
from allegroai import DataView, Task
from allegroai.dataview import SingleFrame

warnings.filterwarnings("ignore")


def main(train_path="../data/train/images/",
         val_path="../data/train/images/",
         labels_path="../data/train/yolo_labels/",
         weights_path="../checkpoints/",
         preload_weights_file="best_weights_kitti.pth",
         output_path="../output",
         yolo_config_file="../config/yolov3-kitti.cfg",
         fraction=0.8,
         learning_rate=1e-3,
         weight_decay=1e-4,
         batch_size=2,
         epochs=30,
         freeze_struct=[True, 5]):
    """
        This is the point of entry to the neural network program.
        All the training history will be saved as a csv in the output path
        
        Args
            train_path (string): Directory containing the training images
            val_path (string):: Directory containing the val images
            labels_path (string):: Directory containing the yolo format labels for data
            weights_path (string):: Directory containing the weights (new weights for this program will also be added
            here)
            preload_weights_file (string): Name of preload weights file
            output_path (string): Directory to store the training history outputs as csv
            yolo_config_file (string): file path of yolo configuration file
            fraction (float): fraction of data to use for training
            learning_rate (float): initial learning rate
            weight_decay (float): weight decay value
            batch_size (int): batch_size for both training and validation
            epochs (int): maximum number of epochs to train the model
            freeze_struct (list): [bool, int] indicating whether to freeze the Darknet backbone and until which epoch
            should it be frozen
            
        Returns
            None
    
    """

    # Load dataview
    dataview = DataView()
    dataview.add_query(
        dataset_name='KITTI 2D', version_name='training and validation', roi_query=['training'])
    dataview.set_labels({
        'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7
    })
    train_list = dataview.to_list()

    dataview = DataView('validation')
    dataview.add_query(
        dataset_name='KITTI 2D', version_name='training and validation', roi_query=['validation'])
    dataview.set_labels({
        'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7
    })
    validation_list = dataview.to_list()

    print("Dataview split: training {} images, validation {} images".format(len(train_list), len(validation_list)))

    # Set up checkpoints path
    checkpoints_path = weights_path

    # Set up env variables and create required directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # Set up cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)

    # Create model and load pretrained darknet weights
    model = Darknet(yolo_config_file)
    print("Loading imagenet weights to darknet")
    try:
        model.load_weights(os.path.join(weights_path, preload_weights_file))
    except FileNotFoundError:
        print('Skipping loading weights file, {} could not be found'.format(
            os.path.join(weights_path, preload_weights_file)))
    model.to(device)
    # print(model)

    # create torch Datasets
    train_dataset = AllegroDataset(train_list, train=True)
    valid_dataset = AllegroDataset(validation_list, train=False)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Create optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                                 weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

    # Create log csv files
    train_log_file = open(os.path.join(output_path, "train_results.csv"), "w", newline="")
    valid_log_file = open(os.path.join(output_path, "valid_results.csv"), "w", newline="")
    train_csv = csv.writer(train_log_file)
    valid_csv = csv.writer(valid_log_file)

    print("Starting to train yolov3 model...")

    # Train model here
    train_model(model,
                device,
                optimizer,
                lr_scheduler,
                train_dataloader,
                valid_dataloader,
                train_csv,
                valid_csv,
                weights_path,
                max_epochs=epochs,
                tensor_type=torch.cuda.FloatTensor,
                update_gradient_samples=1,
                freeze_darknet=freeze_struct[0],
                freeze_epoch=freeze_struct[1])

    # Close the log files
    train_log_file.close()
    valid_log_file.close()

    print("Training completed")


if __name__ == "__main__":
    task = Task.init('example', 'training')
    kwargs = dict(
        train_path="./data/train/images/",
        val_path="./data/train/images/",
        labels_path="./data/train/yolo_labels/",
        weights_path="./checkpoints/",
        preload_weights_file="best_weights_kitti.pth",
        output_path="./output",
        yolo_config_file="./config/yolov3-kitti.cfg",
        fraction=0.8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=2,
        epochs=30,
        freeze_struct=[False, 0])
    main(**kwargs)
