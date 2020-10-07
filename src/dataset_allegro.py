from typing import List

import numpy as np
import torch

from allegroai import SingleFrame

from .dataset import KITTI2D


class AllegroDataset(KITTI2D):
    def __init__(self,
                 single_frames,  # type: List[SingleFrame]
                 image_size=(416, 416),
                 max_objects=50,
                 train=True):
        super().__init__(image_dir=None, label_dir=None,
                         image_size=image_size,
                         max_objects=max_objects,
                         fraction=1,
                         split_ratio=1,
                         train=train)
        self.single_frames = single_frames

    def _load_filenames(self):
        # bypassing the original function, we dont need it
        return

    def __getitem__(self, index):
        """
            Args
                index (int): Index

            Returns
                tuple: (image_paths, image, labels) where labels is a yolo vector of [max_objects x 5]

        """
        # Returns img_path, img(as PIL), bbox (as np array), labels (as np array)
        image = self._read_image(index)
        label = self._read_label(index)

        return self._get_img_path(index), image, label

    def __len__(self):
        """
            Returns
                size of the dataset
        """
        return len(self.single_frames)

    def _get_img_path(self, index):
        """
            Args
                index (int): Index

            Returns
                relative path of image
        """
        return self.single_frames[index].get_local_source()

    def _read_label(self, index):
        """
            Read the txt file corresponding to the label and output the label tensor
            following the yolo format [max_objects x 5]

            Args
                index (int): Index

            Returns
                Torch tensor that encodes the labels for the image

        """
        single_frame = self.single_frames[index]

        labels = None

        if single_frame.annotations:

            # list of tuples (label_num, x, y, w, h)
            labels = np.array([[a.label_enum, ] + list(a.get_bounding_box()) for a in single_frame.annotations
                               if a.label_enum is not None and a.label_enum >= 0 and hasattr(a, 'get_bounding_box')])

            # Access state variables
            w, h, pad, padded_h, padded_w = \
                self.state_variables["w"], self.state_variables["h"], \
                self.state_variables["pad"], self.state_variables["padded_h"], self.state_variables["padded_w"]

            # Extract coordinates for unpadded + unscaled image
            x1 = labels[:, 1]
            y1 = labels[:, 2]
            x2 = labels[:, 1] + labels[:, 3]
            y2 = labels[:, 2] + labels[:, 4]

            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]

            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] /= padded_w
            labels[:, 4] /= padded_h

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))

        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]

        filled_labels = torch.from_numpy(filled_labels)

        return filled_labels
