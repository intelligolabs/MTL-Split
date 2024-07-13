#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import torch
import itertools

import numpy as np

from torch.utils.data import Dataset
from utils.transformations import SaltAndPepperNoise


class Shapes3D(Dataset):
    """
    Shapes3D dataset.

    The ground-truth factors of variation are:
    0 - floor color (10 different values)
    1 - wall color (10 different values)
    2 - object color (10 different values)
    3 - object size (8 different values)
    4 - object type (4 different values)
    5 - azimuth (15 different values)
    """

    with h5py.File('DATASET_PATH_HERE', 'r') as dataset:
        images = dataset['images'][()]
        labels = dataset['labels'][()]
    n_samples = images.shape[0]
    images = images.reshape([n_samples, 64, 64, 3]).astype(np.float32) / 255.
    labels = labels.reshape([n_samples, 6])

    def __init__(self, dataset_path, transforms, partition):
        super().__init__()

        self.partition = partition
        self.transforms = transforms
        self.dataset_path = dataset_path

        images = Shapes3D.images
        labels = Shapes3D.labels

        factor_sizes = [10, 10, 10, 8, 4, 15]
        integer_labels = np.array([list(i) for i in itertools.product(*[range(x) for x in factor_sizes])])  # 480000.

        # Reduce dataset size.
        # factor_sizes = np.array([f//2 for f in factor_sizes])
        # factor_sizes = factor_sizes.reshape(1, -1)
        # reduced_idx = np.where(np.all(integer_labels < factor_sizes, axis=1))[0]
        # images = images[reduced_idx]
        # integer_labels = integer_labels[reduced_idx]                                                        # 7000.

        self.noise = SaltAndPepperNoise(noiseType="SnPn", treshold=0.08)

        # Prepare dataset specific information.
        y1_lbls = integer_labels[:, 3]          # Object size.
        y2_lbls = integer_labels[:, 4]          # Object type.

        self.images = images
        self.y = np.stack([y1_lbls, y2_lbls]).T

        # MTL specific information.
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [8,   # Object_size.
                               4    # Object_type.
                               ]

        # Split dataset into train and test.
        split_factor_idx = [0, 1]
        num_test_set_exclusive = [5, 5]
        subset_factor_sizes = np.array(factor_sizes).flatten()

        if split_factor_idx and num_test_set_exclusive:
            print(f"Dataset splitted on factors {split_factor_idx} by {num_test_set_exclusive}")
            assert not np.in1d(3, split_factor_idx).any() and len(split_factor_idx) == len(num_test_set_exclusive)
            subset_factor_sizes[split_factor_idx] -= num_test_set_exclusive

        subset_idxs = None
        if self.partition == 'train':
            subset_idxs = np.where(
                (integer_labels[:, split_factor_idx] < subset_factor_sizes[split_factor_idx]).sum(1) > 0
                )[0]
        else:
            subset_idxs = np.where(
                (integer_labels[:, split_factor_idx] < subset_factor_sizes[split_factor_idx]).sum(1) == 0
                )[0]

        self.images = self.images[subset_idxs]
        self.y = self.y[subset_idxs]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        img = torch.tensor(self.images[index])
        labels = torch.tensor(self.y[index])

        self.noise(img)

        return img.permute(2,0,1), labels
