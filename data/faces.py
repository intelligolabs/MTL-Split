#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Faces(Dataset):
    def __init__(self, dataset_path, transforms, partition):
        super().__init__()

        self.partition = partition
        self.transforms = transforms
        self.dataset_path = dataset_path

        images = []
        # ids = []
        ages = []
        genders = []
        expressions = []
        # picture_sets = []

        # Read person age, gender and expression from filename.
        for img in sorted(os.listdir(os.path.join(self.dataset_path, self.partition))):
            img_labels = img.split("_")
            # ids.append(img_labes[0])
            ages.append(img_labels[1])
            genders.append(img_labels[2])
            expressions.append(img_labels[3])
            # picture_sets.append(img_labes[4].split('.')[0])

            images.append(img)

        # Prepare the dataset specific information.
        # id_lbls = {x:e for e, x in enumerate(sorted(set(ids)))}
        ages_lbls = {x:e for e, x in enumerate(sorted(set(ages)))}
        gender_lbls = {x:e for e, x in enumerate(sorted(set(genders)))}
        expression_lbls = {x:e for e, x in enumerate(sorted(set(expressions)))}
        # picture_sets_lbls = {x:e for e, x in enumerate(sorted(set(picture_sets)))}

        # id_lbls_encoded = [id_lbls[x] for x in ids]
        age_lbls_encoded = [ages_lbls[x] for x in ages]
        gender_lbls_encoded = [gender_lbls[x] for x in genders]
        expression_lbls_encoded = [expression_lbls[x] for x in expressions]
        # picture_sets_lbls_encoded = [picture_sets_lbls[x] for x in picture_sets]

        self.images = images
        self.y = np.stack([age_lbls_encoded,
                           gender_lbls_encoded,
                           expression_lbls_encoded], axis=1)
        # self.y = np.stack([id_lbls_encoded, age_lbls_encoded,
        #                    gender_lbls_encoded, expression_lbls_encoded,
        #                    picture_sets_lbls_encoded], axis=1)

        # MTL specific information.
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [len(set(ages)),          # 3.
                               len(set(genders)),       # 2.
                               len(set(expressions))    # 6.
                               ]
        # self.task_lbl_sizes = [len(set(ids)), len(set(ages)), len(set(genders)),
        #                        len(set(expressions)), len(set(picture_sets))]

    def __len__(self):
        # Train: 1642. Val: 410.
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.dataset_path / self.partition / self.images[index]).convert('RGB')
        img = self.transforms(img)

        return img, self.y[index]
