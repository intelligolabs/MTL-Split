#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset


class Medic(Dataset):
    def __init__(self, dataset_path, transforms, partition):
        super().__init__()

        self.partition = partition
        self.transforms = transforms
        self.dataset_path = dataset_path

        # Load metadata.
        data = pd.read_csv(partition, sep='\t')

        # Get info from pandas df.
        images = data.loc[:, 'image_path'].values
        y1 = data.loc[:, 'damage_severity'].values
        y2 = data.loc[:, 'disaster_types'].values

        # Filter data.
        correct_idx = np.where((y2 != 'not_disaster') & (y2 != 'other_disaster'))[0]
        images = images[correct_idx]
        y1 = y1[correct_idx]
        y2 = y2[correct_idx]

        # Prepare dataset specific information.
        y1_lbls = {x:e for e, x in enumerate(sorted(set(y1)))}
        y2_lbls = {x:e for e, x in enumerate(sorted(set(y2)))}
        y1_lbl_encoded = [y1_lbls[x] for x in y1]
        y2_lbl_encoded = [y2_lbls[x] for x in y2]

        self.images = images
        self.y = np.stack([y1_lbl_encoded, y2_lbl_encoded], axis=1)

        # MTL specific information.
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [3,   # Damage_severity.
                               5    # Disaster_types.
                               ]

    def __len__(self):
        # Train: 23075. Val: 5649.
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.dataset_path / self.images[index]).convert('RGB')
        img = self.transforms(img)

        return img, self.y[index]
