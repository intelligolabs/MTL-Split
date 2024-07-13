#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class SaltAndPepperNoise(object):
    """
    Implements 'Salt-and-Pepper' noise adding grain (salt and pepper) noise
    (https://en.wikipedia.org/wiki/Salt-and-pepper_noise)

    assumption: high values=white, low values=black

    Inputs:
            - threshold (float):
            - imgType (str): {"cv2","PIL"}
            - lowerValue (int): value for "pepper"
            - upperValue (int): value for "salt"
            - noiseType (str): {"SnP", "RGB"}
    Output:
            - image ({np.ndarray, PIL.Image}): image with noise added
    """
    def __init__(self,
                 treshold:float = 0.005,
                 lowerValue:int = 0.98,
                 upperValue:int = 0.02,
                 noiseType:str = "SnP"):
        self.treshold = treshold
        self.lowerValue = lowerValue
        self.upperValue = upperValue
        self.noiseType = noiseType
        super(SaltAndPepperNoise).__init__()

    def __call__(self, tensor):
        if self.noiseType == "SnP":
            self.treshold = 0.05
            random_matrix = torch.rand(tensor.size())
            tensor[random_matrix >= (1-self.treshold)] = self.upperValue
            tensor[random_matrix <= self.treshold] = self.lowerValue
        elif self.noiseType == "SnPn":
            random_matrix = torch.randn(tensor.size())
            tensor[random_matrix>=(1-self.treshold)] = self.upperValue
            tensor[random_matrix<=self.treshold] = self.lowerValue
        elif self.noiseType == "RGB":
            random_matrix = torch.rand(tensor.size())
            tensor[random_matrix>=(1-self.treshold)] = self.upperValue
            tensor[random_matrix<=self.treshold] = self.lowerValue

        return tensor
