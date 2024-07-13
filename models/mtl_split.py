#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from torchvision.models import vgg16, vgg16_bn, resnet18, mobilenet_v3_small, efficientnet_b0
from sklearn.metrics import accuracy_score, f1_score


class MTLSplit(pl.LightningModule):
    def __init__(self,
                 hidden_dim,
                 dataset_name,
                 task_ids,
                 task_out_sizes,
                 num_tasks,
                 learning_rate,
                 shared_backbone):
        super(MTLSplit, self).__init__()
        # This to save (e.g., hidden_dim, learning_rate) to the checkpoint.
        self.save_hyperparameters()

        self.task_ids = task_ids
        self.num_tasks = num_tasks
        self.dataset_name = dataset_name
        self.task_out_sizes = task_out_sizes

        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Use pretrained weights for specific datasets.
        pretrained_weights = False
        # pretrained_weights = None         # torchvision >= 0.13
        if self.dataset_name in ['faces']:
            pretrained_weights = True

        # Define the shared backbones.
        if shared_backbone == 'vgg':
            # https://arxiv.org/abs/1409.1556.
            self.shared_backbone = vgg16(pretrained=pretrained_weights).features
            # self.shared_backbone = vgg16(weights=pretrained_weights).features                   # torchvision >= 0.13
        elif shared_backbone == 'vgg_bn':
            # https://arxiv.org/abs/1409.1556.
            self.shared_backbone = vgg16_bn(pretrained=pretrained_weights).features
            # self.shared_backbone = vgg16_bn(weights=pretrained_weights).features                # torchvision >= 0.13
        elif shared_backbone == 'resnet':
            # https://arxiv.org/abs/1512.03385.
            resnet = resnet18(pretrained=pretrained_weights)
            # resnet = resnet18(weights=pretrained_weights)                                         # torchvision >= 0.13
            modules = list(resnet.children())[:-2]
            self.shared_backbone = nn.Sequential(*modules)
        elif shared_backbone == 'mobilenet':
            # https://arxiv.org/abs/1905.02244.
            self.shared_backbone = mobilenet_v3_small(pretrained=pretrained_weights).features
            # self.shared_backbone = mobilenet_v3_small(weights=pretrained_weights).features      # torchvision >= 0.13
        elif shared_backbone == 'efficientnet':
            # https://arxiv.org/abs/1905.11946.
            self.shared_backbone = efficientnet_b0(pretrained=pretrained_weights).features
            # self.shared_backbone = efficientnet_b0(weights=pretrained_weights).features         # torchvision >= 0.13

        # Define the classification heads.
        self.task_solving_heads = nn.ModuleList()
        if shared_backbone in ['vgg', 'vgg_bn', 'resnet']:
            task_head_in_dim = 512
        elif shared_backbone == 'mobilenet':
            task_head_in_dim = 576
        elif shared_backbone == 'efficientnet':
            task_head_in_dim = 1280

        for out_dim in self.task_out_sizes:
            self.task_solving_heads.append(
                nn.Sequential(
                    # nn.Linear(classhead_in_dim*2*2, self.hidden_dim),       # 64x64 imgs.
                    nn.Linear(task_head_in_dim*7*7, self.hidden_dim),         # 224x224 imgs.
                    nn.Dropout(0.1),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, out_dim)
                )
            )

    def forward(self, x):
        # Common features from the shared backbone.
        features = self.shared_backbone(x)
        features = torch.flatten(features, start_dim=1)

        task_logits = []
        for i in range(self.num_tasks):
            logits_ti = self.task_solving_heads[i](features)
            task_logits.append(logits_ti)

        return task_logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            # weight_decay=self.learning_rate
            weight_decay=self.learning_rate*1e-2
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.max_epochs//2, gamma=0.1)
        return [optimizer], [scheduler]

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=5e-2, patience=5)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

        # return optimizer

    def on_train_epoch_start(self):
        self.train_pred = []
        self.train_gt = []

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)

        losses = []
        for i, task in enumerate(self.task_ids):
            lgs, gt = logits[i], y[:, task]
            loss = F.cross_entropy(lgs, gt)
            self.log(f'train_task{task}_loss', loss, sync_dist=True)
            losses.append(loss)
        mtl_loss = sum(losses)
        # https://arxiv.org/abs/1705.07115.
        # mtl_loss = (losses[0] * 0.1) + (losses[1] * 0.9)
        self.log('train_loss', mtl_loss, sync_dist=True)

        # Save for evaluation.
        self.train_pred.append(logits)
        self.train_gt.append(y)

        # Log learning rate for monitoring.
        self.log("lr", self.optimizers().param_groups[0]['lr'],
                 prog_bar=True, on_step=True, sync_dist=True)

        return mtl_loss

    def on_train_epoch_end(self):
        self.mtl_evaluation(self.train_pred, self.train_gt, self.training)

    def on_validation_epoch_start(self):
        self.val_pred = []
        self.val_gt = []

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        losses = []
        for i, task in enumerate(self.task_ids):
            lgs, gt = logits[i], y[:, task]
            loss = F.cross_entropy(lgs, gt)
            self.log(f'val_task{task}_loss', loss, sync_dist=True)
            losses.append(loss)
        mtl_loss = sum(losses)
        self.log('val_loss', mtl_loss, sync_dist=True)

        # Save for evaluation.
        self.val_pred.append(logits)
        self.val_gt.append(y)

    def on_validation_epoch_end(self):
        self.mtl_evaluation(self.val_pred, self.val_gt, self.training)

    def mtl_evaluation(self, pred_list, gt_list, training):
        log_str = 'train' if training else 'val'
        for i, task in enumerate(self.task_ids):
            pred = torch.vstack([x[i] for x in pred_list])
            pred = pred.argmax(-1).detach().cpu().numpy().flatten()

            gt = torch.hstack([y[:, task] for y in gt_list])
            gt = gt.detach().cpu().numpy().flatten()

            acc = accuracy_score(gt, pred)
            self.log(f'{log_str}_task{task}_accuracy', acc, sync_dist=True)

            f1 = f1_score(gt, pred, average='micro')
            self.log(f'{log_str}_task{task}_f1', f1, sync_dist=True)

            if log_str == 'val':
                print(f'Task: {task}, Acc: {acc}, F1: {f1}')
