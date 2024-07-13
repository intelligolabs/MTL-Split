#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

from pathlib import Path
from datetime import datetime
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data.faces import Faces
from data.medic import Medic
from data.shapes import Shapes3D
from models.mtl_split import MTLSplit


def main(args):
    # Seed.
    seed_everything(args.seed)

    # Dataset and dataloaders.
    dataset_path = Path(args.dataset_path)
    transforms = T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ])

    print(f'Loading {args.dataset_name} dataset and dataloaders [...]')
    if args.dataset_name == 'faces':
        trainset = Faces(dataset_path, transforms, partition='train')
        testset = Faces(dataset_path, transforms, partition='val')
    elif args.dataset_name == 'medic':
        train_partition = os.path.join(dataset_path, 'MEDIC_train.tsv')
        test_partition = os.path.join(dataset_path, 'MEDIC_test.tsv')
        trainset = Medic(dataset_path, transforms, partition=train_partition)
        testset = Medic(dataset_path, transforms, partition=test_partition)
    elif args.dataset_name == 'shapes':
        trainset = Shapes3D(dataset_path, transforms, partition='train')
        testset = Shapes3D(dataset_path, transforms, partition='val')

    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    print('Done!')
    print(f'Trainset task label sizes: {trainset.task_lbl_sizes}')
    print(f'Testset task label sizes: {testset.task_lbl_sizes}')

    # Build the model.
    num_tasks = len(args.task_ids)
    assert len(args.task_out_sizes) >= num_tasks, 'Please provide one or more task IDs and their corresponding output sizes in order!'

    model = MTLSplit(
        hidden_dim=args.hidden_dim,
        dataset_name=args.dataset_name,
        task_ids=args.task_ids,
        task_out_sizes=args.task_out_sizes,
        num_tasks=num_tasks,
        learning_rate=args.learning_rate,
        shared_backbone=args.feature_extractor,
    )

    # Training and evaluation.
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
    run_name = f'MTLSplit_{args.feature_extractor}-{args.dataset_name}-usesMTL_{num_tasks > 1}-tasks_{args.task_ids}'

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename=f'{run_name}'+ '_{epoch}-' + dt_string,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        verbose=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"last_{run_name}"
    # early_stop_callback = EarlyStopping(monitor="val_loss",
    #                                     patience=args.epochs//3,
    #                                     verbose=True, mode="min")

    trainer = None
    if args.debug:
        trainer = Trainer(
            # devices=2,
            # strategy='ddp',
            devices=[args.gpu_num],
            accelerator='gpu',
            max_epochs=args.epochs,
            check_val_every_n_epoch=1,
            callbacks=[checkpoint_callback]
        )
    else:
        wandb_logger = WandbLogger(
            project='', name=run_name, reinit=True, entity=''
        )

        trainer = Trainer(
            # devices=2,
            # strategy='ddp',
            devices=[args.gpu_num],
            accelerator='gpu',
            max_epochs=args.epochs,
            check_val_every_n_epoch=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback]
        )
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SC and MTL')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='outputs/')

    # Network specific parameters.
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--feature_extractor', type=str, default='vgg')

    # Dataset specific parameters.
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--task_ids', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--task_out_sizes', type=int, nargs='+', default=[3, 5])

    args = parser.parse_args()
    main(args)
