#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : train
# @Date : 2021-08-30-08-29
# @Project : 3d_alpha_wgan_gp
# @Author : seungmin

import yaml

from train.trainer import Trainer
from dataloader.dataloader import DatasetWrapper

def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    trainset = DatasetWrapper(**config['dataset'])

    train_all = Trainer(config, trainset)
    train_all.train()

if __name__ == "__main__":
    main()