#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : model
# @Date : 2021-08-30-08-43
# @Project : 3d_alpha_wgan_gp
# @Author : seungmin

import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, latent=1000):
        super(Generator, self).__init__()
        ngf = 64*2
        self.latent = latent
        self.interpolate = Interpolate(scale=4, mode='nearest')
        self.main = nn.Sequential(
            # layer1
            nn.ConvTranspose3d(self.latent, ngf * 8, 4, 1, 0),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            # layer2
            self.interpolate,
            nn.Conv3d(ngf * 8, ngf * 4, 3, 1, 1),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            # layer3
            self.interpolate,
            nn.Conv3d(ngf * 4, ngf * 2, 3, 1, 1),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # layer4
            self.interpolate,
            nn.Conv3d(ngf * 2, ngf * 1, 3, 1, 1),
            nn.BatchNorm3d(ngf * 1),
            nn.ReLU(True),
            # layer5
            self.interpolate,
            nn.Conv3d(ngf * 1, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        input = input.view(-1, self.latent, 1, 1, 1)
        return self.main(input)

class EDModel(nn.Module):

    def __init__(self, out_num=1):
        super(EDModel, self).__init__()
        ndf = 64*2
        self.main = nn.Sequential(
            # layer1
            nn.Conv3d(1, ndf * 1, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # layer2
            nn.Conv3d(ndf * 1, ndf * 2, 4, 2, 1),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # layer3
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # layer4
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # layer5
            nn.Conv3d(ndf * 8, out_num, 4, 2, 0),
        )
    def forward(self, input):
        return self.main(input)


class CodeDiscriminator(nn.Module):

    def __init__(self, b_size, latent=1000):
        super(CodeDiscriminator, self).__init__()
        ndf = 128*64 #4096
        self.b_size = b_size
        self.latent = latent
        self.main = nn.Sequential(
            # layer1
            nn.Linear(self.latent, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # layer2
            nn.Linear(ndf, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # layer3
            nn.Linear(ndf, 1),
        )

    def forward(self, input):
        input = input.view(self.b_size, self.latent)
        return self.main(input)

class Interpolate(nn.Module):

    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale, mode=self.mode)
        return x