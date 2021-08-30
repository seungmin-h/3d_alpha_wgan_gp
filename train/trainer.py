#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : train
# @Date : 2021-08-30-08-47
# @Project : 3d_alpha_wgan_gp
# @Author : seungmin

import os, time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from model.model import Generator, EDModel, CodeDiscriminator

import matplotlib.pyplot as plt
from matplotlib import gridspec

cudnn.benchmark = True

def _make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class Trainer(object):

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.device = self._get_device()
        self.loss = nn.L1Loss()

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _get_model(self, latent):
        e = EDModel(out_num=latent)
        g = Generator()
        d = EDModel(out_num=1)
        cd = CodeDiscriminator()
        return e, g, d, cd

    def gradient_penalty(self, model, x, x_fake, w=10):
        alpha_size = tuple((len(x), *(1, ) * (x.dim() - 1)))
        alpha_t = torch.Tensor
        alpha = alpha_t(*alpha_size).to(self.device).uniform_()
        x_hat = (x.data * alpha + x_fake.data * (1 - alpha)).requires_grad_()

        def eps_norm(x, _eps = 1e-15):
            x = x.view(len(x), -1)
            return (x * x + _eps).sum(-1).sqrt()

        def bi_penalty(x):
            return (x - 1)**2

        grad_x_hat = torch.autograd.grad(model(x_hat).sum(),
                                         x_hat,
                                         create_graph=True,
                                         only_inputs=True)[0]

        penalty = w * bi_penalty(eps_norm(grad_x_hat)).mean()
        return penalty

    def freeze_params(self, *args):
        for module in args:
            if module:
                for p in module.parameters():
                    p.requires_grad = False

    def unfreeze_params(self, *args):
        for module in args:
            if module:
                for p in module.parameters():
                    p.requires_grad = True

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def load_weights(self, model, saved_step, save_dir_model):
        if saved_step == 0:
            print("initializing models ...")
            return model.apply(self.weights_init)
        else:
            print("loading models ... ", saved_step)
            return model.load_state_dict(
                torch.load(os.path.join(save_dir_model, str(saved_step).zfill(5) + "_" + str(model) + ".pth")))

    def save_weights(self, model, range_i, save_period, save_dir_model):
        if range_i % save_period == 0:
            print("saving models ... ", str(range_i).zfill(5))
            torch.save(model.state_dict(),
                       os.path.join(save_dir_model, str(range_i).zfill(5) + "_" + str(model) + ".pth"))

    def inf_train_gen(self, data_loader):

        while True:
            for data in data_loader:
                yield data

    def train(self):
        train_loader = self.dataset._get_data_loader()

        # Hyperparameters
        lr = self.config['train']['learning_rate']
        betas = (self.config['train']['beta_1'], self.config['train']['beta_2'])

        save_dir_img = _make_dir(self.config['train']['save_dir_img'])
        save_dir_model = _make_dir(self.config['train']['save_dir_model'])

        saved_step = self.config['train']['saved_step']
        max_step = self.config['train']['max_step']

        iter_g = self.config['train']['iter_g']
        iter_d = self.config['train']['iter_d']
        iter_cd = self.config['train']['iter_cd']

        latent_dim = self.config['model']['latent_dim']

        lambda_1 = self.config['train']['lambda_1']
        lambda_2 = self.config['train']['lambda_2']

        log_period = self.config['train']['log_period']
        save_period = self.config['train']['save_period']

        cube_len = self.config['plot']['cube_len']

        E, G, D, CD = self._get_model(latent_dim)
        E.to(self.device).train()
        G.to(self.device).train()
        D.to(self.device).train()
        CD.to(self.device).train()

        creterion = self.loss.to(self.device)

        optimizer_E = optim.Adam(E.parameters(), lr=lr, betas=betas)
        optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
        optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)
        optimizer_CD = optim.Adam(CD.parameters(), lr=lr, betas=betas)

        self.load_weights(E, saved_step, save_dir_model)
        self.load_weights(G, saved_step, save_dir_model)
        self.load_weights(D, saved_step, save_dir_model)
        self.load_weights(CD, saved_step, save_dir_model)

        data_iter = self.inf_train_gen(train_loader)
        start = time.time()

        for i in range(saved_step + 1, max_step + 1):

            x_real = data_iter.__next__().to(self.device)

            # 랜덤 노이즈
            b_size = x_real.size(0)
            z_rand = torch.randn((b_size, latent_dim)).to(self.device)

            # =========
            # 변환기, 생성기 학습
            # =========

            # E, G만 학습 가능
            self.freeze_params(E, G, D, CD)
            self.unfreeze_params(E, G)

            for step_g in range(iter_g):

                # 정답 데이터를 노이즈로 변환
                z_enc = E(x_real)
                # 변환된 노이즈로부터 데이터를 재구성
                x_rec = G(z_enc)
                # 랜덤 노이즈로부터 데이터 생성
                x_rand = G(z_rand)

                # 재구성 데이터 평가
                validity_x_rec = -D(x_rec).mean()
                # 생성 데이터 평가
                validity_x_rand = -D(x_rand).mean()
                # 변환된 노이즈 평가
                validity_z_enc = -CD(z_enc).mean()
                # 재구성 오차
                loss_recon = lambda_2 * creterion(x_rec, x_real)
                # 전체 손실값
                loss_G = validity_x_rec + validity_x_rand + validity_z_enc + loss_recon

                # 그라디언트 리셋
                E.zero_grad()
                G.zero_grad()

                if step_g == 0:
                    loss_G.backward()
                    # E는 한번만 갱신
                    optimizer_E.step()
                else:
                    loss_G.backward(retain_graph=True)
                # G는 두번 갱신
                optimizer_G.step()

            # =========
            # 식별기 학습
            # =========

            # D 만 학습 가능하게 함
            self.freeze_params(E, G, D, CD)
            self.unfreeze_params(D)

            for step_d in range(iter_d):
                # G 학습시와 같은 처리이나, G/E는 갱신되어있으므로 다른 결과를 얻음
                z_enc = E(x_real)
                x_rec = G(z_enc)
                x_rand = G(z_rand)

                # 재구성 데이터 평가
                validity_x_rec = D(x_rec).mean()
                # 셍성 데이터 평가
                validity_x_rand = D(x_rand).mean()
                # 정답 데이터 평가, G가 생성한 데이터가 2건 있기 때문에, 밸런스를 맞추기 위해 2를 곱합
                validity_x_real = -2 * D(x_real).mean()
                # D의 gradient penalty 계산
                penalty_x_rec = self.gradient_penalty(D, x_real, x_rec, lambda_1)
                penalty_x_rand = self.gradient_penalty(D, x_real, x_rand, lambda_1)
                # 전체 손실값
                loss_D = validity_x_rec + validity_x_rand + validity_x_real + penalty_x_rec + penalty_x_rand

                # D 갱신
                D.zero_grad()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # =========
            # 코드 식별기 학습
            # =========

            # CD 만 학습 가능하게 함
            self.freeze_params(E, G, D, CD)
            self.unfreeze_params(CD)

            for step_cd in range(iter_cd):
                # 변환된 노이즈를 평가, G 학습시의 반대
                validity_z_enc = -validity_z_enc
                # 랜덤 노이즈 평가
                validity_z_rand = -CD(z_rand).mean()
                # CD의 gradient penalty 계산
                penalty_z_enc = self.gradient_penalty(CD, z_rand,
                                                 z_enc.view(b_size, latent_dim),
                                                 lambda_1)
                # 전체 손실값
                loss_CD = validity_z_enc + validity_z_rand + penalty_z_enc

                # CD 갱신
                CD.zero_grad()
                loss_CD.backward()
                optimizer_CD.step()

            # =========
            # 로그 출력과 모델 저장
            # =========
            if i % log_period == 0:
                # 손실값
                print("step: {}".format(i))
                print("loss_E/G: {}".format(loss_G))
                print("loss_D: {}".format(loss_D))
                print("loss_CD: {}".format(loss_CD))

                # 학습 데이터, 재구성 데이터, 생성 데이터 플롯
                # 행방향으로 데이터 종류, 열방향으로 단면 3종류씩, 합계 9장
                plt.figure(figsize=(8, 8))
                gs = gridspec.GridSpec(nrows=3, ncols=3)

                xs = [x_real, x_rec, x_rand]
                categories = ["real", "reconstruct", "rand"]
                axes = ["_sagittal", "_coronal", "_axial"]
                methods = [
                    lambda x: np.flipud(x[:, :, cube_len // 2]).T,
                    lambda x: x[cube_len // 2],
                    lambda x: np.flipud(x[:, cube_len // 2]).T
                ]

                for r in range(3):
                    # 생성 데이터는 -1~1로 정규화 되어있으므로, 0~1로 변환
                    x = xs[r].data.to('cpu').numpy().copy()[0][0] * 0.5 + 0.5
                    for c in range(3):
                        title = categories[r] + axes[c]
                        voxel = methods[c](x)

                        plt.subplot(gs[r, c])
                        plt.title(title)
                        plt.imshow(voxel, cmap="gray", origin='lower')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir_img, str(i).zfill(5) + '.png'))
                plt.clf()
                plt.close()

            # 모델 저장
            self.save_weights(E, i, save_period, save_dir_model)
            self.save_weights(G, i, save_period, save_dir_model)
            self.save_weights(D, i, save_period, save_dir_model)
            self.save_weights(CD, i, save_period, save_dir_model)

        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time / 3600) + "[h]")