# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm

class AdvMixTrainer_adam(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None,
                 generator=None,
                 lrg=None,
                 mu=None):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha
        # learning rate of the generator
        assert lrg is not None
        self.alpha_g = lrg

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        
        # the generator part
        assert generator is not None
        assert mu is not None
        self.optimizer_g = None
        self.generator = generator
        self.batch_size = self.model.batch_size
        self.generator.cuda()
        self.mu = mu
        self.beta = 0.1
        
        # PGD
        self.policy_net = 1
        self.generator_train_threshold = 0.7
        self.generator_lr_decay_factor = 0.95
        self.generator_lr_increase_factor = 1.05


    def train_one_step(self, data):
        ######################
        # training D
        ######################
        self.optimizer.zero_grad()
        loss, p_score, real_embs = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        real_embs = [
            real_embs[0][:self.batch_size],
            real_embs[1][:self.batch_size],
            real_embs[2][:self.batch_size]
        ]

        # generate fake multimodal feature
        batch_h_gen = self.to_var(data['batch_h'][0: self.batch_size], self.use_gpu)
        batch_t_gen = self.to_var(data['batch_t'][0: self.batch_size], self.use_gpu)
        batch_r = self.to_var(data['batch_r'][0: self.batch_size], self.use_gpu)
        batch_hs, batch_hv, batch_ht = self.model.model.get_batch_ent_multimodal_embs(batch_h_gen)
        batch_ts, batch_tv, batch_tt = self.model.model.get_batch_ent_multimodal_embs(batch_t_gen)
        batch_gen_hv, batch_gen_ht = self.generator(batch_hs, batch_hv, batch_ht)
        batch_gen_tv, batch_gen_tt = self.generator(batch_ts, batch_tv, batch_tt)
        scores, fake_embs = self.model.model.get_fake_score(
            batch_h=batch_h_gen,
            batch_r=batch_r,
            batch_t=batch_t_gen,
            mode=data['mode'],
            fake_hv=batch_gen_hv,
            fake_tv=batch_gen_tv,
            fake_ht=batch_gen_ht,
            fake_tt=batch_gen_tt
        )
        # when training D: positive_score > fake_score
        for score in scores:
            loss += self.mu * (-torch.mean(p_score) + torch.mean(score))
        loss += self.mu * self.calc_gradient_penalty(real_embs, fake_embs)
        
        ######################
        # generating adversarial samples using PGD
        ######################
        self.optimizer.zero_grad()

        adv_h, adv_t = self.generate_pgd_adversarial_samples(
            self.model, data, epsilon=0.01, alpha=0.001, num_iterations=10, use_gpu=self.use_gpu
        )

        # 计算对抗样本的损失
        loss_pgd, _, _ = self.model({
            'batch_h': adv_h,
            'batch_t': adv_t,
            'batch_r': batch_r,
            'batch_y': self.to_var(data['batch_y'][0: self.batch_size], self.use_gpu),
            'mode': data['mode']
        })

        loss -= self.mu * loss_pgd
        loss.backward()
        self.optimizer.step()

        ######################
        # training G
        ######################
        self.optimizer_g.zero_grad()
        batch_hs, batch_hv, batch_ht = self.model.model.get_batch_ent_multimodal_embs(batch_h_gen)
        batch_ts, batch_tv, batch_tt = self.model.model.get_batch_ent_multimodal_embs(batch_t_gen)
        batch_gen_hv, batch_gen_ht = self.generator(batch_hs, batch_hv, batch_ht)
        batch_gen_tv, batch_gen_tt = self.generator(batch_ts, batch_tv, batch_tt)
        scores, _ = self.model.model.get_fake_score(
            batch_h=batch_h_gen,
            batch_r=batch_r,
            batch_t=batch_t_gen,
            mode=data['mode'],
            fake_hv=batch_gen_hv,
            fake_tv=batch_gen_tv,
            fake_ht=batch_gen_ht,
            fake_tt=batch_gen_tt
        )
        loss_g = 0.0
        for score in scores:
            loss_g += torch.mean(self.model.model.margin - score) / 3
        loss_g.backward()
        self.optimizer_g.step()

        return loss.item(), loss_g.item()


    def calc_gradient_penalty(self, real_data, fake_data):
        batchsize = real_data[0].shape[0]
        alpha = torch.rand(batchsize, 1).cuda()
        inter_h = alpha * real_data[0].detach() + ((1 - alpha) * fake_data[0].detach())
        inter_r = alpha * real_data[1].detach() + ((1 - alpha) * fake_data[1].detach())
        inter_t = alpha * real_data[2].detach() + ((1 - alpha) * fake_data[2].detach())
        inter_h = torch.autograd.Variable(inter_h, requires_grad=True)
        inter_r = torch.autograd.Variable(inter_r, requires_grad=True)
        inter_t = torch.autograd.Variable(inter_t, requires_grad=True)
        inters = [inter_h, inter_r, inter_t]
        scores = self.model.model.cal_score(inters)

        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=inters,
            grad_outputs=torch.ones(scores.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        # Change the gradient penalty calculation to GAN-QP
        gradient_penalty = (gradients.norm(2, dim=1) ** 2).mean() * self.beta
        return gradient_penalty

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
            print(
                "Learning Rate of D: {}\nLearning Rate of G: {}".format(
                    self.alpha, self.alpha_g)
            )
        else:
            raise NotImplementedError
        print("Finish initializing...")

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            res_g = 0.0
            for data in self.data_loader:
                loss, loss_g = self.train_one_step(data)
                res += loss
                res_g += loss_g
            training_range.set_description("Epoch %d | D loss: %f, G loss %f" % (epoch, res, res_g))

            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def generate_pgd_adversarial_samples(self, model, data, epsilon, alpha, num_iterations, use_gpu):
        # 使用 to_var 方法将 numpy.ndarray 转换为 torch.Tensor
        batch_h = self.to_var(data['batch_h'][0: self.batch_size], use_gpu)
        batch_t = self.to_var(data['batch_t'][0: self.batch_size], use_gpu)
        batch_r = self.to_var(data['batch_r'][0: self.batch_size], use_gpu)
        batch_y = self.to_var(data['batch_y'][0: self.batch_size], use_gpu)

        # 确保 batch_h 和 batch_t 是浮点数类型的张量
        if not batch_h.is_floating_point():
            batch_h = batch_h.float()
        if not batch_t.is_floating_point():
            batch_t = batch_t.float()

        adv_h = batch_h.clone().detach().requires_grad_(True)
        adv_t = batch_t.clone().detach().requires_grad_(True)

        for _ in range(num_iterations):
            loss, _, _ = model({
                'batch_h': adv_h,
                'batch_t': adv_t,
                'batch_r': batch_r,
                'batch_y': batch_y,
                'mode': data['mode']
            })
            loss.backward(retain_graph=True)

            with torch.no_grad():
                if adv_h.grad is not None:
                    adv_h += alpha * adv_h.grad.sign()
                if adv_t.grad is not None:
                    adv_t += alpha * adv_t.grad.sign()
                adv_h = torch.max(torch.min(adv_h, batch_h + epsilon), batch_h - epsilon)
                adv_t = torch.max(torch.min(adv_t, batch_t + epsilon), batch_t - epsilon)

            # 重新设置为可计算梯度的张量
            adv_h = adv_h.detach().requires_grad_(True)
            adv_t = adv_t.detach().requires_grad_(True)

        return adv_h, adv_t