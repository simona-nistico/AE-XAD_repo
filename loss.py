import torch
import torch.nn as nn
import numpy as np


class AEXAD_loss(nn.Module):

    def __init__(self, lambda_p=None, lambda_s=None, f=lambda x: torch.where(x >= 0.5, 0.0, 1.0), cuda=True):
        '''
        AEXAD loss
        :param lambda_p: float, anomalous pixels weight, if None it is inferred from the number of anomalous pixels, defaults to None
        :param lambda_s: float, anomalous samples weight, if None it is inferred from the number of anomalous samples, defaults to None
        :param f: func, function used to transform the anomalous pixels, defaults to lambda x: torch.where(x >= 0.5, 0.0, 1.0)
        :param cuda: bool, if True cuda is used, defaults to True
        '''
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.f = f
        self.use_cuda = cuda

    def forward(self, input, target, gt, y):
        rec_n = (input - target) ** 2
        rec_o = (self.f(target) - input) ** 2

        if self.lambda_p is None:
            lambda_p = torch.reshape(np.prod(gt.shape[1:]) / torch.sum(gt, dim=(1, 2, 3)), (-1, 1))
            ones_v = torch.ones((gt.shape[0], np.prod(gt.shape[1:])))
            if self.use_cuda:
                lambda_p = lambda_p.cuda()
                ones_v = ones_v.cuda()
            lambda_p = ones_v * lambda_p
            lambda_p = torch.reshape(lambda_p, gt.shape)
            lambda_p = torch.where(gt == 1, lambda_p, 1 - gt)
        else:
            lambda_p = self.lambda_p
            if self.use_cuda:
                lambda_p = lambda_p.cuda()

        loss_vec = (1 - gt) * rec_n + lambda_p * gt * rec_o

        # Peso calcolato a livello di batch per momento: lambda_s calcolato qui
        loss = torch.sum(loss_vec, dim=(1, 2, 3))
        loss_n = torch.sum((1 - gt) * rec_n)
        loss_a = torch.sum(lambda_p * gt * rec_o)

        lambda_vec = torch.where(y == 1, self.lambda_s, 1.0)
        weighted_loss = torch.sum(loss * lambda_vec)
        return weighted_loss / torch.sum(lambda_vec), loss_n, loss_a # / torch.sum(lambda_vec)


class AEXAD_loss_weighted(nn.Module):

    def __init__(self, lambda_p, lambda_s, f, cuda):
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s # is treated as the normalization term norm = a_max * (n_normali / sum(max_area / tot_pixel_anom))
        # norm = a_max * (n_normali / sum(max_area / tot_pixel_anom))
        self.f = f
        self.use_cuda = cuda

    def forward(self, input, target, gt, y):
        rec_n = (input - target) ** 2
        rec_o = (self.f(target) - input) ** 2

        if self.lambda_p is None:
            lambda_p_v = torch.reshape(np.prod(gt.shape[1:]) / torch.sum(gt, dim=(1, 2, 3)), (-1, 1))
            ones_v = torch.ones((gt.shape[0], np.prod(gt.shape[1:])))
            if self.use_cuda:
                lambda_p_v = lambda_p_v.cuda()
                ones_v = ones_v.cuda()
            lambda_p = ones_v * lambda_p_v
            lambda_p = torch.reshape(lambda_p, gt.shape)
            lambda_p = torch.where(gt == 1, lambda_p, 1 - gt)
        else:
            lambda_p = torch.full((gt.shape[0], np.prod(gt.shape[1:])), fill_value=self.lambda_p)
            if self.use_cuda:
                lambda_p = lambda_p.cuda()
            lambda_p = torch.reshape(lambda_p, gt.shape)
            lambda_p = torch.where(gt == 1, lambda_p, 1 - gt)

        loss_vec = (1 - gt) * rec_n + lambda_p * gt * rec_o

        # Peso calcolato a livello di batch per momento: lambda_s calcolato qui
        # print(loss_vec.shape)
        loss = torch.sum(loss_vec, dim=(1, 2, 3))
        loss_n = torch.sum((1 - gt) * rec_n)
        loss_a = torch.sum(lambda_p * gt * rec_o)
        # lambda_vec = torch.Tensor(np.where(y==1, self.lambda_s, 1.0))
        areas_a = torch.sum(gt, dim=(1, 2, 3))
        lambda_vec = self.lambda_s / torch.where(y == 1, areas_a, torch.full_like(areas_a, fill_value=self.lambda_s,
                                                                            dtype=torch.float32))  # If the sample is normal the resulting weigth is 1.

        weighted_loss = torch.sum(loss * lambda_vec)
        return weighted_loss / torch.sum(lambda_vec), loss_n, loss_a