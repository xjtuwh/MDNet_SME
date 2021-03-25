import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError('Duplicated param name: {:s}'.format(name))


def set_optimizer(model, lr_base, lr_mult, train_all=False, momentum=0.9, w_decay=0.0005):
    if train_all:
        params = model.get_all_params()
    else:
        params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        # nn.BatchNorm2d(96),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        # nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        # nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True))),
                ('fc4',   nn.Sequential(nn.Linear(1024 * 3 * 3, 512),
                                        nn.ReLU(inplace=True))),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU(inplace=True)))]))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduction = 16
        channel = 512
        self.fc = nn.Sequential(
                        nn.Linear(channel, channel // reduction),
                        nn.ReLU(inplace=True),
                        nn.Linear(channel // reduction, channel)
                )
        self.fc_a = nn.Linear(channel, 2)
        self.weighting = nn.Sigmoid()

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])

        # mask = np.array([[1.0321, 1.0079, 1.0325], [0.9944, 0.9910, 0.9943], [1.0247, 1.0033, 1.0226]])
        # self.mask = nn.Parameter(torch.ones(3, 3))
        # self.mask = nn.Parameter(torch.from_numpy(mask).float())

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)

        self.params['fc_a1_weight'] = self.fc[0].weight
        self.params['fc_a1_bias'] = self.fc[0].bias
        self.params['fc_a2_weight'] = self.fc[2].weight
        self.params['fc_a2_bias'] = self.fc[2].bias

        self.params['fc_a3_weight'] = self.fc_a.weight
        self.params['fc_a3_bias'] = self.fc_a.bias

        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_{:d}'.format(k))


    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params
    
    def get_all_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6', query=None):
        # forward model from in_layer to out_layer
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':

                    # attention aa
                    b, c, _, _ = x.size()
                    b1 = self.avg_pool(x).view(b, c)
                    b1 = self.fc(b1).view(b, c, 1)
                    b2 = self.max_pool(x).view(b, c)
                    b2 = self.fc(b2).view(b, c, 1)
                    b_ = torch.cat((b1, b2), 2)
                    b3 = self.avg_pool(x).view(b, c)
                    b3 = self.fc_a(b3).view(b, 2, 1)
                    b3 = F.softmax(b3, 1)
                    out = torch.matmul(b_, b3).view(b, c)
                    out = self.weighting(out).view(b, c, 1, 1)
                    x = torch.cat((x * out, x), 1)

                    x = x.view(x.size(0), -1)
                if name == out_layer:
                    return x

        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x, dim=1)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)
        fc = states['fc']
        self.fc.load_state_dict(fc)
        fc_a = states['fc_a']
        self.fc_a.load_state_dict(fc_a)
        # mask = states['mask']
        # self.mask = mask

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])


class BCELoss(nn.Module):
    def forward(self, pos_score, neg_score, average=True):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = pos_loss.sum() + neg_loss.sum()
        if average:
            loss /= (pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()
        acc = (pos_correct + neg_correct) / (pos_score.size(0) + neg_score.size(0) + 1e-8)
        return acc.item()


class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
        return prec.item()
