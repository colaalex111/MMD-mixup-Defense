
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import numpy as np
import torch

def onehot_transform(label):
    class_number = len(np.unique(label))
    length = label.shape[0]
    onehot_label = []
    for i in range(length):
        new_label = np.zeros(class_number)
        new_label[label[i]] = 1
        onehot_label.append(new_label)
    return np.array(onehot_label)

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def find_best_thres(feature,label):

    sorted_index = np.argsort(feature)
    feature = feature[sorted_index]
    label = label[sorted_index]

    #for i in range(40):
    #    print (feature[i*100])

    #print (feature[:100],label[:100])
    #print (feature[-100:],label[-100:])
    max_acc = -1
    max_thres = -1
    for index in range(len(label)):
        this_thres = feature[index]
        this_count = 0

        for i in range(len(label)):
            if (feature[i]>=this_thres and label[i]==1):
                this_count+=1
            if (feature[i]<this_thres and label[i]==0):
                this_count+=1

        if (this_count>max_acc):
            max_thres = this_thres
            max_acc = this_count

            #print ("this threshold %.2f, this acc %.2f " %(this_thres,this_count/len(label)))

    #print ("max acc %.2f" %(max_acc/len(label)))

    return max_thres
