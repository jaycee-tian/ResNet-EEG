# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/12/6 20:17
 @desc:
"""
import os
import cv2
import numpy as np
from torch.utils.data import Subset
import torch
from datetime import datetime


def count_three_fusion_dataset(dataloader):
    """
    统计数据集中每个类别的数量
    :param dataloader:
    :return:
    """
    num_classes = 40
    count = np.zeros(num_classes)
    for step, (_, _, _, y) in enumerate(dataloader):
        # 将y转换为NumPy数组
        y = y.numpy()
        # 使用NumPy的bincount函数统计每个类别的数量
        class_count = np.bincount(y, minlength=num_classes)
        # 将统计结果累加到总计数数组中
        count += class_count
    return count


def count_fusion_dataset(dataloader):
    """
    统计数据集中每个类别的数量
    :param dataloader:
    :return:
    """
    num_classes = 40
    count = np.zeros(num_classes)
    for step, (_, _, y) in enumerate(dataloader):
        # 将y转换为NumPy数组
        y = y.numpy()
        # 使用NumPy的bincount函数统计每个类别的数量
        class_count = np.bincount(y, minlength=num_classes)
        # 将统计结果累加到总计数数组中
        count += class_count
    return count


def get_simple_log_dir():
    log_dir = './tblog/' + getNow() + '/'
    return log_dir


def get_log_dir(args):
    return './tblog/' + getNow() + '/' + args.model_name + '/' + get_pid() + '/'


def dct_2d(eeg):
    return cv2.dct(eeg)


def approximated_dct(eeg):
    # [t d]
    oo = ((eeg[::2, ::2] + eeg[1::2, ::2]) +
          (eeg[::2, 1::2] + eeg[1::2, 1::2])) * 2
    ol = ((eeg[::2, ::2] + eeg[1::2, ::2]) -
          (eeg[::2, 1::2] + eeg[1::2, 1::2])) * 2
    lo = ((eeg[::2, ::2] - eeg[1::2, ::2]) +
          (eeg[::2, 1::2] - eeg[1::2, 1::2])) * 2
    ll = ((eeg[::2, ::2] - eeg[1::2, ::2]) -
          (eeg[::2, 1::2] - eeg[1::2, 1::2])) * 2
    # [[oo, ol],
    # [lo, ll]]
    assert np.shape(oo) == np.shape(ll) == (256, 48)
    return np.array([oo, ol, lo, ll])


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def get_pid():
    pid = str(os.getpid())[-3:]
    return pid


def get_feature_log_dir(args):
    log_dir = './3.log/feature/' + getNow() + '/' + get_pid()+'_' + \
        args.model_name+'_' + str(args.lr) + '/'
    return log_dir


def get_model_dir(args):
    model_dir = './4.materials/model/' + getNow() + '/' + get_pid() + '_' + \
        args.model_name + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze layer 4
    # for param in model.layer4.parameters():
    #     param.requires_grad = True
    # for param in model.layer3.parameters():
    #     param.requires_grad = True
    # for param in model.layer2.parameters():
    #     param.requires_grad = True
    # for param in model.layer1.parameters():
    #     param.requires_grad = True


def getNow():
    now = datetime.now()
    current_month = now.month
    current_day = now.day
    current_hour = now.hour
    current_minute = now.minute
    return str(current_month).zfill(2) + str(current_day).zfill(2) + '/' + str(current_hour).zfill(2) + str(current_minute).zfill(2)


def get_test_setting(dataset):
    subset_indices = list(range(3000))  # 构造要使用的数据的下标
    return Subset(dataset, subset_indices)  # 构造子集


def clip_imgs(imgs):
    return [clip_img(img) for img in imgs]


def clip_img(img):
    denominator = img.max() - img.min()
    if denominator == 0:
        # handle the case where the denominator is zero
        normalized_img = np.zeros_like(img)
    else:
        normalized_img = np.nan_to_num((img - img.min()) / denominator)
    return normalized_img


def learning_rate_scheduler(epoch, lr, decay):
    if epoch >= 14:
        lr = decay*(((epoch-14)//5)+1)
    return lr
