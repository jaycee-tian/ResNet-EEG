

import argparse
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from run.loss import ContrastiveLoss
from prepare.data import filter_sample, filter_two_samples, mixup_data


import subprocess

from utils.eegutils import get_pid


class MyEarlyStopping:

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.prev_acc = None
        self.stop = False

    # acc 连续10轮下降就停止
    def check(self, acc):
        if self.prev_acc is None:
            self.prev_acc = acc
            return
        if acc < self.prev_acc:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            if self.counter > 2:
                print('warning - early stopping counter:', self.counter)
        else:
            self.counter = 0
        self.prev_acc = acc
# class MyEarlyStopping:
    
#     def __init__(self, patience=5):
#         self.patience = patience
#         self.counter = 0
#         self.best_loss = float('inf')
#         self.stop = False
        
#     # 连续5轮上升就停止
#     def check(self, loss):
#         if loss < self.best_loss:
#             self.best_loss = loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.stop = True
#             if self.counter > 2:
#                 print('warning - early stopping counter:',self.counter)
#         print('loss: %.3f, best loss: %.3f, counter: %d' % (loss, self.best_loss, self.counter))
        
def get_gpu_usage():
    """Returns a list of dictionaries containing GPU usage information."""
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], encoding='utf-8')
    lines = output.strip().split('\n')
    # print(lines)
    # gpu memory left percentage
    gpu_info = []
    for line in lines:
        memory_used, memory_total = line.strip().split(',')
        gpu_info.append(int(int(memory_used) / int(memory_total)*100))
    print(gpu_info)
    # return least use gpu
    return gpu_info.index(min(gpu_info))

def get_device(mode='auto',gpu=0):
    if mode == 'auto':
        device = torch.device(f"cuda:{get_gpu_usage()}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
    print('device:',device)
    return device

def get_data_dir():
    data_path = '/data0/tianjunchao/dataset/CVPR2021-02785/data/img_pkl/32x32'
    return data_path


def get_pretrain_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. very important')
    parser.add_argument('--model_name', type=str, default='resnet18', help='model name')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for dataloader')
    args = parser.parse_args()
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default="run", help='use test dataset or not')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for') 
    parser.add_argument('--k', type=int, default=5, help='k-fold')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--opt', type=str, default='mid', help='how to select images')
    # window_size
    parser.add_argument('--window_size', type=int, default=20, help='window_size')
# parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    # parser.add_argument('--gpu', type=int, default=7, help='use gpu')
    parser.add_argument('--grid_size', type=int, default=8, help='grid size')
    # no of channels
    parser.add_argument('--n_channels', type=int, default=1, help='no of channels')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--simple', action='store_true', default=False, help='use simple dataset')
    parser.add_argument('--diff', action='store_true', default=False, help='use diff dataset')
    parser.add_argument('--model_name', type=str, default='resnet18', help='model name')
    parser.add_argument('--ptr', type=str, default='yes', help='ptr weights')
    parser.add_argument('--num_classes', type=int, default=40, help='num classes')
    # 采样次数
    parser.add_argument('--sample_times', type=int, default=8, help='sample_times')
    # alpha
    parser.add_argument('--alpha', type=float, default=1, help='alpha')
    # is_parallel
    parser.add_argument('--is_parallel', type=str, default='no', help='is_parallel')
    args = parser.parse_args()
    args.pid = get_pid()
    return args

def run(device, loader, model, summary, epoch, task='Test', alpha=0.0, smoothing=0.0, optimizer=None, lr_schedulers=None):
    correct = 0
    loss = 0
    y_true = []
    y_pred = []

    if alpha > 0:
        is_mixup = True
    else:
        is_mixup = False

    if smoothing > 0:
        is_smooth = True
    else:
        is_smooth = False

    # mixup = DynamicMixup(alpha = 0.9, num_epochs=10)
    n_samples = 0
    for step, (x, y) in enumerate(loader):
        # if task == 'Train':
        #     x,y = filter_sample(x,y)
        # print(x.shape)
        n_samples += len(y)
        if isinstance(x,list):
            x = [i.to(device) for i in x]
        else:
            x = x.to(device)
        y = y.to(device)
        if task == 'Test':
            batch_loss, y_ = test_one(model=model, x=x, y=y)
        else:
            if smoothing > 0:
                smooth_y = smooth_labels(y,smoothing)
            if is_mixup:
                # 生成 mixup 后的数据和标签
                mix_x, mix_y = mixup_data(x, y, epoch,alpha)
                
                mix_x = mix_x.to(device)
                mix_y = mix_y.to(device)
                batch_loss, y_ = train(model=model, x=mix_x, y=mix_y, optimizer=optimizer)
            elif is_smooth:
                batch_loss, y_ = train_one(model=model, x=x, y=smooth_y, optimizer=optimizer)
            else:
                batch_loss, y_ = train_one(model=model, x=x, y=y, optimizer=optimizer)
            lr_schedulers[0].step()
            lr_schedulers[1].step()

        _, predicted = torch.max(y_, dim=1)
        y_true += y.tolist()
        y_pred += predicted.tolist()
        correct += (torch.argmax(y_, dim=1).data ==
                    y.data).cpu().int().sum().numpy()
        loss += batch_loss.item()
        if task == 'Train':
            if step % 10 == 0:
                summary.add_scalar(tag=task+' step Loss', scalar_value=batch_loss.item(), global_step=step+epoch*len(loader))
        
        




    acc = correct / n_samples * 100
    loss /= len(loader)

    # cm = confusion_matrix(y_true, y_pred)
    # fig = plot_cm(cm)
    summary.add_scalar(tag=task+'Acc', scalar_value=acc, global_step=epoch)
    summary.add_scalar(tag=task+'Loss', scalar_value=loss, global_step=epoch)
    # summary.add_figure(tag=task+"CM", figure=fig, global_step=epoch)
    return acc, loss


def fusion_train(model, raw_x, diff_x, y, optimizer):
    model.train()
    optimizer.zero_grad()
    y_ = model(raw_x, diff_x)
    loss = torch.nn.functional.cross_entropy(y_, y, reduction='sum')
    loss.backward()
    optimizer.step()
    return loss, y_


def fusion_test(model, raw_x, diff_x,  y):
    model.eval()
    with torch.no_grad():
        y_ = model(raw_x, diff_x)
        loss = torch.nn.functional.cross_entropy(y_, y, reduction='sum')
    return loss, y_

def distinct_permutation_np_random_shift(original_permutation):
    shift_amount = np.random.randint(1, len(original_permutation))  # 随机选择移位量
    original_permutation_np = np.array(original_permutation)
    new_permutation = np.roll(original_permutation_np, -shift_amount)
    return new_permutation.tolist()

def con_loss_1(emb0,emb1,y):

    # get an index list of permutation of x
    contrastive_loss = ContrastiveLoss(margin=1.0)
    perm = torch.randperm(emb0.shape[0])
    diff_emb0 = emb0[perm]
    diff_y = (y != y[perm]).to(torch.int)
    return contrastive_loss(emb0, emb1, diff_emb0, diff_y)

# contrastive loss
def con_loss_2(emb0,emb1):

        # 计算对比损失
    cos=torch.nn.CosineSimilarity(dim=-1)
    temperature = 0.5
    mu=5
    pos = cos(emb0, emb1)
    perm = torch.randperm(emb0.shape[0])
    neg = cos(emb0, emb0[perm])
    logits = torch.cat([pos.reshape(-1,1), neg.reshape(-1,1)], dim=1)
    logits /= temperature
    labels = torch.zeros(emb0.shape[0]).to(emb0.device).long()

    return mu * torch.nn.functional.cross_entropy(logits, labels, reduction='sum')

def train_one(model, x, y, optimizer):
    model.train()
    optimizer.zero_grad()
    _,_, y_ = model(x)
    loss = torch.nn.functional.cross_entropy(y_, y, reduction='mean')
    loss.backward()
    optimizer.step()
    return loss, y_

def train(model, x, y, optimizer):
    model.train()
    optimizer.zero_grad()
    emb0, pro0, out0 = model(x[0])
    emb1, pro1, out1 = model(x[1])

    loss1 = torch.nn.functional.cross_entropy(out0, y, reduction='mean')
    # loss2 = con_loss_1(emb0,emb1,y)
    # loss3 = con_loss_2(pro0,pro1)

    # print('loss1: ',loss1.item(),'loss3: ',loss3.item())
    # print('loss1: ',loss1.item())
    # loss = loss1 + loss3
    loss = loss1
    
    loss.backward()
    optimizer.step()
    return loss, out0

def test_one(model, x, y):
    model.eval()
    with torch.no_grad():
        _,_, out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y, reduction='mean')
    return loss, out
def test(model, x, y):
    model.eval()
    with torch.no_grad():
        emb0, proj0, out0 = model(x[0])
        loss = torch.nn.functional.cross_entropy(out0, y, reduction='mean')
    return loss, out0


def seperate_run(device, loader, n_samples, models, summary, epoch, task='Test', optimizers=None):
    raw_model, diff_model = models
    if optimizers is not None:
        raw_optimizer, diff_optimizer = optimizers
    raw_correct = 0
    diff_correct = 0
    raw_loss = 0
    diff_loss = 0
    y_true = []
    raw_y_pred = []
    diff_y_pred = []
    num_classes = 40

    for step, (raw_x, diff_x, y) in enumerate(loader):
        raw_x = raw_x.to(device)
        diff_x = diff_x.to(device)
        y = y.to(device)
        if task == 'Test':
            raw_batch_loss, raw_y_ = test(model=raw_model, x=raw_x, y=y)
            diff_batch_loss, diff_y_ = test(model=diff_model, x=diff_x, y=y)

            raw_y_pred.extend((torch.max(torch.exp(raw_y_), 1)[
                              1]).data.cpu().numpy())  # Save Prediction
            diff_y_pred.extend((torch.max(torch.exp(diff_y_), 1)[
                               1]).data.cpu().numpy())  # Save Prediction
            y_true.extend(y.data.cpu().numpy())  # Save Truth
        else:
            raw_batch_loss, raw_y_ = train(
                model=raw_model, x=raw_x, y=y, optimizer=raw_optimizer)
            diff_batch_loss, diff_y_ = train(
                model=diff_model, x=diff_x, y=y, optimizer=diff_optimizer)

        raw_correct += (torch.argmax(raw_y_, dim=1).data ==
                        y.data).cpu().int().sum().numpy()
        diff_correct += (torch.argmax(diff_y_, dim=1).data ==
                         y.data).cpu().int().sum().numpy()
        raw_loss += raw_batch_loss.item()
        diff_loss += diff_batch_loss.item()

    raw_acc = raw_correct / n_samples * 100
    diff_acc = diff_correct / n_samples * 100
    raw_loss /= n_samples
    diff_loss /= n_samples

    if task == 'Test':
        raw_cm = confusion_matrix(y_true, raw_y_pred)
        diff_cm = confusion_matrix(y_true, diff_y_pred)
        delta_cm = abs(raw_cm - diff_cm)

        # raw correct ids
        raw_correct_ids = np.equal(raw_y_pred, y_true).astype(int)
        # diff correct ids
        diff_correct_ids = np.equal(diff_y_pred, y_true).astype(int)

        # union of raw and diff correct ids
        num_union_correct = np.count_nonzero(np.logical_or(
            raw_correct_ids, diff_correct_ids).astype(int))
        # intersection of raw and diff correct ids
        num_intersection_correct = np.count_nonzero(
            np.logical_and(raw_correct_ids, diff_correct_ids).astype(int))
        summary.add_scalar(tag='Complimentary', scalar_value=int(
            round((1-num_intersection_correct/num_union_correct)*100)), global_step=epoch)

        # format print np.count_nonzero(raw_correct_ids), np.count_nonzero(diff_correct_ids),num_union_correct)
        # print('total raw_correct, diff_correct, union_correct', np.count_nonzero(raw_correct_ids), np.count_nonzero(diff_correct_ids),num_union_correct)

        # print('total raw_correct, diff_correct', len(raw_correct_ids[0]，len(diff_correct_ids[0]))

        # raw_f1_scores = [f1_score(y_true, raw_y_pred, labels=[j], average="micro") for j in range(num_classes)]
        # diff_f1_scores = [f1_score(y_true, diff_y_pred, labels=[j], average="micro") for j in range(num_classes)]
        # delta_f1_scores = [(raw_f1_scores[i] - diff_f1_scores[i]) for i in range(num_classes)]

        # raw_f1_fig = plot_f1(raw_f1_scores)
        # summary.add_figure(tag="Raw-F1", figure=raw_f1_fig, global_step=epoch)
        # diff_f1_fig = plot_f1(diff_f1_scores)
        # summary.add_figure(tag="Diff-F1", figure=diff_f1_fig, global_step=epoch)
        # delta_f1_fig = plot_f1(delta_f1_scores)
        # summary.add_figure(tag="Delta-F1", figure=delta_f1_fig, global_step=epoch)

        # raw_fig = plot_cm(raw_cm)
        # summary.add_figure(tag="Raw-CM", figure=raw_fig, global_step=epoch)
        # diff_fig = plot_cm(diff_cm)
        # summary.add_figure(tag="Diff-CM", figure=diff_fig, global_step=epoch)
        # delta_fig = plot_cm(delta_cm)
        # summary.add_figure(tag="Delta-CM", figure=delta_fig, global_step=epoch)

    summary.add_scalar(tag=task+'RawAcc',
                       scalar_value=raw_acc, global_step=epoch)
    summary.add_scalar(tag=task+'RawLoss',
                       scalar_value=raw_loss, global_step=epoch)
    summary.add_scalar(tag=task+'DiffAcc',
                       scalar_value=diff_acc, global_step=epoch)
    summary.add_scalar(tag=task+'DiffLoss',
                       scalar_value=diff_loss, global_step=epoch)
    return raw_acc, diff_acc, raw_loss, diff_loss

# raw, diff, avg


def three_seperate_run(device, loader, n_samples, models, summary, epoch, task='Test', optimizers=None):
    raw_model, diff_model, avg_model = models
    if optimizers is not None:
        raw_optimizer, diff_optimizer, avg_optimizer = optimizers
    raw_correct = 0
    diff_correct = 0
    avg_correct = 0
    raw_loss = 0
    diff_loss = 0
    avg_loss = 0
    y_true = []
    raw_y_pred = []
    diff_y_pred = []
    avg_y_pred = []

    for step, (raw_x, diff_x, avg_x, y) in enumerate(loader):
        raw_x = raw_x.to(device)
        diff_x = diff_x.to(device)
        avg_x = avg_x.to(device)
        y = y.to(device)
        if task == 'Test':
            raw_batch_loss, raw_y_ = test(model=raw_model, x=raw_x, y=y)
            diff_batch_loss, diff_y_ = test(model=diff_model, x=diff_x, y=y)
            avg_batch_loss, avg_y_ = test(model=avg_model, x=avg_x, y=y)

            raw_y_pred.extend((torch.max(torch.exp(raw_y_), 1)[
                              1]).data.cpu().numpy())  # Save Prediction
            diff_y_pred.extend((torch.max(torch.exp(diff_y_), 1)[
                               1]).data.cpu().numpy())  # Save Prediction
            avg_y_pred.extend((torch.max(torch.exp(avg_y_), 1)[
                              1]).data.cpu().numpy())  # Save Prediction
            y_true.extend(y.data.cpu().numpy())  # Save Truth
        else:
            raw_batch_loss, raw_y_ = train(
                model=raw_model, x=raw_x, y=y, optimizer=raw_optimizer)
            diff_batch_loss, diff_y_ = train(
                model=diff_model, x=diff_x, y=y, optimizer=diff_optimizer)
            avg_batch_loss, avg_y_ = train(
                model=avg_model, x=avg_x, y=y, optimizer=avg_optimizer)

        raw_correct += (torch.argmax(raw_y_, dim=1).data ==
                        y.data).cpu().int().sum().numpy()
        diff_correct += (torch.argmax(diff_y_, dim=1).data ==
                         y.data).cpu().int().sum().numpy()
        avg_correct += (torch.argmax(avg_y_, dim=1).data ==
                        y.data).cpu().int().sum().numpy()
        raw_loss += raw_batch_loss.item()
        diff_loss += diff_batch_loss.item()
        avg_loss += avg_batch_loss.item()

    raw_acc = raw_correct / n_samples * 100
    diff_acc = diff_correct / n_samples * 100
    avg_acc = avg_correct / n_samples * 100

    raw_loss /= n_samples
    diff_loss /= n_samples
    avg_loss /= n_samples

    if task == 'Test':

        # raw correct ids
        raw_correct_ids = np.equal(raw_y_pred, y_true).astype(int)
        # diff correct ids
        diff_correct_ids = np.equal(diff_y_pred, y_true).astype(int)
        # avg correct ids
        avg_correct_ids = np.equal(avg_y_pred, y_true).astype(int)

        # raw-diff
        raw_diff_union_correct = np.count_nonzero(
            np.logical_or(raw_correct_ids, diff_correct_ids).astype(int))
        raw_diff_intersection_correct = np.count_nonzero(
            np.logical_and(raw_correct_ids, diff_correct_ids).astype(int))
        summary.add_scalar(tag='Raw-Diff-Compli', scalar_value=int(round(
            (1-raw_diff_intersection_correct/raw_diff_union_correct)*100)), global_step=epoch)

        # raw-avg
        raw_avg_union_correct = np.count_nonzero(
            np.logical_or(raw_correct_ids, avg_correct_ids).astype(int))
        raw_avg_intersection_correct = np.count_nonzero(
            np.logical_and(raw_correct_ids, avg_correct_ids).astype(int))
        summary.add_scalar(tag='Raw-Avg-Compli', scalar_value=int(round(
            (1-raw_avg_intersection_correct/raw_avg_union_correct)*100)), global_step=epoch)

        # diff-avg
        diff_avg_union_correct = np.count_nonzero(
            np.logical_or(diff_correct_ids, avg_correct_ids).astype(int))
        diff_avg_intersection_correct = np.count_nonzero(
            np.logical_and(diff_correct_ids, avg_correct_ids).astype(int))
        summary.add_scalar(tag='Diff-Avg-Compli', scalar_value=int(round(
            (1-diff_avg_intersection_correct/diff_avg_union_correct)*100)), global_step=epoch)

    summary.add_scalar(tag=task+'RawAcc',
                       scalar_value=raw_acc, global_step=epoch)
    summary.add_scalar(tag=task+'RawLoss',
                       scalar_value=raw_loss, global_step=epoch)
    summary.add_scalar(tag=task+'DiffAcc',
                       scalar_value=diff_acc, global_step=epoch)
    summary.add_scalar(tag=task+'DiffLoss',
                       scalar_value=diff_loss, global_step=epoch)
    summary.add_scalar(tag=task+'AvgAcc',
                       scalar_value=avg_acc, global_step=epoch)
    summary.add_scalar(tag=task+'AvgLoss',
                       scalar_value=avg_loss, global_step=epoch)
    return raw_acc, diff_acc, avg_acc, raw_loss, diff_loss, avg_loss


# 将真实标签转换为one-hot向量
def smooth_labels(targets, smoothing=0.2):
    num_classes = 40
    # 将目标标签转换为one-hot表示形式
    targets_one_hot = torch.zeros(targets.size(0), num_classes, device=targets.device)
    targets_one_hot.scatter_(1, targets.view(-1, 1), 1)

    # 计算平滑后的标签分布
    targets_smoothed = (1.0 - smoothing) * targets_one_hot + smoothing / num_classes
    return targets_smoothed
