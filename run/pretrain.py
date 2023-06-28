from prepare.data import AddGaussianNoise
from run.loss import nt_xent_loss
from run.start import get_pretrain_args
from torch.utils.data import DataLoader
from prepare.eegdataset import C_GeneralEEGImageDataset, MySubset
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as transforms
from run.resnet import TesNet
import torch.nn.functional as F
from run.start import get_device
from prepare.show import plot_tsne
from prepare.show import get_material_dir
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from utils.eegutils import get_feature_log_dir, get_model_dir


args = get_pretrain_args()
device = get_device()
log_dir = get_feature_log_dir(args)
# use tensorboard
summary = SummaryWriter(log_dir=log_dir)

data_path = '/data0/tianjunchao/dataset/CVPR2021-02785/data/img_pkl/32x32'
train_transforms = transforms.Compose([
    # transforms.Resize((256, 256), antialias=True),
    # AddGaussianNoise(0., 0.1),
    # transforms.RandomRotation(90),
    # transforms.RandomCrop(224),
    transforms.Normalize([0.512, 0.512, 0.512], [0.228, 0.228, 0.228]),
])
test_transforms = transforms.Compose([
    transforms.Normalize([0.512, 0.512, 0.512], [0.228, 0.228, 0.228]),
])
dataset = C_GeneralEEGImageDataset(
    path=data_path, n_channels=1, grid_size=8, n_samples=1)

train_dataset, test_dataset = random_split(
    dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
train_dataset = MySubset(train_dataset, range(
    len(train_dataset)), train_transforms)
test_dataset = MySubset(test_dataset, range(
    len(test_dataset)), test_transforms)
train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3, prefetch_factor=2)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=3, prefetch_factor=2)


model = TesNet(model_name=args.model_name).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


material_dir = get_material_dir()
print('args:', args)
for epoch in range(args.epochs):
    model.train()
    # 保存特征
    embs = []
    projs = []
    labels = []
    total = 0
    n_samples = 2000
    for i, (x, y) in enumerate(train_dataloader):
        # 清空优化器的梯度
        optimizer.zero_grad()
        dx = torch.stack(x, dim=1)
        dx0 = dx[:, 0, :, :, :].to(device)
        dx1 = dx[:, 1, :, :, :].to(device)
        emb0, pro0, out0 = model(dx0)
        emb1, pro1, out1 = model(dx1)

        loss = nt_xent_loss(pro0, pro1)

        loss.backward()
        optimizer.step()

        total += dx0.size(0)*2
        if total < n_samples:
            embs.append(emb0.view(emb0.size(0), -1).detach().cpu().numpy())
            projs.append(pro0.view(pro0.size(0), -1).detach().cpu().numpy())
            labels.append(y.detach().numpy())
            embs.append(emb1.view(emb1.size(0), -1).detach().cpu().numpy())
            projs.append(pro1.view(pro1.size(0), -1).detach().cpu().numpy())
            labels.append(y.detach().numpy())

        if i % 10 == 0:
            summary.add_scalar('train loss', loss.item(),
                               i+epoch*len(train_dataloader))
    # print loss
    print('epoch: {}, loss: {}'.format(epoch, loss.item()))

    embs = np.concatenate(embs, axis=0)
    projs = np.concatenate(projs, axis=0)
    labels = np.concatenate(labels, axis=0)
    plt = plot_tsne(embs, projs, labels, filename=args.model_name +
                    'train epoch'+str(epoch), plot=True)
    summary.add_figure('tsne train', plt.gcf(), epoch)

    # compute test loss
    model.eval()
    embs = []
    projs = []
    labels = []
    total = 0
    test_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dataloader):
            dx = torch.stack(x, dim=1)
            dx0 = dx[:, 0, :, :, :].to(device)
            dx1 = dx[:, 1, :, :, :].to(device)
            emb0, pro0, out0 = model(dx0)
            emb1, pro1, out1 = model(dx1)

            total += dx0.size(0)*2
            if total < n_samples:
                embs.append(emb0.view(emb0.size(0), -1).cpu().numpy())
                projs.append(pro0.view(pro0.size(0), -1).cpu().numpy())
                labels.append(y.numpy())
                embs.append(emb1.view(emb1.size(0), -1).cpu().numpy())
                projs.append(pro1.view(pro1.size(0), -1).cpu().numpy())
                labels.append(y.numpy())

            loss = nt_xent_loss(pro0, pro1)
            test_loss += loss.item()
    test_loss /= len(test_dataloader)
    summary.add_scalar('test loss', test_loss, epoch)

    embs = np.concatenate(embs, axis=0)
    projs = np.concatenate(projs, axis=0)
    labels = np.concatenate(labels, axis=0)
    plt = plot_tsne(embs, projs, labels, filename=args.model_name +
                    'test epoch'+str(epoch), plot=True)
    summary.add_figure('tsne test', plt.gcf(), epoch)

# save model
model_dir = get_model_dir(args)
torch.save(model.state_dict(), model_dir + 'model.pth')
