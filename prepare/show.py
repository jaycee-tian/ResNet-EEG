import io
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

from utils.eegutils import getNow


def get_tsne_features(features):
    print('features shape: ', features.shape)
    features = PCA(n_components=50).fit_transform(features)
    features = TSNE(verbose=0,learning_rate='auto',init='pca').fit_transform(features)
    return features
    
def plot_tsne(embs,projs,labels, filename='test', material_dir=None, plot=False):

    # 随机选10个类别来画t-sne
    unique_classes = np.unique(labels)
    selected_classes = np.random.choice(unique_classes, 10, replace=False)

    # 仅选择选定类别的数据
    selected_data_indices = np.isin(labels, selected_classes)
    selected_embs = embs[selected_data_indices]
    selected_projs = projs[selected_data_indices]
    selected_labels = labels[selected_data_indices]

    # t-sne
    tsne_embs = get_tsne_features(selected_embs)
    tsne_projs = get_tsne_features(selected_projs)
    # draw two tsne
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for c in selected_classes:
        ax[0].scatter(tsne_embs[selected_labels == c, 0], tsne_embs[selected_labels == c, 1])
        ax[1].scatter(tsne_projs[selected_labels == c, 0], tsne_projs[selected_labels == c, 1])
    ax[0].set_title('Embeddings t-SNE')
    ax[1].set_title('Projections t-SNE')
    if plot:
        return plt  # if draw in tensorboard, default dont' save image
    if material_dir is None:
        material_dir = get_material_dir()
    plt.savefig(material_dir + filename + '.png')
    plt.close()     #这里可能有问题，因为notebook要求关闭

def get_hidden_features(model, valid_loader, device, n_samples=4000):
    # 保存特征
    embs = []
    projs = []
    labels = []
    model.eval()
    total = 0
    for i, (xs, y) in enumerate(valid_loader):
        x = xs[0].to(device)
        with torch.no_grad():
            emb, proj, out = model(x)
            emb = emb.view(emb.size(0), -1)
            emb = emb.cpu().numpy()
            embs.append(emb)
            proj = proj.view(proj.size(0), -1)
            proj = proj.cpu().numpy()
            projs.append(proj)
            labels.append(y.numpy())
        total += x.size(0)
        if total > n_samples:
            break
    embs = np.concatenate(embs, axis=0)
    projs = np.concatenate(projs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embs, projs, labels

def plot_tsne_all(model, valid_loader, device, filename='test',n_samples=4000, material_dir=None, plot=False):

    embs, projs, labels = get_hidden_features(model, valid_loader, device, n_samples)
    
    # 随机选10个类别来画t-sne
    unique_classes = np.unique(labels)
    selected_classes = np.random.choice(unique_classes, 10, replace=False)

    # 仅选择选定类别的数据
    selected_data_indices = np.isin(labels, selected_classes)
    selected_embs = embs[selected_data_indices]
    selected_projs = projs[selected_data_indices]
    selected_labels = labels[selected_data_indices]

    # t-sne
    tsne_embs = get_tsne_features(selected_embs)
    tsne_projs = get_tsne_features(selected_projs)
    # draw two tsne
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for c in selected_classes:
        ax[0].scatter(tsne_embs[selected_labels == c, 0], tsne_embs[selected_labels == c, 1])
        ax[1].scatter(tsne_projs[selected_labels == c, 0], tsne_projs[selected_labels == c, 1])
    ax[0].set_title('Embeddings t-SNE')
    ax[1].set_title('Projections t-SNE')
    if plot:
        return plt  # if draw in tensorboard, default dont' save image
    if material_dir is None:
        material_dir = get_material_dir()
    plt.savefig(material_dir + filename + '.png')
    plt.close()     #这里可能有问题，因为notebook要求关闭

    # 绘制 t-sne 图

def get_material_dir():
    material_dir = "4.materials/features/"+getNow()+'/'
    if not os.path.exists(material_dir):
        os.makedirs(material_dir)
    return material_dir
    

def plot_bar(class_count):
    """
    绘制柱状图
    :param class_count: 每个类别的数量
    :return:
    """
    fig = plt.figure(figsize=(8,5))
    plt.bar(np.arange(len(class_count)), class_count)
    plt.show()
    return fig

# plot simple confusion matrix
def plot_cm(cm):
    cm_denom = np.sum(cm, axis=1, keepdims=True)
    cm_denom[cm_denom == 0] = 1
    cm = np.round(cm / cm_denom*100).astype(int)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(cm, cmap=plt.cm.Blues)

    # 显示刻度标签
    ax.set_xticks(np.arange(len(cm)))
    ax.set_yticks(np.arange(len(cm)))
    # 在刻度标签上显示正确的类别名
    ax.set_xticklabels(np.arange(0, len(cm)))
    ax.set_yticklabels(np.arange(0, len(cm)))

    # 为每个矩形添加标签
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="white")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

    return fig


def plot_f1(f1_scores,num_classes=40):
    
    # 绘制当前时刻下所有类别的 F1 值柱状图
    fig = plt.figure(figsize=(8,5))
    plt.bar(np.arange(num_classes), f1_scores)

    # 设置x轴的范围和标签
    plt.xlim(-0.5, num_classes+0.5)
    plt.xticks(np.arange(num_classes), np.arange(num_classes))
    

    # 将x轴移动到y=0的位置
    ax = plt.gca()
    ax.spines['bottom'].set_position('zero')

    plt.show()
    return fig