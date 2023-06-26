# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/9 15:22
 @name: 
 @desc:
"""
import torch
from torch.utils.data.dataset import Dataset
from prepare.data import *
from utils.eegutils import moving_average
from utils.serialize import file_scanf
import pickle
import numpy as np
from torch.utils.data import Subset


# my subset for train and test dataset use different transform
class MySubset(Subset):
    def __init__(self, dataset, indices, transform=None, filter=False):
        super(MySubset, self).__init__(dataset, indices)
        self.transform = transform
        self.filter = filter

    def __getitem__(self, idx):
        x, y = super(MySubset, self).__getitem__(idx)
        if self.transform is None:
            return x, y
        if isinstance(x, list):
            if self.filter == True:
                for xi in x:
                    if x_is_bad(xi):
                        return [torch.ones_like(xi) for xi in x], y
            x = [self.transform(xi) for xi in x]
        else:
            if self.filter == filter:
                if x_is_bad(x):
                    return torch.zeros_like(x), y
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class BaseEEGImageDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""

    def __init__(self, path):
        self.filepaths = file_scanf(path, endswith='.pkl')

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            y = int(pickle.load(f))

            y = y - 1
            assert 0 <= y <= 39

        return torch.tensor(x, dtype=torch.float).permute(1, 2, 0).unsqueeze(0), torch.tensor(y, dtype=torch.long)


class FusionEEGDataset(BaseEEGImageDataset):

    def __init__(self, path, grid_size=4):
        self.filepaths = file_scanf(path, endswith='.pkl')
        # 计算每个维度上的图像数量
        self.grid_size = grid_size

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]

            # 挑选图片
            # image_list = x[:self.grid_size*self.grid_size]
            image_list = get_image_list(x, self.grid_size)
            dif_image_list = get_image_list(np.diff(x, axis=0), self.grid_size)
            _, w, h = x.shape
            img_size = w

            # 初始化一个大的正方形图片
            large_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))
            dif_large_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))
            # 将小图片放入大图片中
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = image_list[i * self.grid_size + j]
                    dif_large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = dif_image_list[i * self.grid_size + j]

            y = int(pickle.load(f))

            assert 0 <= y <= 39

        return torch.tensor(large_image, dtype=torch.float).unsqueeze(0), torch.tensor(dif_large_image, dtype=torch.float).unsqueeze(0), torch.tensor(y, dtype=torch.long)


# raw, diff, average data
class ThreeFusionEEGDataset(BaseEEGImageDataset):

    def __init__(self, path, grid_size=4):
        self.filepaths = file_scanf(path, endswith='.pkl')
        # 计算每个维度上的图像数量
        self.grid_size = grid_size

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]

            # 挑选图片
            segments = divide_segments(x, self.grid_size)
            mid_image_list = get_image_list(
                segments, self.grid_size, opt='mid')
            dif_image_list = get_image_list(np.diff(x, axis=0), self.grid_size)
            avg_image_list = get_image_list(
                segments, self.grid_size, opt='avg')

            _, w, h = x.shape
            img_size = w

            # 初始化一个大的正方形图片
            mid_large_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))
            dif_large_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))
            avg_large_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))

            # 将小图片放入大图片中
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    mid_large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = mid_image_list[i * self.grid_size + j]
                    dif_large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = dif_image_list[i * self.grid_size + j]
                    avg_large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = avg_image_list[i * self.grid_size + j]

            y = int(pickle.load(f))

            assert 0 <= y <= 39

        return torch.tensor(mid_large_image, dtype=torch.float).unsqueeze(0), torch.tensor(dif_large_image, dtype=torch.float).unsqueeze(0), torch.tensor(avg_large_image, dtype=torch.float).unsqueeze(0), torch.tensor(y, dtype=torch.long)


class EEGImages4x4Dataset(BaseEEGImageDataset):

    def __init__(self, path, grid_size=4, diff=False):
        self.filepaths = file_scanf(path, endswith='.pkl')
        # 计算每个维度上的图像数量
        self.grid_size = grid_size
        self.diff = diff

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            if self.diff:
                x = np.diff(x, axis=0)

            # 挑选图片
            # image_list = x[:self.grid_size*self.grid_size]
            image_list = get_image_list(x, self.grid_size)
            _, w, h = x.shape
            img_size = w

            # 初始化一个大的正方形图片
            large_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))

            # 将小图片放入大图片中
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = image_list[i * self.grid_size + j]

            y = int(pickle.load(f))

            # y = y - 1
            # print(y)
            assert 0 <= y <= 39

        return torch.tensor(large_image, dtype=torch.float).unsqueeze(0), torch.tensor(y, dtype=torch.long)


class DGeneralEEGImageDataset(BaseEEGImageDataset):

    def __init__(self, path, n_channels=3, grid_size=8):
        super().__init__(path)
        self.n_channels = n_channels
        self.grid_size = grid_size

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            x = x[:, 2:-2, 2:-2]
            x = np.diff(x, axis=0)
            # 挑选图片
            segments = divide_segments(x, self.grid_size)
            if self.n_channels == 1:
                x = get_1C_img(segments, self.grid_size)
            elif self.n_channels == 3:
                # x = get_3C_img(segments, self.grid_size)
                x = get_3C_img_B(segments, self.grid_size)

            y = int(pickle.load(f))
            assert 0 <= y <= 39
            y = torch.tensor(y, dtype=torch.long)

        return x, y

# 暂时只是测试一下，估计是没用的
# 每个eeg是一系列的点，每个点是一个图像的均值


class GeneralEEGPointDataset(BaseEEGImageDataset):

    # 默认3通道，8x8拼图，不滑动平均，不用差分，不用差分降序, 取一段时间的平均
    def __init__(self, path, window_size=0):
        super().__init__(path)
        self.window_size = window_size

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:

            # 读取数据 [t, w, h]
            x = pickle.load(f)

            # 去除边缘
            x = x[:, 2:-2, 2:-2]

            # 计算x的每个图像的均值
            x = np.mean(x, axis=(1, 2))

            # 滑动平均
            if self.window_size > 0:
                x = moving_average(x, self.window_size)

            y = int(pickle.load(f))
            assert 0 <= y <= 39
            # y = torch.tensor(y, dtype=torch.long)

        return torch.tensor(x, dtype=torch.float).unsqueeze(1), torch.tensor(y, dtype=torch.long)

# 单张图像


class GeneralEEGImageDataset(BaseEEGImageDataset):

    # 默认3通道，8x8拼图，不滑动平均，不用差分，不用差分降序, 取一段时间的平均
    def __init__(self, path, n_channels=3, grid_size=8, window_size=0, diff=False, desc=False, opt='avg'):
        super().__init__(path)
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.window_size = window_size
        self.diff = diff
        self.desc = desc
        self.opt = opt

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:

            # 读取数据 [t, w, h]
            x = pickle.load(f)

            # 去除边缘
            x = x[:, 2:-2, 2:-2]

            # 差分
            if self.diff:
                x = np.diff(x, axis=0)

            # 滑动平均
            if self.window_size > 0:
                x = moving_average_images(x, self.window_size)

            # 差分降序
            if self.desc:
                segments = divide_segments(x, self.grid_size)
            # 均匀划分
            else:
                segments = divide_simple_segments(x, self.grid_size)

            if self.n_channels == 1:
                x = get_1C_img(segments, self.grid_size, self.opt)
            elif self.n_channels == 3:
                # avg, max, min
                x = get_3C_img(segments, self.grid_size)

            y = int(pickle.load(f))
            assert 0 <= y <= 39
            # y = torch.tensor(y, dtype=torch.long)

        return x, y

# EEG feature 图像，64个小图，随机挑16个用wavelet, 随机挑16个用dct


class FeatureEEGImageDataset(BaseEEGImageDataset):

    # 默认3通道，8x8拼图，不滑动平均，不用差分，不用差分降序, 取一段时间的平均
    def __init__(self, path, n_channels=3, grid_size=8, window_size=0, diff=False, desc=False, opt='avg'):
        super().__init__(path)
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.window_size = window_size
        self.diff = diff
        self.desc = desc
        self.opt = opt

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:

            # 读取数据 [t, w, h]
            x = pickle.load(f)

            # 去除边缘
            x = x[:, 2:-2, 2:-2]

            # 差分
            if self.diff:
                x = np.diff(x, axis=0)

            # 滑动平均
            if self.window_size > 0:
                x = moving_average_images(x, self.window_size)

            # 差分降序
            if self.desc:
                segments = divide_segments(x, self.grid_size)
            # 均匀划分
            else:
                segments = divide_simple_segments(x, self.grid_size)

            if self.n_channels == 1:
                x = get_1C_feature_img(segments, self.grid_size, self.opt, feature_num=16, use_dct=True, use_wavelet=True)
            elif self.n_channels == 3:
                # avg, max, min
                x = get_3C_img(segments, self.grid_size)

            y = int(pickle.load(f))
            assert 0 <= y <= 39
            # y = torch.tensor(y, dtype=torch.long)

        return x, y


class N2_GeneralEEGImageDataset(GeneralEEGImageDataset):
    def __init__(self, path, n_channels=3, grid_size=8):
        super().__init__(path, n_channels, grid_size)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            x = x[:, 2:-2, 2:-2]
            # 挑选图片
            segments = divide_segments(x, self.grid_size)
            # segments = divide_simple_segments(x, self.grid_size)

            if self.n_channels == 1:
                x1 = get_1C_img(segments, self.grid_size)
                x2 = get_1C_img(segments, self.grid_size)
            elif self.n_channels == 3:
                # x = get_3C_img(segments, self.grid_size)
                x1 = get_3C_img_B(segments, self.grid_size)
                x2 = get_3C_img_B(segments, self.grid_size)

            y = int(pickle.load(f))
            assert 0 <= y <= 39
            y = torch.tensor(y, dtype=torch.long)

        # if out_of_range(x1) or out_of_range(x2):
        #     return None
        return x1, x2, y


# 差分图片。n_samples: 如果为2，则差分排序的图片数量为2，均匀排序的图片数量为2，也就是一共4张大图。
class DC_GeneralEEGImageDataset(GeneralEEGImageDataset):
    def __init__(self, path, n_channels=3, grid_size=8, n_samples=2):
        super().__init__(path, n_channels, grid_size)
        self.n_samples = n_samples

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            x = x[:, 2:-2, 2:-2]
            x = np.diff(x, axis=0)
            # 挑选图片
            segments = divide_segments(x, self.grid_size)
            c_segments = divide_simple_segments(x, self.grid_size)
            x_list = []

            if self.n_channels == 1:
                for i in range(self.n_samples):
                    x_list.append(get_1C_img(segments, self.grid_size))
                    x_list.append(get_1C_img(c_segments, self.grid_size))
            elif self.n_channels == 3:
                for i in range(self.n_samples):
                    x_list.append(get_3C_img_B(segments, self.grid_size))
                    x_list.append(get_3C_img_B(c_segments, self.grid_size))

            y = int(pickle.load(f))
            assert 0 <= y <= 39
            y = torch.tensor(y, dtype=torch.long)

        # if out_of_range(x1) or out_of_range(x2):
        #     return None

        return x_list, y

# n_samples: 如果为2，则差分排序的图片数量为2，均匀排序的图片数量为2，也就是一共4张大图。


class C_GeneralEEGImageDataset(GeneralEEGImageDataset):
    def __init__(self, path, n_channels=3, grid_size=8, n_samples=2):
        super().__init__(path, n_channels, grid_size)
        self.n_samples = n_samples

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            x = x[:, 2:-2, 2:-2]
            diff_x = np.diff(x, axis=0)
            # ddiff_x = np.diff(diff_x, axis=0)
            # 挑选图片
            # c_segments = divide_segments(x, self.grid_size)
            # 简单划分，均匀划分，效果不明显
            # c_segments = divide_simple_segments(x, self.grid_size)
            # 直接用差分图片
            # segments = divide_segments(x, self.grid_size)
            # c_segments = divide_segments(diff_x, self.grid_size)
            segments = divide_simple_segments(x, self.grid_size)
            c_segments = divide_simple_segments(diff_x, self.grid_size)
            x_list = []

            if self.n_channels == 1:
                for i in range(self.n_samples):
                    x_list.append(get_1C_img(segments, self.grid_size))
                    x_list.append(get_1C_img(c_segments, self.grid_size))
            elif self.n_channels == 3:
                for i in range(self.n_samples):
                    x_list.append(get_3C_img_B(segments, self.grid_size))
                    x_list.append(get_3C_img_B(c_segments, self.grid_size))

            y = int(pickle.load(f))
            assert 0 <= y <= 39
            y = torch.tensor(y, dtype=torch.long)

        # if out_of_range(x1) or out_of_range(x2):
        #     return None

        return x_list, y

# N个差分降序图片，随机选图


class N_GeneralEEGImageDataset(GeneralEEGImageDataset):
    def __init__(self, path, n_channels=3, grid_size=8, n_samples=2):
        super().__init__(path, n_channels, grid_size)
        self.n_samples = n_samples

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            x = x[:, 2:-2, 2:-2]
            # 挑选图片
            segments = divide_segments(x, self.grid_size)
            # segments = divide_simple_segments(x, self.grid_size)
            x_list = []

            if self.n_channels == 1:
                for i in range(self.n_samples):
                    x_list.append(get_1C_img(segments, self.grid_size))
            elif self.n_channels == 3:
                for i in range(self.n_samples):
                    x_list.append(get_3C_img_B(segments, self.grid_size))

            y = int(pickle.load(f))
            assert 0 <= y <= 39
            y = torch.tensor(y, dtype=torch.long)

        # if out_of_range(x1) or out_of_range(x2):
        #     return None

        return x_list, y


# 差分图片
class DN_GeneralEEGImageDataset(GeneralEEGImageDataset):
    def __init__(self, path, n_channels=3, grid_size=8, n_samples=2):
        super().__init__(path, n_channels, grid_size)
        self.n_samples = n_samples

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            x = x[:, 2:-2, 2:-2]
            x = np.diff(x, axis=0)
            # x =
            # 挑选图片
            segments = divide_segments(x, self.grid_size)
            # segments = divide_simple_segments(x, self.grid_size)
            x_list = []

            if self.n_channels == 1:
                for i in range(self.n_samples):
                    x_list.append(get_1C_img(segments, self.grid_size))
            elif self.n_channels == 3:
                for i in range(self.n_samples):
                    x_list.append(get_3C_img_B(segments, self.grid_size))

            y = int(pickle.load(f))
            assert 0 <= y <= 39
            y = torch.tensor(y, dtype=torch.long)

        # if out_of_range(x1) or out_of_range(x2):
        #     return None

        return x_list, y


class EEGImages3CDataset(BaseEEGImageDataset):

    def __init__(self, path, grid_size=8, opt='ran', alpha=0.8):
        self.filepaths = file_scanf(path, endswith='.pkl')
        # 计算每个维度上的图像数量
        self.grid_size = grid_size
        self.opt = opt
        self.alpha = alpha

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            w = x.shape[1]
            x = x[:, 2:-2, 2:-2]
            # 挑选图片
            segments = divide_segments(x, self.grid_size)
            # count_segments(segments)
            ran_image_list = get_image_list(segments, opt='ran')
            mid_image_list = get_image_list(segments, opt='mid')
            std_image_list = get_image_list(segments, opt='std')
            # dif_image_list = get_image_list(segments, opt='dif')
            avg_image_list = get_image_list(segments, opt='avg')

            img_size = w-4
            y = int(pickle.load(f))
            assert 0 <= y <= 39
            y = torch.tensor(y, dtype=torch.long)

            # 初始化一个大的正方形图片
            raw_large_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))
            dif_large_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))
            avg_large_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))
            std_large_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))

            # 将小图片放入大图片中
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    raw_large_image[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size] = mid_image_list[i *
                                                                                                                       self.grid_size + j]*self.alpha + ran_image_list[i * self.grid_size + j]*(1-self.alpha)
                    # dif_large_image[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size] = dif_image_list[i * self.grid_size + j]
                    avg_large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = avg_image_list[i * self.grid_size + j]
                    std_large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = std_image_list[i * self.grid_size + j]

        return torch.stack((torch.tensor(raw_large_image, dtype=torch.float), torch.tensor(avg_large_image, dtype=torch.float), torch.tensor(std_large_image, dtype=torch.float)), 0), y

# 4个4x4图像


class EEGImages3CDatasetB(BaseEEGImageDataset):

    def __init__(self, path, grid_size=8, opt='ran', alpha=0.8):
        self.filepaths = file_scanf(path, endswith='.pkl')
        # 计算每个维度上的图像数量
        self.grid_size = grid_size
        self.opt = opt
        self.alpha = alpha

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            w = x.shape[1]
            x = x[:, 2:-2, 2:-2]
            # 挑选图片
            segments = divide_segments(x, self.grid_size//2)
            # count_segments(segments)
            ran_image_list = []
            for i in range(self.grid_size//2):
                ran_image_list.append(get_image_list(segments, opt='ran'))
            mid_image_list = get_image_list(segments, opt='mid')
            std_image_list = get_image_list(segments, opt='std')
            avg_image_list = get_image_list(segments, opt='avg')

            img_size = w-4
            y = int(pickle.load(f))
            assert 0 <= y <= 39
            y = torch.tensor(y, dtype=torch.long)

            raw_middle_image = np.zeros(
                (self.grid_size//2 * img_size, self.grid_size//2 * img_size))
            std_middle_image = np.zeros(
                (self.grid_size//2 * img_size, self.grid_size//2 * img_size))
            avg_middle_image = np.zeros(
                (self.grid_size//2 * img_size, self.grid_size//2 * img_size))

            raw_middle_image_list = []
            for k in range(self.grid_size//2):
                for i in range(self.grid_size//2):
                    for j in range(self.grid_size//2):
                        raw_middle_image[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size] = mid_image_list[i *
                                                                                                                            self.grid_size//2 + j]*self.alpha + ran_image_list[k][i * self.grid_size//2 + j]*(1-self.alpha)
                raw_middle_image_list.append(raw_middle_image)

            for i in range(self.grid_size//2):
                for j in range(self.grid_size//2):
                    std_middle_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = std_image_list[i * self.grid_size//2 + j]
                    avg_middle_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = avg_image_list[i * self.grid_size//2 + j]

            # 初始化一个大的正方形图片
            raw_large_image = paste_image(
                self.grid_size, raw_middle_image_list)
            avg_large_image = paste_image(self.grid_size, [avg_middle_image]*4)
            std_large_image = paste_image(self.grid_size, [std_middle_image]*4)

        return torch.stack((torch.tensor(raw_large_image, dtype=torch.float), torch.tensor(avg_large_image, dtype=torch.float), torch.tensor(std_large_image, dtype=torch.float)), 0), y


class SimpleEEGImages4x4Dataset(BaseEEGImageDataset):

    def __init__(self, path, grid_size=4):
        self.filepaths = file_scanf(path, endswith='.pkl')
        # 计算每个维度上的图像数量
        self.grid_size = grid_size

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]

            # 挑选图片
            # image_list = x[:self.grid_size*self.grid_size]
            image_list = get_simple_image_list(x, self.grid_size)
            _, w, h = x.shape
            img_size = w

            # 初始化一个大的正方形图片
            large_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))

            # 将小图片放入大图片中
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = image_list[i * self.grid_size + j]

            y = int(pickle.load(f))

            # y = y - 1
            # print(y)
            assert 0 <= y <= 39

        return torch.tensor(large_image, dtype=torch.float).unsqueeze(0), torch.tensor(y, dtype=torch.long)


class EEGImages128FusionDataset(EEGImages4x4Dataset):

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            delta_x = np.abs(np.diff(x, axis=0)).sum(axis=1)

            # 挑选图片
            # image_list = x[:self.grid_size*self.grid_size]
            image_list = get_image_list(x, self.grid_size)
            delta_image_list = get_image_list(delta_x, self.grid_size)

            _, w, h = x.shape
            img_size = w

            # 初始化一个大的正方形图片
            large_raw_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))
            large_delta_image = np.zeros(
                (self.grid_size * img_size, self.grid_size * img_size))

            # 将小图片放入大图片中
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    large_raw_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = image_list[i * self.grid_size + j]
                    large_delta_image[i * img_size:(i + 1) * img_size, j * img_size:(
                        j + 1) * img_size] = delta_image_list[i * self.grid_size + j]

            y = int(pickle.load(f))

            assert 0 <= y <= 39

        return torch.tensor(large_raw_image, dtype=torch.float).unsqueeze(0), torch.tensor(large_delta_image, dtype=torch.float).unsqueeze(0), torch.tensor(y, dtype=torch.long)
