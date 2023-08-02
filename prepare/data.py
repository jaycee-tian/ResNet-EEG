import random
import numpy as np
import torch
import pywt
from utils.eegutils import clip_img, clip_imgs
import numpy as np

from scipy.fftpack import dct
import torch
import numpy as np

class PatchIndex:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.patch_indexes = []
        
    def get_patch_indexes(self):
        if len(self.patch_indexes) == 0:
            self.patch_indexes = fill_square(self.grid_size*self.grid_size)
        return self.patch_indexes


class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, image_tensor):
        noise = torch.normal(self.mean, self.std, image_tensor.size())
        noisy_image = image_tensor + noise
        noisy_image = torch.clamp(noisy_image, 0, 1)
        return noisy_image


class DynamicMixup():
    def __init__(self, alpha, num_epochs):
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.curr_epoch = 0
        self.first_alpha = 0.2

    def mixup_data(self, x, y):
        # get batch size
        batch_size = x.size()[0]
        # generate mixup indices
        idx = torch.randperm(batch_size)

        if self.curr_epoch < self.num_epochs:
            alpha = self.first_alpha + \
                (self.alpha - self.first_alpha) * \
                self.curr_epoch / self.num_epochs
        else:
            alpha = self.alpha
        # compute mixup coefficients
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
        # mixup samples
        mixed_x = lam * x + (1 - lam) * x[idx]
        # mixup labels
        y_a, y_b = y, y[idx]

        # y_a,y_b to one hot
        y_a = torch.nn.functional.one_hot(y_a, num_classes=40)
        y_b = torch.nn.functional.one_hot(y_b, num_classes=40)

        mixed_y = lam * y_a + (1 - lam) * y_b
        return mixed_x, mixed_y

    def __call__(self, x, y):

        # mixed_x, mixed_y = self.mixup_data(x, y)
        # self.curr_epoch += 1
        # return mixed_x, mixed_y
        if self.curr_epoch > self.num_epochs:
            mixed_x, mixed_y = self.mixup_data(x, y)
            self.curr_epoch += 1
        else:
            mixed_x, mixed_y = x, y
        return mixed_x, mixed_y


def mixup_data(x, y, epoch, alpha=1.0):
    """
    生成 mixup 后的数据和标签
    x: 输入数据
    y: 输入标签
    alpha: mixup 参数
    """
    # if epoch<10:
    #     return x, y

    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    lam = np.random.beta(alpha, alpha)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    # y_a,y_b to one hot
    y_a = torch.nn.functional.one_hot(y_a, num_classes=40)
    y_b = torch.nn.functional.one_hot(y_b, num_classes=40)

    mixed_y = lam * y_a + (1 - lam) * y_b

    return mixed_x, mixed_y


def compute_index(x, grid_size):
    # 计算每个网格单元格应该包含多少个元素
    num_elems_per_cell = len(x) // (grid_size * grid_size)

    # 计算每个网格单元格的起始索引
    cell_start_idxs = num_elems_per_cell * np.arange(grid_size * grid_size)

    # 计算每个网格单元格的中心索引
    cell_center_idx = len(x) // (2 * grid_size * grid_size)

    # 计算每个网格单元格的中心位置
    return cell_start_idxs + cell_center_idx

# 前16，后16，随机16，跳着16


def x_is_bad(x, alpha=0.2):
    median = torch.median(x)

    # if median < alpha or > 1-alpha, mask it to false
    if median < alpha or median > 1-alpha:
        return True
    return False


def image_is_ok(images, alpha=1):  # 重塑张量以计算中位数
    images = images.view(images.shape[0], images.shape[1], -1)
    # 计算中位数
    medians, _ = torch.median(images, dim=-1)

    return medians < alpha


def filter_sample(raw_image, raw_labels, alpha=1):
    raw_n_samples = len(raw_labels)
    # dimension 0 to 1,1 to 0, others keep
    # 将样本数据展平为一维
    # if raw_image is list
    if isinstance(raw_image, list):
        # 判断每组的四张图片是否都符合要求
        raw_image = torch.stack(raw_image, dim=1)
        image_requirements = image_is_ok(raw_image, alpha)
        masks = torch.all(image_requirements, dim=1)
    else:
        flat_x = raw_image.view(raw_n_samples, -1)
        # get median
        median, _ = torch.median(flat_x, dim=1)
        masks = torch.abs(median) < alpha

    image = raw_image[masks]
    image = torch.permute(image, (1, 0, 2, 3, 4))
    labels = raw_labels[masks]
    # print('filter_sample',image.shape[1])
    if len(image) == 0:
        return raw_image, raw_labels
    return image, labels


def filter_two_samples(raw_image1, raw_image2, raw_labels, alpha=0.4):
    raw_n_samples = len(raw_labels)
    # 将样本数据展平为一维
    flat_x1 = raw_image1.view(raw_n_samples, -1)
    flat_x2 = raw_image2.view(raw_n_samples, -1)
    # get median

    median_1, _ = torch.median(flat_x1, dim=1)
    median_2, _ = torch.median(flat_x2, dim=1)

    # if median < 0.2 or > 0.8, mask it to false
    masks = torch.logical_and(torch.logical_and(median_1 > alpha, median_1 < (
        1-alpha)), torch.logical_and(median_2 > alpha, median_2 < 1-alpha))

    image1 = raw_image1[masks]
    image2 = raw_image2[masks]
    labels = raw_labels[masks]
    return image1, image2, labels


def get_simple_image_list(x, grid_size, opt='ran'):
    if opt == 'fir':
        return x[:grid_size*grid_size]
    elif opt == 'las':
        return x[-grid_size*grid_size:]
    elif opt == 'ran':
        indices = torch.randperm(len(x))[:grid_size*grid_size]
        return [x[idx] for idx in indices]
    elif opt == 'mid':
        indices = compute_index(x, grid_size)
        return [x[idx] for idx in indices]

# 排序，分组，挑中间


def get_image_list(segments, opt='ran'):
    image_list = None
    if opt == 'ran':
        image_list = [random.choice(segment) for segment in segments]
    elif opt == 'fir':
        image_list = [segment[0] for segment in segments]
    elif opt == 'las':
        image_list = [segment[-1] for segment in segments]
    elif opt == 'mid':
        image_list = [segment[len(segment)//2] for segment in segments]
    elif opt == 'avg':
        image_list = [np.mean(segment, axis=0) for segment in segments]
    elif opt == 'dif':
        image_list = [segment[0] if len(segment) == 1 else np.amax(
            segment, axis=0) - np.amin(segment, axis=0) for segment in segments]
    elif opt == 'std':
        image_list = [segment[0] if len(segment) == 1 else np.std(
            segment, axis=0) for segment in segments]
    elif opt == 'max':
        image_list = [np.amax(segment, axis=0) for segment in segments]
    elif opt == 'min':
        image_list = [np.amin(segment, axis=0) for segment in segments]
    elif opt == 'med':
        image_list = [np.median(segment, axis=0) for segment in segments]
    return image_list
    # return clip_imgs(image_list)


def moving_average_images(x, window_size=20):
    if window_size < 1 or not isinstance(window_size, int):
        raise ValueError("Window size should be a positive integer.")

    if len(x) < window_size:
        raise ValueError(
            "Window size should be smaller or equal to the number of images in the sequence.")

    averaged_images = []
    for i in range(len(x) - window_size + 1):
        window_images = x[i:i + window_size]
        averaged_image = np.mean(window_images, axis=0)
        averaged_images.append(averaged_image)

    return np.array(averaged_images)


def divide_simple_segments(x, grid_size):
    segments = np.array_split(x, grid_size*grid_size)
    return segments


def divide_segments(x, grid_size):
    delta_x = np.sum(np.abs(np.diff(x, axis=0)), axis=(1, 2))
    sorted_diff = sorted(enumerate(delta_x), key=lambda x: x[1], reverse=True)
    sorted_indexes = [i for i, diff in sorted_diff][:grid_size*grid_size-1]
    sorted_indexes.sort()
    segments = np.array_split(x, np.array(sorted_indexes) + 1)
    return segments


def count_segments(segments):
    lens = [len(segment) for segment in segments]
    print(lens)

def fill_square(N):
    n = int(np.sqrt(N))
    if n * n != N:
        raise ValueError("N must be a perfect square.")

    block_size = 2
    if n % block_size != 0:
        raise ValueError("Array size must be divisible by the block size.")

    arr = np.zeros((n, n), dtype=int)
    num_blocks = n // block_size

    counter = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            arr[i * block_size:i * block_size + 2, j * block_size:j * block_size + 2] = [
                [counter, counter + 1],
                [counter + 2, counter + 3]
            ]
            counter += 4
    # to 1D array
    arr = arr.reshape(-1)
    return arr

def paste_image(grid_size, imgs=[]):
    img_size = imgs[0].shape[0]
    large_image = np.zeros((2 * img_size, 2 * img_size))
    large_image[:img_size, :img_size] = imgs[0]
    large_image[:img_size, img_size:] = imgs[1]
    large_image[img_size:, :img_size] = imgs[2]
    large_image[img_size:, img_size:] = imgs[3]
    return large_image

def get_patch_order(grid_size):
    # if patch_indexes is empy
    # if patchIndex.grid_size != grid_size:
    patchIndex = PatchIndex(grid_size)
    # print(patchIndex.get_patch_indexes())
    return patchIndex.get_patch_indexes()

def get_dct_image_list(image_list, indexes):
    # image in image_list[indexes], change to dct
    dct_image_list = []
    for index in indexes:
        image = image_list[index]
        dct_image = dct(dct(image.T, norm='ortho').T, norm='ortho')
        dct_image_list.append(dct_image)
    return dct_image_list

def get_wavelet_image_list(image_list, indexes):
    wavelet_image_list = []
    for index in indexes:
        image = image_list[index]
        # 将torch张量转为numpy数组进行小波变换
        wavelet_img_np = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            coeffs = pywt.dwt2(image, 'haar')
            LL, (LH, HL, HH) = coeffs
            # 将四个系数矩阵拼接成原始图像
            image = np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))
        wavelet_image_list.append(wavelet_img_np)
    return wavelet_image_list

def get_img(segments, grid_size, opt='avg', feature_num=0, use_dct=False, use_wavelet=False):
    img_size = segments[0].shape[1]
    # count_segments(segments)
    image_list = get_image_list(segments, opt=opt)
    # to numpy
    image_list = np.array(image_list)
    # image_list is 64,and empty
    # image_list = np.zeros((grid_size*grid_size, img_size, img_size))


    # 随机选1/4的segment，获得index的list
    # feature_num = grid_size*grid_size//4*3
    # only use dct features
    indexes = np.random.choice(grid_size*grid_size, feature_num, replace=False)
    if feature_num > 0:
        
        if use_dct and use_wavelet:
            dct_indexes = indexes[:feature_num//2]
            wavelet_indexes = indexes[feature_num//2:]
        elif use_dct:
            dct_indexes = indexes
        elif use_wavelet:
            wavelet_indexes = indexes
            
        if use_dct:
            dct_image_list = get_dct_image_list(image_list, dct_indexes)
            dct_image_list = np.array(dct_image_list)
            image_list[dct_indexes] = dct_image_list
        if use_wavelet:
            wavelet_image_list = get_wavelet_image_list(image_list, wavelet_indexes)
            wavelet_image_list = np.array(wavelet_image_list)
            image_list[wavelet_indexes] = wavelet_image_list
                                        
    # 初始化一个大的正方形图片
    large_image = np.zeros((grid_size * img_size, grid_size * img_size))

    # 将小图片放入大图片中
    for i in range(grid_size):
        for j in range(grid_size):
            large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                j + 1) * img_size] = image_list[i * grid_size + j]
        
    return torch.tensor(clip_img(large_image), dtype=torch.float).repeat(3, 1, 1)


def get_all_img(segments, grid_size, opt='avg', feature_num=16):
    return get_img(segments=segments, grid_size=grid_size, opt=opt, feature_num=feature_num, use_dct=True, use_wavelet=True)

def get_raw_img(segments, grid_size, opt='avg'):
    return get_img(segments=segments, grid_size=grid_size, opt=opt, feature_num=0, use_dct=False, use_wavelet=False)

def get_dct_img(segments, grid_size, opt='avg', feature_num=16):
    return get_img(segments=segments, grid_size=grid_size, opt=opt, feature_num=feature_num, use_dct=True, use_wavelet=False)

def get_wavelet_img(segments, grid_size, opt='avg', feature_num=16):
    return get_img(segments=segments, grid_size=grid_size, opt=opt, feature_num=feature_num, use_dct=False, use_wavelet=True)

def get_1C_img(segments, grid_size, opt='avg'):
    img_size = segments[0].shape[1]
    # count_segments(segments)
    image_list = get_image_list(segments, opt=opt)
    # indexes =  get_patch_order(grid_size)-1
    # random indexes
    # indexes = np.random.permutation(grid_size * grid_size)
    # image_list = np.array(image_list)[indexes]

    # 初始化一个大的正方形图片
    large_image = np.zeros((grid_size * img_size, grid_size * img_size))

    # 将小图片放入大图片中
    for i in range(grid_size):
        for j in range(grid_size):
            large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                j + 1) * img_size] = image_list[i * grid_size + j]
        
    return torch.tensor(clip_img(large_image), dtype=torch.float).repeat(3, 1, 1)

def get_3C_img(segments, grid_size):
    img_size = segments[0].shape[1]

    avg_image_list = get_image_list(segments, opt='avg')
    max_image_list = get_image_list(segments, opt='max')
    min_image_list = get_image_list(segments, opt='min')

    # 初始化一个大的正方形图片
    avg_large_image = np.zeros((grid_size * img_size, grid_size * img_size))
    max_large_image = np.zeros((grid_size * img_size, grid_size * img_size))
    min_large_image = np.zeros((grid_size * img_size, grid_size * img_size))

    # 将小图片放入大图片中
    for i in range(grid_size):
        for j in range(grid_size):
            avg_large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                j + 1) * img_size] = avg_image_list[i * grid_size + j]
            max_large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                j + 1) * img_size] = max_image_list[i * grid_size + j]
            min_large_image[i * img_size:(i + 1) * img_size, j * img_size:(
                j + 1) * img_size] = min_image_list[i * grid_size + j]
    return torch.stack((torch.tensor(avg_large_image, dtype=torch.float), torch.tensor(max_large_image, dtype=torch.float), torch.tensor(min_large_image, dtype=torch.float)), 0)


def out_of_range(x):
    if torch.median(x) > 0.8 or torch.median(x) < 0.2:
        return True
    return False


def get_3C_img_B(segments, grid_size):
    img_size = segments[0].shape[1]

    # count_segments(segments)
    ran_image_list_1 = get_image_list(segments, opt='ran')
    ran_image_list_2 = get_image_list(segments, opt='ran')
    ran_image_list_3 = get_image_list(segments, opt='ran')

    # 初始化一个大的正方形图片
    raw_large_image_1 = np.zeros((grid_size * img_size, grid_size * img_size))
    raw_large_image_2 = np.zeros((grid_size * img_size, grid_size * img_size))
    raw_large_image_3 = np.zeros((grid_size * img_size, grid_size * img_size))

    # 将小图片放入大图片中
    for i in range(grid_size):
        for j in range(grid_size):
            raw_large_image_1[i * img_size:(i + 1) * img_size, j * img_size:(
                j + 1) * img_size] = ran_image_list_1[i * grid_size + j]
            raw_large_image_2[i * img_size:(i + 1) * img_size, j * img_size:(
                j + 1) * img_size] = ran_image_list_2[i * grid_size + j]
            raw_large_image_3[i * img_size:(i + 1) * img_size, j * img_size:(
                j + 1) * img_size] = ran_image_list_3[i * grid_size + j]
    return torch.stack((torch.tensor(raw_large_image_1, dtype=torch.float), torch.tensor(raw_large_image_2, dtype=torch.float), torch.tensor(raw_large_image_3, dtype=torch.float)), 0)
