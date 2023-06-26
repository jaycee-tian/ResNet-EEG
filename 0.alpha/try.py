#%%
from preprocessing.dataset import EEGImages128Dataset
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


dataset = EEGImages128Dataset(path='/data0/tianjunchao/dataset/CVPR2021-02785/data/img_pkl/32x32')

# 获取第一张图片
img, label = dataset[0]
arr = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

print(torch.max(img),torch.min(img))
print(torch.max(arr),torch.min(arr))
#%% 显示灰度图像
plt.imshow(arr.squeeze(), cmap='gray')
plt.title('norm: Label: {}'.format(label))
plt.show()

#%% 显示图片
plt.imshow(img.squeeze(), cmap='gray')
plt.title('no norm: Label: {}'.format(label))
plt.show()
# %%
import numpy as np
arrt = np.random.uniform(low=-2, high=2, size=(200, 200))
plt.imshow(arrt, cmap='gray')
plt.title('test: Label: {}'.format(label))
plt.show()
# %%
arra = (arrt - np.min(arrt)) / (np.max(arrt) - np.min(arrt))
plt.imshow(arra, cmap='gray')
plt.title('test: Label: {}'.format(label))
plt.show()
# %%
import numpy as np
import random

# 生成示例数据
x = list(range(100))
y = [0, 17, 44, 88]
y=y+1
print(y)
# 将x切分成5段
segments = np.array_split(x, y)
print(segments)

# 从每段中随机选择一个元素
selected_items = [random.choice(segment) for segment in segments]

print(selected_items)

# %%
import numpy as np
import random

# 输入列表 x，每个元素都是列表
x = [[1,1,1],[3,3,3],[2,2,2]]

print(np.abs(np.diff(x, axis=0)).sum(axis=1))
# 使用嵌套的列表解析，对每个元素进行 np.diff，取绝对值，求和
result = [np.sum(np.abs(np.diff(np.array(sublist)))) for sublist in x]

# 输出结果
# print(result)  # 输出 [3, 4]


# %%
print(123)
# %%
36000/128
# %%
x=[1,2,3,4,5,6,7,8,9,10]
print(x[::len(x)//(2*2)])
# 把x平均分成4份，每份取中间的一个元素
print([x[len(x)//4*i+len(x)//8] for i in range(4)])
# x的长度为700，分成16份，每份取中间的一个元素
print([x[len(x)//16*i+len(x)//32] for i in range(16)])
# %%
# x的维度为[20,5] random的数字
import numpy as np
x = np.random.randint(0, 100, size=(20, 5))
# divide x into 6 segments
# 从0到20之间随机选4个数字



[np.mean(segment, axis=0) for segment in segments]

print(x)
# %%
