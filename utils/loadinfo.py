# copyright @ Zhang Xin (SZU)

import numpy as np
from matplotlib import pyplot as plt
import mne

from PIL import Image
import numpy as np


classes = {"n02106662": 0,
           "n02124075": 1,
           "n02281787": 2,
           "n02389026": 3,
           "n02492035": 4,
           "n02504458": 5,
           "n02510455": 6,
           "n02607072": 7,
           "n02690373": 8,
           "n02906734": 9,
           "n02951358": 10,
           "n02992529": 11,
           "n03063599": 12,
           "n03100240": 13,
           "n03180011": 14,
           "n03272010": 15,
           "n03272562": 16,
           "n03297495": 17,
           "n03376595": 18,
           "n03445777": 19,
           "n03452741": 20,
           "n03584829": 21,
           "n03590841": 22,
           "n03709823": 23,
           "n03773504": 24,
           "n03775071": 25,
           "n03792782": 26,
           "n03792972": 27,
           "n03877472": 28,
           "n03888257": 29,
           "n03982430": 30,
           "n04044716": 31,
           "n04069434": 32,
           "n04086273": 33,
           "n04120489": 34,
           "n04555897": 35,
           "n07753592": 36,
           "n07873807": 37,
           "n11939491": 38,
           "n13054560": 39}

class LabelReader(object):
    def __init__(self, one_hot=False):
        self.file_path = None  # '/data0/tianjunchao/dataset/CVPR2021-02785/design/run-00.txt'
        self.one_hot = one_hot
        self.lines = None

    def read(self):
        with open(self.file_path) as f:
            lines = f.readlines()
        return [line.split('_')[0] for line in lines]

    def get_set(self, file_path):
        if self.file_path == file_path:
            return [classes[e] for e in self.lines]
        else:
            self.file_path = file_path
            self.lines = self.read()
            return [classes[e] for e in self.lines]


def read_auto(file_path, montage_path):
    raw = mne.io.read_raw_bdf(file_path, preload=True, exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'],
                              stim_channel='Status')
    
    montage = mne.channels.read_custom_montage(montage_path)
    raw.set_montage(montage, on_missing='ignore')
    events = mne.find_events(raw, stim_channel='Status', initial_event=True, output='step')
    event_dict = {'stim': 65281, 'end': 0}
    epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True).drop_channels('Status')
    epochs.equalize_event_counts(['stim'])
    stim_epochs = epochs['stim']

    # 获取电极位置
    electrode_coords = [ch['loc'][:3] for ch in raw.info['chs']]
    electrode_coords = electrode_coords[:-1]

    del raw, epochs, events
    return stim_epochs.get_data().transpose(0, 2, 1), electrode_coords  # [b, c, t]


def read_bdf(file_path, montage_path):
    raw = mne.io.read_raw_bdf(file_path, preload=True, exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'], 
                          stim_channel='Status')
    montage = mne.channels.read_custom_montage(montage_path)
    raw.set_montage(montage, on_missing='ignore')

    n_samples = 400
    T = raw.n_times // n_samples # 5021696 // 400 = 12254
    t_length = 32 # time length of each sample, 32 always
    channels = 96
    X_3D = raw._data

    new_shape = (n_samples, X_3D.shape[1]//n_samples, X_3D.shape[0])
    reshaped_data = np.reshape(X_3D, new_shape)

    # 对数组进行转置，形状变为 (400, 97, 12554)
    transposed_data = np.transpose(reshaped_data, (0, 2, 1))

    assert (channels, t_length, n_samples) == np.shape(X_3D)
    X_3D = X_3D[::2, :, :].transpose(2, 1, 0)  # [n_samples=5184, t_length=32, channels=62]

    
    # 获取电极位置
    electrode_coords = [ch['loc'][:3] for ch in raw.info['chs']]
    electrode_coords = electrode_coords[:-1]

    return X_3D, electrode_coords



def save_img(images):

    # Rescale pixel values to [0, 255]
    images = (images + 0.5) * 255

    # Convert from Tensor to numpy array
    images_np = images.astype(np.uint8)

    # Loop over images and save as PNG files
    for i in range(images_np.shape[0]):
        img = Image.fromarray(images_np[i].transpose((1, 2, 0)), mode='RGB')
        img.save(f'/data0/tianjunchao/code/EEGLearn/gen_img/image_{i}.png')

