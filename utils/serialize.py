# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/15 19:13
 @name: 
 @desc:
"""
import glob
import platform
from tqdm import tqdm
# from data_load.read_mat import read_eeg_mat, read_locs_mat
from joblib import Parallel, delayed
import pickle
import numpy as np
from utils.aep import azim_proj, gen_images
import einops
from utils import loadinfo

parallel_jobs = 1
# locs = None
montage_path = None
locs_2d = None

def thread_read_write(x, y, pkl_filename):
    """Writes and dumps the processed pkl file for each stimulus(or called subject).
    [time, channels=127], y
    """
    with open(pkl_filename + '.pkl', 'wb') as file:
        pickle.dump(x, file)
        pickle.dump(y, file)


def go_through(data_filenames, label_filenames, pkl_path):

    n_gridpoints=32

    for data_file, label_file in tqdm(zip(data_filenames,label_filenames), desc=' Total', position=0, leave=True, colour='YELLOW', ncols=80):
        # eeg, y = read_eeg_mat(f)  # [n_samples=5184, t_length=32, channels=62]

        y = loadinfo.LabelReader().get_set(label_file)
        # 400

        eeg, locs = loadinfo.read_auto(data_file, montage_path)
        # (400, 2868, 96), (96,3)

        eeg = eeg[:, ::4, :]
        # 降采样，每隔4，717

        locs_2d = [azim_proj(e) for e in locs]

        # -----------------
        samples, time, channels = np.shape(eeg)
        eeg = einops.rearrange(eeg, 'n t c -> (n t) c', n=samples, t=time, c=channels)


        imgs = gen_images(locs=np.array(locs_2d),  # [samples*time, colors, W, H]
                          features=eeg,
                          n_gridpoints=n_gridpoints,
                          normalize=True).squeeze()

        imgs = einops.rearrange(imgs, '(n t) w h -> n t w h', n=samples, t=time)
        # -------------------

        name = data_file.split('/')[-1].replace('.bdf', '')
        Parallel(n_jobs=parallel_jobs)(
            delayed(thread_read_write)(imgs[i], y[i], pkl_path+str(n_gridpoints)+'x'+str(n_gridpoints)+'/' + name+'_' + str(i) + '_'+str(y[i])+'_'+str(n_gridpoints)+'x'+str(n_gridpoints))
            for i in tqdm(range(len(y)), desc=' write '+name, position=1, leave=False, colour='WHITE', ncols=80))


def file_scanf(path, endswith, sub_ratio=1):
    files = glob.glob(path + '/*')
    if platform.system().lower() == 'windows':
        files = [f.replace('\\', '/') for f in files]
    disallowed_file_endings = (".gitignore", ".DS_Store")
    _input_files = files[:int(len(files) * sub_ratio)]
    return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(endswith), _input_files))


if __name__ == "__main__":
    # 文件所在目录
    dir_path = '/data0/tianjunchao/dataset/CVPR2021-02785/'
    # 数据文件
    file_data = 'data/imagenet40-1000-1-00.bdf'
    # 坐标文件
    file_montage = 'data/biosemi96.sfp'
    # 标签文件
    file_label = 'design/run-00.txt'

    file_path = "/data0/tianjunchao/dataset/CVPR2021-02785/data"
    montage_path = file_path  + "/biosemi96.sfp"
    label_path = "/data0/tianjunchao/dataset/CVPR2021-02785/design"

    data_filenames = file_scanf(file_path, endswith='.bdf')
    label_filenames = file_scanf(label_path, endswith='.txt')
    data_filenames.sort()
    label_filenames.sort()
    # 以上正确
    go_through(data_filenames, label_filenames, pkl_path=file_path+'/img_pkl/')
