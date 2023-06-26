import os
import sys
path = '/data0/tianjunchao/code/Tian-EEG-Image'
sys.path.append(path)
# only use one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def getpath():
    return path