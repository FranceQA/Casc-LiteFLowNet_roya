from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
import math
from flowiz import read_flow, convert_from_flow



def load_data(path):
    flo_paths = glob(os.path.join(path, '*.flo'))
    img0_paths = [x.replace('flow.flo', 'img1.tif') for x in flo_paths]
    img1_paths = [x.replace('flow.flo', 'img2.tif') for x in flo_paths]
    return flo_paths, img0_paths, img1_paths

def load_data_roya(path):
  videos = glob(os.path.join(path,'group_1/*'))

  flo, img0, img1 = [], [], []
  for video in videos:
    img_paths = glob(os.path.join(video,'*.png')); img_paths.sort()
    img0_paths = img_paths[0:-1]
    img1_paths = img_paths[1:]
    flo_paths = [x.replace('.png', '.flo') for x in img0_paths]
    flo_paths = [x.replace('group_1', 'flow') for x in flo_paths]

    flo = flo + flo_paths
    img0 = img0 + img0_paths
    img1 = img1 + img0_paths

  return flo, img0, img1

def load_data_casc(path):
  videos = glob(os.path.join(path,'group_1/*'))

  flo = []
  img0 = []
  img1 = []
  for video in videos:
    img_paths = glob(os.path.join(video,'*.png')); img_paths.sort()
    img0_paths = img_paths[0:-1]
    img1_paths = img_paths[1:]
    flo_paths = [x.replace('.png', '.flo') for x in img0_paths]
    flo_paths = [x.replace('group_1', 'flow') for x in flo_paths]

    img0_group = np.array(np.split(np.array(img0_paths),7)).tolist()
    img1_group = np.array(np.split(np.array(img1_paths),7)).tolist()
    flo_group = np.array(np.split(np.array(flo_paths),7)).tolist()

    flo = flo + flo_group
    img0 = img0 + img0_group
    img1 = img1 + img1_group
  return flo, img0, img1
a,b,c = load_data_casc('/content/casc')

class MyDataset(Dataset):
    def __init__(self, path,  transform=None):
        self.flo_paths, self.img0_paths, self.img1_paths = load_data(path)
        self.transform = transform

    def __getitem__(self, i):
        img1 = cv.imread(self.img0_paths[i])
        img2 = cv.imread(self.img1_paths[i])
        flo = readFlowFile(self.flo_paths[i])
        if self.transform is not None:
            img1,img2,flo = self.transform(img1,img2,flo)
        return img1, img2, flo

    def __len__(self):
        return len(self.img0_paths)

class MyDataset_roya(Dataset):
    def __init__(self, path,  transform=None):
        self.flo_paths, self.img0_paths, self.img1_paths = load_data_casc(path)
        self.transform = transform

    def __getitem__(self, i):
        return self.get_values(self.img0_paths[i],self.img1_paths[i],self.flo_paths[i])

    def __len__(self):
        return len(self.img0_paths)

    def get_values(self,img1,img2,flow):
      aux_i1 = []
      aux_i2 = []
      aux_flo = []
      for j in range(len(img1)):
          i1 = cv.imread(img1[j])
          i2 = cv.imread(img2[j])
          flo = read_flow(flow[j])
          i1 = cv.resize(i1,(256,256))
          i2 = cv.resize(i2,(256,256))
          flo = cv.resize(flo,(256,256))
          if self.transform is not None:
              i1,i2,flo = self.transform(i1,i2,flo)
          aux_i1.append(i1)
          aux_i2.append(i2)
          aux_flo.append(flo)
      return aux_i1,aux_i2,aux_flo

