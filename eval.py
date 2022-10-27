from torchsummary import summary
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from glob import glob
import os
from lite_flownet import liteflownet 
import torch
from torch.autograd import Variable
from utils.multiscaleloss import realEPE,RMSE,filtered_aee
from tqdm import tqdm
from utils.dataloader import MyDataset,MyDataset_roya
from utils.augmentations import Basetransform
from utils.flowlib import *
import torch.nn.functional as F

__all__ = [
    'eval'
]


def find_NewFile(path):
    # 获取文件夹中的所有文�?
    lists = glob(os.path.join(path, '*.tar'))
    # 对获取的文件根据修改时间进行排序
    lists.sort(key=lambda x: os.path.getmtime(x))
    # 把目录和文件名合成一个路�?
    file_new = lists[-1]
    return file_new




def eval(model,ImgLoader):
    total_test_rmse = []
    iterator = iter(ImgLoader)
    step = len(ImgLoader)
    # step = 50
    model.eval()
    print('evaluating... ')
    for i in tqdm(range(step)):
        img1, img2, flo = next(iterator)
        img1 = Variable(torch.FloatTensor(img1.float()))
        img2 = Variable(torch.FloatTensor(img2.float()))
        flo = Variable(torch.FloatTensor(flo.float()))
        imgL, imgR, flowl0 = img1.cuda(), img2.cuda(), flo.cuda()
        output = model((imgL, imgR))
        total_test_rmse += [RMSE(output.detach(), flowl0.detach()).cpu().numpy()]
    return np.mean(total_test_rmse)


def eval_with_epe(model,ImgLoader):
    total_test_rmse = []
    total_test_epe = []
    iterator = iter(ImgLoader)
    step = len(ImgLoader)
    # step = 50
    model.eval()
    print('evaluating... ')
    for i in tqdm(range(step)):
        img1, img2, flo = next(iterator)
        img1 = Variable(torch.FloatTensor(img1.float()))
        img2 = Variable(torch.FloatTensor(img2.float()))
        flo = Variable(torch.FloatTensor(flo.float()))
        imgL, imgR, flowl0 = img1.cuda(), img2.cuda(), flo.cuda()
        output = model((imgL, imgR))
        total_test_rmse += [RMSE(output.detach(), flowl0.detach()).cpu().numpy()]
        total_test_epe += [realEPE(output.detach(), flowl0.detach()).cpu().numpy()]
    return np.mean(total_test_rmse), np.mean(total_test_epe)

def eval_with_epe_casc(model,casc_model,ImgLoader):
  total_test_rmse = []
  total_test_aee = []
  total_test_faee = []
  total_test_epe = []
  iterator = iter(ImgLoader)
  step = len(ImgLoader)
  # step = 50
  model.eval()
  casc_model.eval()
  print('evaluating... ')
  for i in tqdm(range(step)):
    img1, img2, flo = next(iterator)
    imgL = Variable(torch.FloatTensor(img1[0].float()))
    imgR = Variable(torch.FloatTensor(img2[0].float()))
    flowl0 = Variable(torch.FloatTensor(flo[0].float()))
    imgL, imgR, flowl0 = imgL.cuda(), imgR.cuda(), flowl0.cuda()
    outs = [model((imgL, imgR))]
    total_test_rmse += [RMSE(outs[0].detach(), flowl0.detach()).cpu().numpy()]
    total_test_epe += [realEPE(outs[0].detach(), flowl0.detach()).cpu().numpy()]
    metrics = [filtered_aee(outs[0].detach(), flowl0.detach()).cpu().numpy()]
    total_test_aee += [metrics['aee']]
    total_test_faee += [metrics['faee']]
    for imgL,imgR,flowl0 in zip(img1[1:],img2[1:],flo[1:]):
      imgL = Variable(torch.FloatTensor(img1[0].float()))
      imgR = Variable(torch.FloatTensor(img2[0].float()))
      flowl0 = Variable(torch.FloatTensor(flo[0].float()))
      imgL, imgR, flowl0 = imgL.cuda(), imgR.cuda(), flowl0.cuda()
      last_out=outs[-1]
      output = casc_model((imgL, imgR),last_out)
      outs.append(output)
      total_test_rmse += [RMSE(output.detach(), flowl0.detach()).cpu().numpy()]
      total_test_epe += [realEPE(output.detach(), flowl0.detach()).cpu().numpy()]
      metrics = [filtered_aee(output.detach(), flowl0.detach())]
      total_test_aee += [metrics['aee']]
      total_test_faee += [metrics['faee']]
  return np.mean(total_test_rmse), np.mean(total_test_epe), np.mean(total_test_aee), np.mean(total_test_faee)
