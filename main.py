from __future__ import print_function
import cv2

cv2.setNumThreads(0)
import sys
import pdb
import argparse
import collections
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import re
from utils.flowlib import flow_to_image
from utils import logger
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.dataloader import MyDataset
from utils.augmentations import Augmentation, Basetransform
torch.backends.cudnn.benchmark = True
from utils.multiscaleloss import MultiscaleLoss, realEPE, RMSE
from glob import glob
import cv2 as cv
from tqdm import tqdm
from eval import eval
from eval import eval_with_epe
import wandb

def find_NewFile(path):
    # 获取文件夹中的所有文�?
    lists = glob(os.path.join(path, '*.tar'))
    # 对获取的文件根据修改时间进行排序
    lists.sort(key=lambda x: os.path.getmtime(x))
    # 把目录和文件名合成一个路�?
    file_new = lists[-1]
    return file_new


parser = argparse.ArgumentParser(description='Casc_LiteFlowNet')
parser.add_argument('--maxdisp', type=int, default=256,
                    help='maxium disparity, out of range pixels will be masked out. Only affect the coarsest cost volume size')
parser.add_argument('--fac', type=float, default=1,
                    help='controls the shape of search grid. Only affect the coarsest cost volume size')
parser.add_argument('--logname', default='logname_en',
                    help='name of the log file')
parser.add_argument('--casc_logname', default='logname_casc_en_four_layer',
                    help='name of the log file')
parser.add_argument('--database', default='/',
                    help='path to the database')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='path of the pre-trained model')
parser.add_argument('--savemodel', default='./',
                    help='path to save the model')
parser.add_argument('--or_resume', default='D:\desktop\Lite-Flownet-master\logname_old\\finetune_1272.tar',  #在此加载基础模型
                    help='whether to reset moving mean / other hyperparameters')
#parser.add_argument('--casc_resume',default=None)
parser.add_argument('--casc_resume',default='D:\desktop\TCN-flownet\程序\Casc-LiteFlownet\\logname_casc_four_layer\\finetune_57.tar')
#parser.add_argument('--casc_resume', default='D:\\desktop\\TCN-flownet\\程序\\Casc-LiteFlownet\\logname_casc_en_re\\finetune_363.tar', #在此加载串联模型
                   # help='whether to reset moving mean / other hyperparameters')
parser.add_argument('--stage', default='chairs',
                    help='one of {chairs, things, 2015train, 2015trainval, sinteltrain, sinteltrainval}')
parser.add_argument('--ngpus', type=int, default=8,
                    help='number of gpus to use.')
args = parser.parse_args()

baselr = 1e-5  #0.00001/2=0.000005=5e-6
batch_size = 8

torch.cuda.set_device(0)

dataset = MyDataset('/home/france/Documents/uniform/train',
                    transform=Augmentation(size=256, mean=(128)))

test_dataset = MyDataset('/home/france/Documents/uniform/test',
                         transform=Basetransform(size=256, mean=(128)))

print('%d batches per epoch' % (len(dataset) // batch_size))
from lite_flownet import liteflownet
from casc_flownet_4layer import casc_liteflownet as casc_en_4layer
from casc_flownet_5layer import casc_liteflownet as casc_en_5layer

model=liteflownet(args.or_resume)
casc_model=casc_en_4layer(args.casc_resume)
model.cuda()
casc_model.cuda()
summary(model, input_size=(3, 3, 256, 256))

optimizer = optim.Adam(casc_model.parameters(), lr=baselr, betas=(0.9, 0.999), amsgrad=False)
# optimizer = optim.SGD(model.parameters(), lr=baselr, momentum=0.9,  weight_decay=5e-4)
criterion = MultiscaleLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True,min_lr=1e-8)
TestImgLoader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True, num_workers=2,
                                            drop_last=True, pin_memory=True)

def train(imgL, imgR, flowl0):
    casc_model.train()
    model.eval()
    Im1=imgL[0].cuda()
    Im2=imgR[0].cuda()
    flowl0[0]=flowl0[0].cuda()
    outs=[model((Im1,Im2))]
    num=1
    total_loss = 0
    optimizer.zero_grad()
    rmse_1=RMSE(outs[0].detach(),flowl0[0].detach())
    for I1,I2,flo in zip(imgL[1:],imgR[1:],flowl0[1:]):
        I1, I2, flo = I1.cuda(), I2.cuda() ,flo.cuda()
        last_out=outs[-1]
        output=casc_model((I1,I2),last_out)
        total_loss+=criterion.forward(output,flo)
        outs.append(output[0])
        num+=1
    mean_loss=total_loss/6
    mean_loss.backward()
    optimizer.step()

    vis = {}
    mean_RMSE=0
    for out1,label in zip(outs,flowl0):
        mean_RMSE+=RMSE(out1.detach(),label.detach().cuda())
    mean_RMSE=mean_RMSE/7
    vis['mean_RMSE'] =mean_RMSE
    return total_loss, vis,rmse_1

def main():
    TrainImgLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                 drop_last=True, pin_memory=True)
    log = logger.Logger(args.savemodel, name=args.logname)

    start_full_time = time.time()
    start_epoch = 1 if args.casc_resume is None else int(re.findall('(\d+)', args.casc_resume)[0]) + 1
    total_iters = 0
    
    wandb.watch(model)
    
    for epoch in range(start_epoch, args.epochs + 1):
        total_train_loss = 0
        total_train_rmse = 0
        total_first_rmse=0
        # training loop
        for batch_idx, (imgL_crop, imgR_crop,flow0) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss, vis,rmse_1 = train(imgL_crop, imgR_crop, flow0)
            if (total_iters) % 50 == 0:
                print('Epoch %d Iter %d/%d training loss = %.3f , RMSE = %.3f   first_RMSE = %.3f ,  , time = %.2f， learn rate*1e5=%.8f' % (epoch,
                                                                                             batch_idx,len(TrainImgLoader),loss,
                                                                                             vis['mean_RMSE'],
                                                                                            rmse_1,
                                                                                             time.time() - start_time,
                                                                                               1e5*scheduler.optimizer.param_groups[0]['lr']  ))
            total_train_loss += loss
            total_first_rmse+=rmse_1
            total_train_rmse += vis['mean_RMSE']
            learn_rate = scheduler.optimizer.param_groups[0]['lr']
            total_iters += 1
        savefilename = args.savemodel + '/' + args.casc_logname + '/casc_finetune_' + str(epoch) + '.tar'
        save_dict = model.state_dict()
        save_dict = collections.OrderedDict(
            {k: v for k, v in save_dict.items() if ('flow_reg' not in k or 'conv1' in k) and ('grid' not in k)})
        torch.save(
            {'epoch': epoch, 'state_dict': save_dict, 'train_loss': total_train_loss / len(TrainImgLoader), },
            savefilename)
        
        
        test_rmsef,test_epe= eval_with_epe(model, TestImgLoader)
        train_rmsef,train_epe= eval_with_epe(model, TrainImgLoader)
        
        wandb.log({"Train loss":total_train_loss / len(TrainImgLoader),
        	"Test rmse":test_rmsef,
        	"Train rmse":train_rmsef,
        	"Learning rate": optimizer.param_groups[0]['lr'],
        	"Test EPE":test_epe,
        	"Train EPE":train_epe})
           
        log.scalar_summary('train/loss', total_train_loss / len(TrainImgLoader), epoch)
        log.scalar_summary('train/RMSE', total_train_rmse / len(TrainImgLoader), epoch)
        log.scalar_summary('test/RMSE', test_rmsef, epoch)
        log.scalar_summary('train/learning rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(total_train_loss / len(TrainImgLoader))
    
	torch.cuda.empty_cache()


    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

if __name__ == '__main__':    
    wandb.init(project="roya",
               name="casc_liteflownet_roya",
               resume=True,
               id="9")
    main()
