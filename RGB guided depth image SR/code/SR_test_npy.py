import torch.utils.data as data
from glob import glob
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import random
import os
import numpy as np
import torch


class cudatatest(data.Dataset):
    def __init__(self,scale):
        super(cudatatest, self).__init__()
        self.imgnames_LR=glob(os.path.join('testset','SR','x'+str(scale)+'_LR'+'*.npy'))
        self.imgnames_HR=glob(os.path.join('testset','SR','x'+str(scale)+'_label'+'*.npy'))
        self.imgnames_guide=glob(os.path.join('testset','SR','x'+str(scale)+'_guide'+'*.npy'))
        self.imgnames_LR.sort()
        self.imgnames_HR.sort()
        self.imgnames_guide.sort()

    def __getitem__(self, item):
    
        self.depth = np.load(self.imgnames_LR[item],allow_pickle=True)
#        print('0',self.depth.shape)
        self.depth = np.transpose(self.depth, (0, 3, 1, 2))
#        print('1',self.depth.shape)
        self.depth = torch.from_numpy(self.depth)
#        print('2',self.depth.shape)
        self.depth=np.squeeze(self.depth,axis=0)
#        print('3',self.depth.shape)

        self.gt = np.load(self.imgnames_HR[item],allow_pickle=True)
        self.gt = np.transpose(self.gt, (0, 3, 1, 2))
        self.gt = torch.from_numpy(self.gt)
        self.gt=np.squeeze(self.gt,axis=0)
        
        self.guide = np.load(self.imgnames_guide[item],allow_pickle=True)
        self.guide = np.transpose(self.guide, (0, 3, 1, 2))
        self.guide = torch.from_numpy(self.guide)
        self.guide=np.squeeze(self.guide,axis=0)
        return (self.depth, self.gt,self.guide)

    def __len__(self):
        return len(self.imgnames_LR)




if __name__ == '__main__':
    dataset = msdatatest(4)
    dataloader = data.DataLoader(dataset, batch_size=1)
    for b1, (img_L,img_gt, img_RGB) in enumerate(dataloader):
#        print(b1)
        print(img_L.shape, img_RGB.shape)