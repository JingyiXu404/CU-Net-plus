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
    def __init__(self,noise):
        super(cudatatest, self).__init__()
        self.imgnames_noise=glob(os.path.join('testset','DN',str(noise)+'_Noise'+'*.npy'))
        self.imgnames_HR=glob(os.path.join('testset','DN','x'+str(noise)+'_label'+'*.npy'))
        self.imgnames_guide=glob(os.path.join('testset','DN','x'+str(noise)+'_guide'+'*.npy'))
        self.imgnames_noise.sort()
        self.imgnames_HR.sort()
        self.imgnames_guide.sort()

    def __getitem__(self, item):
    
        self.noise= np.load(self.imgnames_noise[item],allow_pickle=True)
#        print('0',self.noise.shape)
        self.noise= np.transpose(self.noise, (0, 3, 1, 2))
#        print('1',self.noise.shape)
        self.noise= torch.from_numpy(self.noise)
#        print('2',self.noise.shape)
        self.noise=np.squeeze(self.noise,axis=0)
#        print('3',self.noise.shape)

        self.gt = np.load(self.imgnames_HR[item],allow_pickle=True)
        self.gt = np.transpose(self.gt, (0, 3, 1, 2))
        self.gt = torch.from_numpy(self.gt)
        self.gt=np.squeeze(self.gt,axis=0)
        
        self.guide = np.load(self.imgnames_guide[item],allow_pickle=True)
        self.guide = np.transpose(self.guide, (0, 3, 1, 2))
        self.guide = torch.from_numpy(self.guide)
        self.guide=np.squeeze(self.guide,axis=0)
        return (self.noise, self.gt,self.guide)

    def __len__(self):
        return len(self.imgnames_noise)




if __name__ == '__main__':
    dataset = msdatatest(4)
    dataloader = data.DataLoader(dataset, batch_size=1)
    for b1, (img_L,img_gt, img_RGB) in enumerate(dataloader):
#        print(b1)
        print(img_L.shape, img_RGB.shape)