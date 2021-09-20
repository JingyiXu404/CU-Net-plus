import torch.utils.data as data
from glob import glob
import matplotlib.pyplot as plt
from torchvision import transforms
#import cv2
from PIL import Image
import random
import os
import numpy as np
import torch

class cudataset(data.Dataset):
    def __init__(self,target_name,guide_name,label_name):
        super(cudataset, self).__init__()
        self.noise = np.load('trainset/DN/'+target_name+'.npy')#(batch,height,width,c)
        self.noise = np.transpose(self.noise, (0, 3, 1, 2))
        self.noise_t = torch.from_numpy(self.noise)

        self.gt = np.load('trainset/DN/'+label_name+'.npy')  # (batch,height,width,c)
        self.gt = np.transpose(self.gt, (0, 3, 1, 2))
        self.gt_t = torch.from_numpy(self.gt)

        self.guide = np.load('trainset/DN/'+guide_name+'.npy')  # (batch,height,width,c)
        self.guide = np.transpose(self.guide, (0, 3, 1, 2))
        self.guide_t = torch.from_numpy(self.guide)

    def __getitem__(self, item):
        img_noise = self.noise_t[item]
        img_gt = self.gt_t[item]
        img_guide = self.guide_t[item]

        return (img_noise, img_gt,img_guide)

    def __len__(self):
        return len(self.noise)

if __name__ =='__main__':
    dataset=cudataset('data_x','data_y','label')
    dataloader=data.DataLoader(dataset,batch_size=1)
    for b1,(img_L,img_H,img_guide) in enumerate(dataloader):
        print(b1)
        print(img_L.shape,img_H.shape,img_guide.shape)