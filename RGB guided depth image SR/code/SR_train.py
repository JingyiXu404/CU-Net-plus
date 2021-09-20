import os
import math
import time
import torch
import random
import matplotlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import transforms
from SR_train_npy import cudataset
from SR_test_npy import cudatatest
from CUNet_plus import CUNet_plus
from torch.utils.data import DataLoader
import psnr
import cv2
import scipy.io as sio
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def data_process_npy(data_in,mean,batch):
    data_out=data_in.cpu().numpy().astype(np.float32)
    data_out=np.transpose(data_out,(0,2,3,1))
    data_out=np.squeeze(data_out)
    data_out=data_out+mean
    data=data_out*255.
    return data
def save_mat(data_in):
    data_out=data_in.cpu().numpy()
    data_out=np.transpose(data_out,(0,2,3,1))
    return data_out

class Trainer:
    def __init__(self):
        self.epoch = 1000
        self.batch_size = 32
        self.lr = 0.0001
        self.best_psnr=0
        self.best_model=0
        print("===> Loading datasets")
        self.train_set = cudataset('data_x','data_y','label')
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

        print("===> Building model")
        self.model = CUNet_plus()
        self.model = self.model.cuda()
        self.criterion = nn.MSELoss(reduction='mean')

        print("===> Setting Optimizer")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.train_loss = []
        self.val_psnr = []
        self.val_ssim = []

        if os.path.exists('model/SR/latest.pth'):
            print('===> Loading pre-trained model...')
            state = torch.load('model/SR/latest.pth')
            self.train_loss = state['train_loss']
            self.model.load_state_dict(state['model'])

    def train(self):
        seed = random.randint(1, 1000)
        print("===> Random Seed: [%d]" % seed)
        random.seed(seed)
        torch.manual_seed(seed)

        for ep in range(1, self.epoch + 1):
            print(self.lr)
            epoch_loss = []
            for batch, (lr, hr, rgb) in enumerate(self.train_loader):
                hr = hr.float()
                lr = lr.float()
                rgb = rgb.float()

                hr = hr.cuda()
                lr = lr.cuda()
                rgb = rgb.cuda()

                self.optimizer.zero_grad()
                torch.cuda.synchronize()
                start_time = time.time()
                z = self.model(lr, rgb)
                loss = self.criterion(z, hr)
                loss = loss * 1000
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

                torch.cuda.synchronize()
                end_time = time.time()

                if batch % 1000 == 0:
                    print('Epoch:{}\tcur/all:{}/{}\tAvg Loss:{:.4f}\tTime:{:.2f}'.format(ep, batch,len(self.train_loader),loss.item(),end_time - start_time))

            self.scheduler.step()
            self.train_loss.append(np.mean(epoch_loss))
            print(np.mean(epoch_loss))

            state = {
                'model': self.model.state_dict(),
                'train_loss': self.train_loss
            }
            if not os.path.exists("model/SR/"):
                os.makedirs("model/SR/")
            torch.save(state, os.path.join('model/SR/', 'latest.pth'))
            torch.save(state, os.path.join('model/SR/', str(ep) + '.pth'))
            matplotlib.use('Agg')
            plot_loss_list = self.train_loss
            fig1 = plt.figure()
            plt.plot(plot_loss_list)
            plt.savefig('SR_x4_loss.png')

            val_psnr, val_ssim = self.test(ep)
            self.val_psnr.append(val_psnr)
            self.val_ssim.append(val_ssim)
            fig2 = plt.figure()
            plt.plot(self.val_psnr)
            plt.savefig('SR_x4_psnr.png')
            fig3 = plt.figure()
            plt.plot(self.val_ssim)
            plt.savefig('SR_x4_ssim.png')
            plt.close('all')
        print('===> Finished Training!')

    def test(self, i):
        test_set = cudatatest(scale=4)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        model = CUNet_plus()
        model = model.cuda()
        lr_img = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
                  'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
                  'twentyone', 'twentytwo', 'twentythree', 'twentyfour', 'twentyfive', 'twentysix', 'twentyseven',
                  'twentyeight', 'twentynine', 'thirty', 'thirtyone', 'thirtytwo', 'thirtythree', 'thirtyfour',
                  'thirtyfive', 'thirtysix', 'thirtyseven', 'thirtyeight', 'thirtynine', 'forty', 'fortyone',
                  'fortytwo', 'fortythree', 'fortyfour', 'fortyfive', 'fortysix', 'fortyseven', 'fortyeight',
                  'fortynine', 'fifty', 'fiftyone', 'fiftytwo', 'fiftythree', 'fiftyfour', 'fiftyfive', 'fiftysix',
                  'fiftyseven', 'fiftyeight', 'fiftynine', 'sixty', 'sixtyone', 'sixtytwo', 'sixtythree', 'sixtyfour',
                  'sixtyfive', 'sixtysix', 'sixtyseven', 'sixtyeight', 'sixtynine', 'seventy', 'seventyone',
                  'seventytwo', 'seventythree', 'seventyfour', 'seventyfive', 'seventysix', 'seventyseven',
                  'seventyeight', 'seventynine', 'eighty', 'eightyone', 'eightytwo', 'eightythree', 'eightyfour',
                  'eightyfive', 'eightysix', 'eightyseven', 'eightyeight', 'eightynine', 'ninety', 'ninetyone',
                  'ninetytwo', 'ninetythree', 'ninetyfour', 'ninetyfive', 'ninetysix', 'ninetyseven', 'ninetyeight',
                  'ninetynine', 'hundred']
        mean_LR=[0.6758,0.5187,0.5025,0.5309,0.5868,0.4299,0.4786]
        mean_guide=[0.3068,0.3956,0.5767,0.1755,0.3952,0.4296,0.3157]
        state = torch.load('model/SR/latest.pth')
        model.load_state_dict(state['model'])
        model.eval()

        dic = {}
        dig = {}
        PSNR = []
        SSIM = []

        with torch.no_grad():

            for batch_test, (lr, hr, rgb) in enumerate(test_loader):
                hr = hr.float()
                lr = lr.float()
                rgb = rgb.float()

                hr = hr.cuda()
                lr = lr.cuda()
                rgb = rgb.cuda()

                sr= model(lr,rgb)
                

                dic[lr_img[batch_test] + '_sr'] = save_mat(sr)
                dig[lr_img[batch_test] + '_gt'] = save_mat(hr)

                sr = data_process_npy(sr, mean_LR[batch_test], batch_test)
                hr = data_process_npy(hr, mean_guide[batch_test], batch_test)
                

                PSNR.append(psnr.psnr(sr, hr))
                SSIM.append(psnr.SSIM(sr, hr))
            ave_psnr = np.mean(PSNR)
            ave_ssim = np.mean(SSIM)
            if not os.path.exists('result/SR/MAT/'):
                os.makedirs('result/SR/MAT/')
            if ave_psnr > self.best_psnr:
                self.best_psnr = ave_psnr
                self.best_model = i
                sio.savemat('result/SR/MAT/best.mat', dic)
                sio.savemat('result/SR/MAT/gt.mat', dig)

                torch.save(state,os.path.join('model/SR/', 'best.pth'))
            sio.savemat('result/SR/MAT/'+str(i)+'.mat', dic)
            print('model' + str(i) + ' ### ' + 'PSNR:', str(ave_psnr)[:8], 'SSIM:', str(ave_ssim)[:8],'### best psnr is', str(self.best_psnr)[:8], 'of model', str(self.best_model))

        return ave_psnr, ave_ssim
if __name__ == '__main__':
    hhh = Trainer()
    hhh.train()