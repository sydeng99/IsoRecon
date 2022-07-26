import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from network import Generator2 as Generator2
from network import Generator1 as Generator1
from datasets import ImageDataset



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='test_all/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--size', type=int, default=500, help='size of the data (squared assumed)')
parser.add_argument('--shortsize', type=int, default=50, help='size of the data (squared assumed)')
parser.add_argument('--cuda', default=True,action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

parser.add_argument('--modelpath', type=str, default='model/')
parser.add_argument('--modelname', type=str, default='Cyc_FIB_model')
parser.add_argument('--savepath',type=str,default='output/')
opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_LR2HR = Generator2(1, 1)
netG_HR2LR = Generator1(opt.input_nc, opt.output_nc,9)

if opt.cuda:
    netG_LR2HR.cuda()
    netG_HR2LR.cuda()

# Load state dicts
netG_LR2HR.load_state_dict(torch.load(opt.modelpath + opt.modelname + '/netG_LR2HR.pth' ))
netG_HR2LR.load_state_dict(torch.load(opt.modelpath + opt.modelname + '/netG_HR2LR.pth' ))
# Set model's test mode
netG_LR2HR.eval()
netG_HR2LR.eval()
# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_LR = Tensor(opt.batchSize, opt.input_nc, opt.shortsize, opt.size)
input_HR = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5]) ]

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
print(len(dataloader))
###################################

###### Testing######

# Create output dirs if they don't exist

if not os.path.exists(opt.savepath+'HR/'):
    os.makedirs(opt.savepath+'HR/')
if not os.path.exists(opt.savepath+'LR/'):
    os.makedirs(opt.savepath+'LR/')

for i, batch in enumerate(dataloader):
    # Set model input
    real_LR = Variable(input_LR.copy_(batch['LR']))
    real_HR = Variable(input_HR.copy_(batch['HR']))

    # Generate output
    fake_LR=0.5*(netG_HR2LR(real_HR)[-1].data+1.0)
    #print(fake_LR.shape)
    fake_HR=0.5*(netG_LR2HR(real_LR)[-1].data+1.0)
    # Save image files
    save_image(fake_LR, opt.savepath+'LR/'+opt.modelname+'_'+str(opt.epoch).zfill(3)+'_LR_%04d.png' % (i+1))
    save_image(fake_HR, opt.savepath+'HR/'+opt.modelname+'_'+str(opt.epoch).zfill(3)+'_HR_%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
