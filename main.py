import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import os
from network import Generator1
from network import Generator2
from network import Discriminator1, Discriminator2
from utils import ReplayBuffer
from utils import LambdaLR
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--save',type=str,default='model/',help='path to save models')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='dataset/train/', help='root directory of the dataset')
parser.add_argument('--Dlr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--Glr1', type=float, default=0.0003, help='initial learning rate')
parser.add_argument('--Glr2', type=float, default=0.0002, help='initial learning rate')

parser.add_argument('--pretrainedG1',type=str,default='',help='path to pretrained model netG_HR2LR')
parser.add_argument('--pretrainedG2',type=str,default='',help='path to pretrained model netG_LR2HR')
parser.add_argument('--pretrainedD1',type=str,default='',help='path to pretrained model netG_LR')
parser.add_argument('--pretrainedD2',type=str,default='',help='path to pretrained model netG_HR')

parser.add_argument('--decay_epoch', type=int, default=30,help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--decay_epoch1', type=int, default=2,help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=500, help='size of the data crop (squared assumed)')
parser.add_argument('--shortsize', type=int, default=50, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', default=True,action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--w_loss_cycle',type=float,default=5)
parser.add_argument('--w_loss_recover',type=float,default=5)
parser.add_argument('--w_loss_G1',type=float,default=0.01)
parser.add_argument('--w_loss_G2',type=float,default=0.01)

opt = parser.parse_args()
print(opt)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
print("define netwroks")
netG_HR2LR = Generator1(opt.input_nc, opt.output_nc,9)
netG_LR2HR = Generator2(opt.output_nc, opt.input_nc)
netD_LR = Discriminator1(opt.input_nc)
netD_HR = Discriminator2(opt.input_nc)


if opt.cuda:
    netG_HR2LR.cuda()
    netG_LR2HR.cuda()
    netD_LR.cuda()
    netD_HR.cuda()

# Losses
print("define losses")
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_L1 = torch.nn.L1Loss()
criterion_L2 = torch.nn.MSELoss()

optimizer_G1 = torch.optim.Adam(netG_HR2LR.parameters(),lr=opt.Glr1, betas=(0.5, 0.999))
optimizer_G2 = torch.optim.Adam(netG_LR2HR.parameters(),lr=opt.Glr2, betas=(0.5, 0.999))
optimizer_D_LR = torch.optim.Adam(netD_LR.parameters(), lr=opt.Dlr, betas=(0.5, 0.999))
optimizer_D_HR = torch.optim.Adam(netD_HR.parameters(), lr=opt.Dlr, betas=(0.5, 0.999))

lr_scheduler_G1 = torch.optim.lr_scheduler.StepLR(optimizer_G1, step_size=1, gamma=0.1)
lr_scheduler_G2 = torch.optim.lr_scheduler.LambdaLR(optimizer_G2,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_LR = torch.optim.lr_scheduler.LambdaLR(optimizer_D_LR,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_HR = torch.optim.lr_scheduler.LambdaLR(optimizer_D_HR,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_HR = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_LR = Tensor(opt.batchSize, opt.output_nc, opt.shortsize, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_LR_buffer = ReplayBuffer()
fake_HR_buffer = ReplayBuffer()

# Dataset loader
print("loading dataset")
transforms_=[ transforms.ToTensor(),
              transforms.Normalize([0.5],[0.5])
              ]

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
print(len(dataloader))


###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        n_iter=epoch*len(dataloader)+i
        # Set model input
        real_HR = Variable(input_HR.copy_(batch['HR']))
        real_LR = Variable(input_LR.copy_(batch['LR']))

        ###### Generators A2B and B2A ######
        optimizer_G1.zero_grad()
        optimizer_G2.zero_grad()

        #optimizer_G.zero_grad()

        # GAN LR loss
        fake_LR = netG_HR2LR(real_HR)
        pred_fake = netD_LR(fake_LR)

        loss_GAN_HR2LR = criterion_GAN(pred_fake, target_real)

        # GAN HR loss
        fake_HR = netG_LR2HR(real_LR)
        loss_GAN_LR2HR = criterion_GAN(netD_HR(fake_HR),target_real)

        # recover loss
        recovered_HR = netG_LR2HR(netG_HR2LR(real_HR))
        loss_recover= criterion_L1(recovered_HR, real_HR)

        # Cycle loss
        recovered_LR=netG_HR2LR(netG_LR2HR(real_LR))
        loss_cycle=criterion_cycle(recovered_LR,real_LR)


        # Total G loss
        loss_G1 = opt.w_loss_G1*loss_GAN_HR2LR  + opt.w_loss_G2*loss_GAN_LR2HR + opt.w_loss_cycle*loss_cycle\
                 + opt.w_loss_recover*loss_recover
        loss_G1.backward(retain_graph=True)

        loss_G2 = opt.w_loss_G1*loss_GAN_HR2LR + opt.w_loss_G2*loss_GAN_LR2HR + opt.w_loss_cycle*loss_cycle\
                 + opt.w_loss_recover*loss_recover
        loss_G2.backward()

        optimizer_G1.step()
        optimizer_G2.step()


        ###################################

        ###### Discriminator_LR ######
        optimizer_D_LR.zero_grad()

        # Real loss
        pred_real = netD_LR(real_LR)
        loss_D1_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_LR = fake_LR_buffer.push_and_pop(fake_LR)
        pred_fake = netD_LR(fake_LR.detach())
        loss_D1_fake = criterion_GAN(pred_fake, target_fake)

        # Total D loss
        loss_D_LR = loss_D1_real + loss_D1_fake
        loss_D_LR.backward()

        optimizer_D_LR.step()

        ###################################

        ###### Discriminator_LR ######
        optimizer_D_HR.zero_grad()

        # real loss
        pred_real = netD_HR(real_HR)
        loss_D2_real = criterion_GAN(pred_real, target_real)

        # fake loss
        fake_HR = fake_HR_buffer.push_and_pop(fake_HR)
        pred_fake = netD_HR(fake_HR.detach())
        loss_D2_fake = criterion_GAN(pred_fake, target_fake)

        # total D2 loss
        loss_D_HR = loss_D2_real + loss_D2_fake
        loss_D_HR.backward()

        optimizer_D_HR.step()

        ###################################

        if i % 10 == 0:
            print("Epoch[{}/{}]({}/{}) --loss_G: {:.4f} | loss_G1_GAN: {:.4f} | loss_G2_GAN: {:.4f} | loss_G_recover: {:.4f} | "
                  "loss_G_cycle: {:.4f} | loss_D_LR: {:.4f} | loss_D_HR: {:.4f} | "
                  "lr_G: {:.4f}".format(
                    epoch, opt.n_epochs, i, len(dataloader), loss_G1.data[0],
                loss_GAN_HR2LR.data[0] , loss_GAN_LR2HR.data[0] , loss_recover.data[0],
                loss_cycle.data[0], loss_D_LR.data[0],
                loss_D_HR.data[0],  lr_scheduler_G1.get_lr()[0]))
        if n_iter%100==0:
            #Save models checkpoints
            torch.save(netG_HR2LR.state_dict(), opt.save+'netG_HR2LR'+'_'+str(n_iter).zfill(4)+'.pth')
            torch.save(netG_LR2HR.state_dict(), opt.save+'netG_LR2HR'+'_'+str(n_iter).zfill(4)+'.pth')
            torch.save(netD_LR.state_dict(), opt.save + 'netD_LR'+'_'+str(n_iter).zfill(4)+'.pth')
            torch.save(netD_HR.state_dict(), opt.save + 'netD_HR' + '_' + str(n_iter).zfill(4) + '.pth')

    # Update learning rates
    lr_scheduler_G1.step()
    lr_scheduler_G2.step()
    lr_scheduler_D_HR.step()
    lr_scheduler_D_LR.step()

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)


