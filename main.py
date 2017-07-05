from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os, sys, math
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.ion()

import mlp, data#, mmd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--input_size', type=int, default=2, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--feature_size', type=int, default=512)
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate for Generator, default=0.00005')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
opt = parser.parse_args()

opt.M = 100

print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

nz = int(opt.nz)
feature_size = int(opt.feature_size)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.06)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

generator = mlp.Generator(opt.input_size, nz, feature_size)
generator.apply(weights_init)
print(generator)

critic = mlp.Critic(opt.input_size, nz, feature_size)
critic.apply(weights_init)
print(critic)

# dataset = data.Circle2D(opt)
dataset = data.BiModalNormal(opt)

noise = torch.FloatTensor(opt.batch_size, nz)
fixed_noise = torch.FloatTensor(opt.batch_size, nz).normal_(0, 1)
gen_input = torch.FloatTensor(opt.batch_size, opt.input_size)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    gen_input = gen_input.cuda()

optimizerD = optim.RMSprop(critic.parameters(), lr = opt.lr)
optimizerG = optim.RMSprop(generator.parameters(), lr = opt.lr)

def compute_errors(x, y):

    errors = [0,0,0,0]
    errors[0] = abs(x[:,0].mean() - y[:,0].mean())
    errors[1] = abs(x[:,1].mean() - y[:,1].mean())
    errors[2] = abs(x[:,0].std() - y[:,0].std())
    errors[3] = abs(x[:,1].std() - y[:,1].std())

    return errors

bernoulli = (torch.bernoulli(torch.FloatTensor(1, 1, opt.feature_size).fill_(0.5))).expand(1, opt.batch_size, opt.feature_size)
bernoulli = torch.cat((bernoulli, 1 - bernoulli), 0)

print(bernoulli)

gen_iterations = 0
logs = [[], [], []]
errors = [[],[],[],[]]
for epoch in range(opt.niter):
    # data_iter = iter(dataset)
    i = 0
    while i < len(dataset):
        ############################
        # (1) Update D network
        ###########################
        for p in critic.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < len(dataset):
            i, j = i+1, j+1

            for k in range(2):
                critic.zero_grad()

                # clamp parameters to a cube
                for p in critic.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                sample = next(dataset)
                errD_real = critic(Variable(sample[k]), Variable(bernoulli[k]))
                errD_real.backward(one)

                # train with fake
                gen_input = (( gen_input[0].copy_(sample[k].mean(0)) ).unsqueeze(0)).expand_as(gen_input)
                fake = Variable(generator( Variable(noise.normal_(0, 1), volatile = True), Variable(gen_input) ).data )
                errD_fake = critic(fake, Variable(bernoulli[k]))
                errD_fake.backward(mone)
                errD = errD_real - errD_fake

                optimizerD.step()

            # sys.exit()

        ############################
        # (2) Update G network
        ###########################
        for p in critic.parameters():
            p.requires_grad = False # to avoid computation

        sample = next(dataset)

        for k in range(2):

            generator.zero_grad()

            gen_input = ((gen_input[0].copy_(sample[k].mean(0))).unsqueeze(0)).expand_as(gen_input)
            fake = generator( Variable(noise.normal_(0, 1)), Variable(gen_input) )

            errG = critic(fake, Variable(bernoulli[k]))
            errG.backward(one)

            optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.niter, i, len(dataset), gen_iterations,
            errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

        logs[0].append(errD.data[0])
        logs[1].append(errG.data[0])

        # plt.subplot(211)
        plt.plot(sample[0,:,0].numpy(), sample[0,:,1].numpy(), '+')
        plt.plot(sample[1,:,0].numpy(), sample[1,:,1].numpy(), '+')
        # plt.plot(fake.data[:,0].numpy(), fake.data[:,1].numpy(), '+')
        sample = next(dataset)
        for k in range(2):
            gen_input = ((gen_input[0].copy_(sample[k].mean(0))).unsqueeze(0)).expand_as(gen_input)
            gen_inputv = Variable(gen_input)

            fake = generator( Variable(noise.normal_(0, 1)), gen_inputv ).data

            plt.plot(fake[:,0].numpy(), fake[:,1].numpy(), '+')

        # plt.subplot(212)
        # plt.plot(logs[0][-100:]); plt.plot(logs[1][-100:])
        plt.pause(0.01)
        plt.clf()

    # do checkpointing
    # torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    # torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
