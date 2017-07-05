import numpy as np
import torch
import os, sys, math
import torchvision.datasets as dset
import torchvision.transforms as transforms

class Circle2D():

	def __init__(self, opt):
		self.B = opt.batch_size
		self.num_partitions = 4

		self.theta_bound = math.pi / 6.
		# self.theta = [np.random.rand() * (math.pi * 2.) \
		# 							for i in range(self.num_partitions)]

		self.theta = [0., math.pi/2, math.pi, 3. * math.pi / 2.]
		self.theta += np.random.rand(4)

		self.radius = 6.
		self.sample = torch.FloatTensor(self.B, opt.input_size).fill_(0.)

		self.M = opt.M
		self.cuda = opt.cuda

		self.circle_N = 10000
		self.circle = torch.FloatTensor(self.circle_N, opt.input_size).fill_(0.)

		self.index = 0

	def __len__(self):
		return self.M

	def __next__(self):
	# def next(self, index):

		if self.cuda:
			self.sample = self.sample.cpu()

		# # single partition
		# if index == -1:
		#
		# 	theta = self.theta[0] + torch.FloatTensor(self.B, 1).uniform_(-self.theta_bound,self.theta_bound)
		# 	radius = self.radius + torch.FloatTensor(self.B, 1).normal_(0, 0.1)
		#
		# 	self.sample[:,0] = radius * torch.sin(theta)
		# 	self.sample[:,1] = radius * torch.cos(theta)
		#
		# else:
		#
		# if self.index >= self.num_partitions:
			# self.index = 0

		# print(index, self.theta[self.index])

		# theta = self.theta[index] + torch.FloatTensor(self.B, 1).uniform_(-self.theta_bound,self.theta_bound)
		# radius = self.radius + torch.FloatTensor(self.B, 1).normal_(0, 0.1)
		#
		# self.sample[:,0] = radius * torch.sin(theta)
		# self.sample[:,1] = radius * torch.cos(theta)
		#
		# self.index += 1

		# theta = self.theta[np.random.randint(self.num_partitions)] \
		# 				+ torch.FloatTensor(self.B, 1).uniform_(-self.theta_bound,self.theta_bound)
		theta = (np.random.rand() * (math.pi*2.)) \
						+ torch.FloatTensor(self.B, 1).uniform_(-self.theta_bound,self.theta_bound)
		radius = self.radius + torch.FloatTensor(self.B, 1).normal_(0, 0.1)

		self.sample[:,0] = radius * torch.sin(theta)
		self.sample[:,1] = radius * torch.cos(theta)

		if self.cuda:
			self.sample = self.sample.cuda()

		return self.sample

	def get_circle(self):

		theta = torch.rand(self.circle_N,1) * 2. * math.pi
		radius = self.radius + torch.FloatTensor(self.circle_N,1).normal_(0.,0.1)

		self.circle[:,0] = radius * torch.sin(theta)
		self.circle[:,1] = radius * torch.cos(theta)

		return self.circle

	def get_partitions(self):

		N = self.num_partitions * self.B

		output = torch.FloatTensor(N, 2)

		for i in range(self.num_partitions):
			# theta = self.theta[i] + torch.FloatTensor(self.B, 1).uniform_(-self.theta_bound,self.theta_bound)
			theta = (np.random.rand() * (math.pi*2.)) \
							+ torch.FloatTensor(self.B, 1).uniform_(-self.theta_bound,self.theta_bound)
			radius = self.radius + torch.FloatTensor(self.B, 1).normal_(0, 0.1)

			output[i*self.B:(i+1)*self.B ,0] = radius * torch.sin(theta)
			output[i*self.B:(i+1)*self.B ,1] = radius * torch.cos(theta)

		return output



class BiModalNormal(object):
    """2D BiModalNormal Sample Generator"""
    def __init__(self, arg):
        super(BiModalNormal, self).__init__()
        self.arg = arg

        self.means = [-2, 2]
        self.vars = [0.5, 0.5]

        self.sample = torch.FloatTensor(2, self.arg.B, self.arg.input_size).fill_(0)

    def __len__(self):
        return self.arg.M

    def __next__(self):















#
