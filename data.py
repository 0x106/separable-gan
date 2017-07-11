import numpy as np
import torch
import os, sys, math
import torchvision.datasets as dset
import torchvision.transforms as transforms

import util

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

		# self.means = [-0.5, 0.5]
		self.means = [-4, 4]
		self.vars = [0.5, 0.5]

		self.sample = torch.FloatTensor(2, self.arg.batch_size, self.arg.input_size).fill_(0)
		self.triplet = torch.FloatTensor(3, self.arg.batch_size, self.arg.input_size).fill_(0)

	def __len__(self):
		return self.arg.M

	def __next__(self):
		self.sample[0].normal_(self.means[0], self.vars[0])
		self.sample[1].normal_(self.means[1], self.vars[1])

		return self.sample

	def next(self):
		self.sample[0].normal_(self.means[0], self.vars[0])
		self.sample[1].normal_(self.means[1], self.vars[1])

		return self.sample

	def next_triplet(self):

		label = np.random.randint(2)

		# print(label, 1 - label)

		self.triplet[0].normal_(self.means[label], self.vars[label])
		self.triplet[1].normal_(self.means[label], self.vars[label])
		self.triplet[2].normal_(self.means[1 - label], self.vars[1 - label])

		return self.triplet

class MultiModalNormal(object):
	"""MultiModalNormal Sample Generator"""
	def __init__(self, arg):
		super(MultiModalNormal, self).__init__()

		self.arg = arg

		self.N = 4

		self.mean = 2
		self.var = 0.5

		self.sample = torch.FloatTensor(self.N, self.arg.batch_size, self.arg.input_size).fill_(0)

		self.mixed = torch.FloatTensor(self.arg.batch_size, self.arg.input_size).fill_(0)
		self.label = torch.LongTensor(self.arg.batch_size).fill_(0)


	def __len__(self):
		return self.arg.M

	def __next__(self):
		self.sample[0].normal_(self.mean, self.var)
		self.sample[1].normal_(-self.mean, self.var)
		self.sample[2].normal_(0,self.var)
		self.sample[3].normal_(0,self.var)

		self.sample[2,:,0] += self.mean
		self.sample[2,:,1] -= self.mean

		self.sample[3,:,0] -= self.mean
		self.sample[3,:,1] += self.mean

		return self.sample

	def next_mixed(self):

		sample = next(self)
		B = self.arg.batch_size // self.N

		for i in range(self.N):
			select = torch.randperm(self.arg.batch_size)[:B]
			self.mixed[i*B:(i+1)*B].copy_(sample[i][select])
			self.label[i*B:(i+1)*B].fill_(i)

		shuffle = util.shuffle(self.arg.batch_size)

		self.mixed = self.mixed[shuffle]
		self.label = self.label[shuffle]

		return self.mixed, self.label

class MNIST():
	def __init__(self, opt):

		self.B = opt.batch_size
		self.cuda = opt.cuda

		data_path = opt.dataroot
		if self.cuda:
			data_path = '/input'

		self.trData = dset.MNIST(data_path, train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))

		self.testData = dset.MNIST(data_path, train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))

		self.train_loader = torch.utils.data.DataLoader(self.trData, batch_size=self.B, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(self.testData, batch_size=len(self.testData), shuffle=True)

		self.train_iter = iter(self.train_loader)
		self.test_iter = iter(self.test_loader)

		self.num_examples = len(self.trData)
		self.labels = [ [] for i in range(10) ]

		for i in range(self.num_examples):
			self.labels[self.trData[i][1]].append(i)

		self.batch = torch.FloatTensor(self.B, 1, opt.input_size, opt.input_size)

	def __len__(self):
		return len(self.train_loader)

	def next(self):
		try:
			output = self.train_iter.next()
		except:
			self.train_iter = iter(self.train_loader)
			output = self.train_iter.next()

		return output

	def next_test(self):
		try:
			output = self.test_iter.next()
		except:
			self.test_iter = iter(self.test_loader)
			output = self.test_iter.next()

		return output

	def next_batch_from_class(self, selector):

		selection = np.random.permutation(len(self.labels[selector]))[:self.B]
		for i in range(self.B):
			self.batch[i].copy_(self.trData[self.labels[selector][selection[i]]][0])

		return (self.batch.view(self.batch.size(0), self.batch.size(2) * self.batch.size(3))).add(-1.0)




#
