import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math, sys
import numpy as np

class Generator(nn.Module):
	def __init__(self, input_size, nz, feature_size):
		super(Generator, self).__init__()

		self.main = nn.Sequential(
			nn.Linear(nz+input_size, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, input_size),
		)

	def forward(self, noise, class_input):
		x = torch.cat((noise, class_input), 1)
		x = self.main(x)
		return x

class Critic(nn.Module):
	def __init__(self, input_size, nz, feature_size):
		super(Critic, self).__init__()

		self.main = nn.Sequential(
			nn.Linear(input_size, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, feature_size),
			nn.Dropout(0.9),
			nn.ReLU(True),
			nn.Linear(feature_size, feature_size),
			nn.Dropout(0.9),
			nn.ReLU(True),
			nn.Linear(feature_size, 1)
		)

		self.fc1 = nn.Linear(input_size, feature_size)
		self.fc2 = nn.Linear(feature_size, feature_size)
		self.fc3 = nn.Linear(feature_size, feature_size)
		self.fc4 = nn.Linear(feature_size, 1)

	def forward(self, x, bernoulli):
		# output = self.main(x).mean(0).view(1)

		x = nn.ReLU()(self.fc1(x) * bernoulli)
		x = nn.ReLU()(self.fc2(x) * bernoulli)
		x = nn.ReLU()(self.fc3(x) * bernoulli)
		output = self.fc4(x).mean(0).view(1)

		# x = (self.fc1(x) * bernoulli)
		# print(x)
		# print('----------------')
		# x = (self.fc2(x) * bernoulli)
		# print(x)
		# print('----------------')
		# x = (self.fc3(x) * bernoulli)
		# print(x)
		# print('----------------')
		# output = self.fc4(x).mean(0).view(1)

		return output
