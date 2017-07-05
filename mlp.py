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
			nn.Linear(nz, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, input_size),
		)

	def forward(self, noise):
		x = self.main(noise)
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

		self.dropout_prob = 0.5

	def forward(self, x):
		output = self.main(x).mean(0).view(1)

		# x = nn.ReLU()(self.fc1(x))
		# x = nn.ReLU()(self.fc2(x))
		# x = nn.ReLU()(nn.Dropout(self.dropout_prob)(self.fc3(x)))
		# x = self.fc4(x).mean(0).view(1)

		return output

	def update_dropout_prob(self, update_factor):
		self.dropout_prob += update_factor

		if self.dropout_prob >= 1.0:
			self.dropout_prob = 1.0
