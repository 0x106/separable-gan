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

class Critic__(nn.Module):
	def __init__(self, input_size, nz, feature_size):
		super(Critic, self).__init__()

		self.fc1 = nn.Linear(input_size, feature_size)
		self.fc2 = nn.Linear(feature_size, feature_size)
		self.fc3 = nn.Linear(feature_size, feature_size)
		self.fc4 = nn.Linear(feature_size, 1)

	def forward(self, x, bernoulli):

		x = nn.ReLU()(self.fc1(x) * bernoulli)
		x = nn.ReLU()(self.fc2(x) * bernoulli)
		x = nn.ReLU()(self.fc3(x) * bernoulli)
		output = self.fc4(x).mean(0).view(1)

		return output

class Critic(nn.Module):
	def __init__(self, input_size, nz, feature_size):
		super(Critic, self).__init__()

		self.fc1 = nn.Linear(input_size, feature_size)
		self.fc2 = nn.Linear(feature_size, feature_size)
		self.fc3 = nn.Linear(feature_size, feature_size)
		self.fc4 = nn.Linear(feature_size, 1)

		#                                              ON    OFF
		# the probability a unit is OFF (i.e 0.1 --> [0.95, 0.05])
		self.drop_prob = 0.4

		# 0.1 --> 0.05 - 0.95
		# 0.2 --> 0.1 - 0.9
		# 0.3 --> 0.15 - 0.85
		# 0.4 --> 0.2 - 0.8
		# 0.5 --> 0.25 - 0.75

	def forward(self, x, bernoulli):

		x = nn.ReLU()(self.dropout(self.fc1(x), bernoulli))
		x = nn.ReLU()(self.dropout(self.fc2(x), bernoulli))
		x = nn.ReLU()(self.dropout(self.fc3(x), bernoulli))
		output = self.fc4(x).mean(0).view(1)

		return output

	def dropout(self, layer, bernoulli):
		probs = bernoulli * (1. - self.drop_prob) + (self.drop_prob / 2.)
		drop = torch.bernoulli(probs)
		layer = layer * drop
		return layer

class Autoencoder(nn.Module):
	def __init__(self, input_size, nz, feature_size):
		super(Autoencoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(input_size, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, feature_size*2),
			nn.ReLU(True),
			nn.Linear(feature_size*2, feature_size),
		)

		self.decoder = nn.Sequential(
			nn.Linear(feature_size, feature_size*2),
			nn.ReLU(True),
			nn.Linear(feature_size*2, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, input_size),
		)

	def forward(self, x):

		encoded = self.encoder(x)
		decoded = self.decoder(encoded)

		binary = torch.round(nn.Sigmoid()(encoded))

		return binary, decoded


class ClassifierEncoder(nn.Module):
	def __init__(self, input_size, nz, feature_size):
		super(ClassifierEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(input_size, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, feature_size*2),
			nn.ReLU(True),
			nn.Linear(feature_size*2, feature_size),
		)

		self.decoder = nn.Sequential(
			nn.Linear(feature_size, feature_size*2),
			nn.ReLU(True),
			nn.Linear(feature_size*2, feature_size),
			nn.ReLU(True),
			nn.Linear(feature_size, input_size + 4),
		)

	def forward(self, x):

		encoded = self.encoder(x)
		decoded = self.decoder(encoded)

		binary = torch.round(nn.Sigmoid()(encoded))

		return binary, decoded

class HammingEncoder(nn.Module):
	def __init__(self, input_size, nz, feature_size):
		super(HammingEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(input_size, 64),
			nn.ReLU(True),
			nn.Linear(64, 128),
			nn.ReLU(True),
			nn.Linear(128, 256),
			nn.ReLU(True),
			nn.Linear(256, feature_size)
		)

	def forward(self, x):

		x = self.encoder(x)
		return x
