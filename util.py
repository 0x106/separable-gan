#
# # ---------------------------------------------------------------------------- #
#
# from sklearn.manifold import TSNE
#
# tsne_model = TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
#
# sample = next(dataset)
# b0, _ = autoencoder(Variable(sample[0]))
# b1, _ = autoencoder(Variable(sample[1]))
# b2, _ = autoencoder(Variable(sample[2]))
# b3, _ = autoencoder(Variable(sample[3]))
#
# binary = torch.cat((b0, b1, b2, b3), 0)
#
# output = tsne_model.fit_transform(binary.data.numpy())
#
# plt.plot(output[:100,0], output[:100,1], '+')
# plt.plot(output[100:200,0], output[100:200,1], '+')
# plt.plot(output[200:300,0], output[200:300,1], '+')
# plt.plot(output[300:400,0], output[300:400,1], '+')
# plt.pause(1000)
#
# sys.exit()
# # ---------------------------------------------------------------------------- #
# # ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os, sys, math
import matplotlib.pyplot as plt

import mlp

def shuffle(size):
	retval = torch.LongTensor(size).copy_(torch.from_numpy(np.random.permutation(size)))
	return retval

# computes the set of pairwise hamming distances. Each row of x is a separate
# point
def hamming_distance(x, y):

	N = x.size(0)
	retval = torch.FloatTensor(N,N)
	for i in range(N):
		for k in range(N):
			retval[i,k] = ((x[i].ne(y[k])).data).sum() / float(x.size(1))

	retval = retval.mean()

	return retval

def random_projection_mnist(opt):

	import data
	# import seaborn as sns
	sigmoid = nn.Sigmoid()
	data = data.MNIST(opt)

	batches = [data.next_batch_from_class(i) for i in range(10)]

	from sklearn.manifold import TSNE
	tsne_model = TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)

	mnist = batches[0]
	for i in range(1,10):
		mnist = torch.cat((mnist, batches[i]), 0)
	output = tsne_model.fit_transform(mnist.numpy())

	print(mnist)

	# for i in range(10):
	#     plt.plot(output[i*100:(i+1)*100, 0], output[i*100:(i+1)*100, 1], '+')
	#
	# plt.pause(10000)

	M = torch.FloatTensor(784, 1024).normal_(0,200)

	embedding = [torch.round(sigmoid(Variable(torch.mm(batches[i], M)))) for i in range(10)]

	mnist = embedding[0]
	for i in range(1,10):
		mnist = torch.cat((mnist, embedding[i]), 0)
	print(mnist)
	output = tsne_model.fit_transform(mnist.data.numpy())

	for i in range(10):
		plt.plot(output[i*100:(i+1)*100, 0], output[i*100:(i+1)*100, 1], '+')

	plt.pause(10000)

	print(embedding[0][:10])
	print(embedding[4][:10])

	hamming = np.zeros((10,10,3))
	for i in range(10):
		for k in range(10):
			hamming[i,k,0] = hamming_distance(embedding[i], embedding[i])
			hamming[i,k,1] = hamming_distance(embedding[k], embedding[k])
			hamming[i,k,2] = hamming_distance(embedding[i], embedding[k])
			print(i,k,hamming[i,k])

	print(hamming[:,:,2])

	sns.heatmap(hamming[:,:,2])
	plt.pause(100)

def random_projection():

	n_in = 2
	n_out = 1024
	N = 10

	M = torch.FloatTensor(n_in, n_out).normal_(0,200)

	d = [[],[],[]]

	# offset = torch.FloatTensor(N, n_in).normal_(1,1)

	idx, idx2 = -10, -5
	for i in range(100):

		M = torch.FloatTensor(n_in, n_out).normal_(idx,200)
		x1 = torch.FloatTensor(N, n_in).normal_(idx,1)#.add(20)# + offset
		y1 = torch.round(nn.Sigmoid()(Variable(torch.mm(x1, M))))
		h1 = hamming_distance(y1, y1)

		M = torch.FloatTensor(n_in, n_out).normal_(idx2,200)
		x2 = torch.FloatTensor(N, n_in).normal_(idx2,1)#.add(20)# + offset
		y2 = torch.round(nn.Sigmoid()(Variable(torch.mm(x2, M))))
		h2 = hamming_distance(y2, y2)

		h3 = hamming_distance(y1, y2)

		d[0].append(h1)
		d[1].append(h2)
		d[2].append(h3)

		print(i, idx, idx2, h1, h2, h3)

		idx += (20. / 100)
		idx2 += (10. / 100)

	plt.plot(d[0])
	plt.plot(d[1])
	plt.plot(d[2])
	plt.show()
	plt.pause(100)
	sys.exit()

def compute_g(h, hp, hn, A, E, G, q):

	# every element of the batch
	for i in range(h.size(0)):
		F = torch.cat(((h[i].data).unsqueeze(0), (hp[i].data).unsqueeze(0), (hn[i].data).unsqueeze(0)), 0)
		C = torch.mm(A, F).transpose(1,0) + E

		config = C.max(1)[1]

		for k in range(q):
			G[:,i,k].copy_(A[config[k]])

	return Variable(G[0]), Variable(G[1]), Variable(G[2])

def triplet_loss(a,b,c):

	loss = (a.ne(b)).sum() - (a.ne(c)).sum() #+ 1
	return loss

def hdml(arg, dataset):
	plt.ion()

	network = mlp.HammingEncoder(int(arg.input_size), int(arg.nz), int(arg.feature_size))
	optimiser = optim.Adam(network.parameters(), lr = arg.lr)

	print(network)

	q = arg.feature_size

	A = torch.FloatTensor(8,3).fill_(0)
	E = torch.FloatTensor(q, 8).fill_(0)
	G = torch.FloatTensor(3, arg.batch_size, q).fill_(0)

	A[0,0] = -1; A[0,1] = -1; A[0,2] = -1
	A[1,0] = -1; A[1,1] = -1; A[1,2] = 1
	A[2,0] = -1; A[2,1] = 1; A[2,2] = -1
	A[3,0] = -1; A[3,1] = 1; A[3,2] = 1
	A[4,0] = 1; A[4,1] = -1; A[4,2] = -1
	A[5,0] = 1; A[5,1] = -1; A[5,2] = 1
	A[6,0] = 1; A[6,1] = 1; A[6,2] = -1
	A[7,0] = 1; A[7,1] = 1; A[7,2] = 1

	E[0,0] = 0; E[0,1] = -1; E[0,2] =  1; E[0,3] = 0;
	E[0,4] = 0; E[0,5] =  1; E[0,6] = -1; E[0,7] = 0;
	E = (E[0].unsqueeze(0)).expand_as(E)

	logs = [[], []]

	for epoch in range(arg.niter):

		network.zero_grad()

		triplet = dataset.next_triplet()

		# print(triplet.size())

		# plt.plot(triplet[0,:,0].numpy(), triplet[0,:,1].numpy(), '+')
		# plt.plot(triplet[1,:,0].numpy(), triplet[1,:,1].numpy(), '+')
		# plt.plot(triplet[2,:,0].numpy(), triplet[2,:,1].numpy(), '+')
		# plt.pause(0.2)
		# plt.clf()

		b = network(Variable(triplet[0]))
		bp = network(Variable(triplet[1]))
		bn = network(Variable(triplet[2]))

		h = torch.sign(b)
		hp = torch.sign(bp)
		hn = torch.sign(bn)

		g, gp, gn = compute_g(b, bp, bn, A, E, G, q)

		h_loss = triplet_loss(h, hp, hn) + 1
		g_loss = triplet_loss(g, gp, gn) - 1

		b.backward((h-g).data)
		bp.backward((hp-gp).data)
		bn.backward((hn-gn).data)

		optimiser.step()

		logs[0].append(h_loss.data[0])
		logs[1].append(g_loss.data[0])

		# print("{0} {1} {2}".format(epoch, h_loss.data[0], g_loss.data[0]))

		if epoch % 10 == 1:
			plt.plot(logs[0])
			plt.plot(logs[1])
			plt.pause(0.000001)
			plt.clf()

		print str(triplet[0,0,0]) + " " + str(triplet[0,0,1]) + " " + \
				  str(triplet[1,0,0]) + " " + str(triplet[1,0,1]) + " " + \
				  str(triplet[2,0,0]) + " " + str(triplet[2,0,1])

		print(h[0].data.unsqueeze(1).transpose(1,0))
		print(hp[0].data.unsqueeze(1).transpose(1,0))
		print(hn[0].data.unsqueeze(1).transpose(1,0))
		print(g[0].data.unsqueeze(1).transpose(1,0))
		print(gp[0].data.unsqueeze(1).transpose(1,0))
		print(gn[0].data.unsqueeze(1).transpose(1,0))

		print('----------------------------------------')
