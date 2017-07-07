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
from torch.autograd import Variable
import numpy as np
import os, sys, math
import matplotlib.pyplot as plt

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
        print(batches[i].size())
        mnist = torch.cat((mnist, batches[i]), 0)

    print(mnist.size())

    output = tsne_model.fit_transform(mnist.numpy())

    print(output)

    print(output.shape)
    # sys.exit()
    i = 0
    # print(output[i*100:(i+1)*100, 0])
    # print(output[i*100:(i+1)*100, 1])
    # plt.plot(output[i*100:(i+1)*100, 0], output[i*100:(i+1)*100, 1], '+')

    # plt.plot(output[:,0], output[:,1], '+')

    for i in range(10):
        plt.plot(output[i*100:(i+1)*100, 0], output[i*100:(i+1)*100, 1], '+')

    plt.pause(10000)

    M = torch.FloatTensor(784, 1024).normal_(0,200)

    embedding = [torch.round(sigmoid(Variable(torch.mm(batches[i], M)))) for i in range(10)]

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

        # M = torch.FloatTensor(n_in, n_out).normal_(idx2,200)
        x = torch.FloatTensor(N, n_in).normal_(idx2,1)#.add(20)# + offset
        y1 = torch.round(nn.Sigmoid()(Variable(torch.mm(x, M))))
        h1 = hamming_distance(y1, y1)

        # M = torch.FloatTensor(n_in, n_out).normal_(idx,200)
        x = torch.FloatTensor(N, n_in).normal_(idx,1)#.add(20)# + offset
        y2 = torch.round(nn.Sigmoid()(Variable(torch.mm(x, M))))
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
