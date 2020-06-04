from __future__ import print_function
from Model import Generator, weights_init, Discriminator
# from Model_v2 import Generator, weights_init, Discriminator
# from Model_v3 import Generator, weights_init, Discriminator
# from Model_v4 import Generator, weights_init, Discriminator
from conf import Args
from train_without_flip import train
from DataLoader import train_loader, test_loader


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import _pickle as pickle
import imageio
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# Create the generator
device = torch.device("cuda:0" if (torch.cuda.is_available() and Args.num_gpu > 0) else "cpu")

netG = Generator(Args.num_gpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (Args.num_gpu > 1):
    netG = nn.DataParallel(netG, list(range(Args.num_gpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
# print(netG)

# Create the Discriminator
netD = Discriminator(Args.num_gpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (Args.num_gpu > 1):
    netD = nn.DataParallel(netD, list(range(Args.num_gpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
# print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()


# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=Args.lr, betas=(Args.beta1, 0.999))
optimizerD = optim.SGD(netD.parameters(), lr=Args.lr_D, momentum=0.9)
optimizerG = optim.Adam(netG.parameters(), lr=Args.lr_G, betas=(Args.beta1, 0.999))

train_hist = train(netG, netD, optimizerD, optimizerG, train_loader, criterion, Args.num_epochs, device)

torch.save(netG.state_dict(), "MNIST_DCGAN_results_D_SGD/generator_param.pkl")
torch.save(netD.state_dict(), "MNIST_DCGAN_results_D_SGD/discriminator_param.pkl")
with open('MNIST_DCGAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='MNIST_DCGAN_results_D_SGD/MNIST_DCGAN_train_hist.png')

images = []
for e in range(Args.num_epochs):
    img_name = 'MNIST_DCGAN_results_D_SGD/Fixed_results/MNIST_DCGAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('MNIST_DCGAN_results_D_SGD/generation_animation.gif', images, fps=5)