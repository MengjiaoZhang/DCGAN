"""
Model.py: add BN on G
Model_2.py: add dropout and BN on G
Model_3.py: add dropout on D; add BN on G
Model_4.py: add BN on D and G
"""



from __future__ import print_function
from Model import Generator, weights_init, Discriminator
# from Model_v2 import Generator, weights_init, Discriminator
# from Model_v3 import Generator, weights_init, Discriminator
#from Model_v4 import Generator, weights_init, Discriminator
from conf import Args
from train_with_flip_D import train as train1
from train_with_flip_G_D import train as train2
from train_without_flip import train as train3

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

trains = {
    'with_flip_G_D': train2,
    'with_flip_D': train1,
    'without_flip': train3
}

train = trains[Args.file_name]


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
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
    optimizerD = optim.Adam(netD.parameters(), lr=Args.lr_D, betas=(Args.beta1, 0.999))
    # optimizerD = optim.SGD(netD.parameters(), lr=0.01, momentum=0.9)
    optimizerG = optim.Adam(netG.parameters(), lr=Args.lr_G, betas=(Args.beta1, 0.999))

    train_hist = train(netG, netD, optimizerD, optimizerG, train_loader, criterion, Args.num_epochs, device)

    torch.save(netG.state_dict(), "MNIST_DCGAN_results/generator_param_" + Args.file_name +"_" + ".pkl")
    torch.save(netD.state_dict(), "MNIST_DCGAN_results/discriminator_param_" + Args.file_name +"_" +".pkl")
    with open('MNIST_DCGAN_results/train_hist.pkl_' + Args.file_name +'_','wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path="MNIST_DCGAN_results/MNIST_DCGAN_train_hist_" + Args.file_name +"_" +".png")

    images = []
    for e in range(Args.num_epochs):
        img_name = 'MNIST_DCGAN_results/Fixed_results/MNIST_DCGAN_' + Args.file_name +'_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave('MNIST_DCGAN_results/generation_animation_' + Args.file_name +'_' +'.gif', images, fps=5)
