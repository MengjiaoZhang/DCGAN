"""Tricks

Reference:
https://github.com/soumith/ganhacks

without label flipping for G and D
other tricks are the same with train_with_flip_D.py and train_with_flip_G_D.py

"""


import itertools
import numpy as np
from torch.autograd import Variable

import torch
import torchvision.utils as vutils
from conf import Args
import os
import matplotlib.pyplot as plt

if not os.path.isdir('MNIST_DCGAN_results_D_SGD'):
    os.mkdir('MNIST_DCGAN_results_D_SGD')
if not os.path.isdir('MNIST_DCGAN_results_D_SGD/Random_results'):
    os.mkdir('MNIST_DCGAN_results_D_SGD/Random_results')
if not os.path.isdir('MNIST_DCGAN_results_D_SGD/Fixed_results'):
    os.mkdir('MNIST_DCGAN_results_D_SGD/Fixed_results')





def train(netG, netD, optimizerD, optimizerG, dataloader, criterion, num_epochs, device):
    img_list = []
    G_losses = []
    D_losses = []
    train_hist = {}
    num_iter = 0
    iters = 0
    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    fixed_noise = torch.randn(64, Args.dim_z, 1, 1, device=device)
    def show_result(num_epoch, show=False, save=False, path='result.png', isFix=False):
        z_ = torch.randn((5 * 5, 100), device=device).view(-1, 100, 1, 1)
        # with torch.no_grad():
        #     z_ = Variable(z_)

        netG.eval()
        if isFix:
            test_images = netG(fixed_noise)
        else:
            test_images = netG(z_)
        netG.train()

        size_figure_grid = 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(5 * 5):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

        label = 'Epoch {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(path)


    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # use soft labels
            # real:0.9-1
            label = label * (0.9 + 0.1 * torch.rand(label.shape, device=device))

            #randomly flip the labels
            # label_indx = np.random.choice(len(label), int(0.05*len(label)),replace=False)
            # label[label_indx] = 1- label[label_indx]

            # print(label.shape)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # print(output.shape)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, Args.dim_z, 1, 1, device=device)
            # Generate fake image batch with G
            # print("noise",noise.shape)
            fake = netG(noise)
            # print("fake",fake.shape)
            label.fill_(fake_label)
            # use soft labels
            # fake 0-0.1
            label = 0.1 * torch.rand(label.shape,device=device)
            # randomly flip the labels

            # label_indx = np.random.choice(len(label), int(0.05 * len(label)), replace=False)
            # label[label_indx] = 1 - label[label_indx]

            # print("label",label.shape)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        p = 'MNIST_DCGAN_results_D_SGD/Random_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
        fixed_p = 'MNIST_DCGAN_results_D_SGD/Fixed_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
        show_result((epoch + 1), save=True, path=p, isFix=False)
        show_result((epoch + 1), save=True, path=fixed_p, isFix=True)

    train_hist["D_losses"]=D_losses
    train_hist["G_losses"]=G_losses
    return train_hist