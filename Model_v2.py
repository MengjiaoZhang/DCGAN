"""

Add dropout and BN on G

"""


from __future__ import print_function
import torch
import torch.nn as nn
from conf import Args

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # input: N, C_in, H_in, W_in
        # output: N, C_out, H_out, W_out
        # H_out=(H_in−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # W_out=(W_in−1)×stride[1]−2×padding[1]+dilation[1]×(kernel_size[1]−1)+output_padding[1]+1
        # input is Z, going into a convolution

        # Input size. 100 x 1 x 1
        self.dropout = nn.Dropout(p=0.3)
        self.ConvTran1 = nn.ConvTranspose2d(Args.dim_z, 512, 4, stride=1, padding=0, bias=False)
        self.BN1 = nn.BatchNorm2d(512)
        # state size. 512 x 4 x 4
        self.ConvTran2 = nn.ConvTranspose2d(512, 256, 4, stride=1, padding=0, bias=False)
        self.BN2 = nn.BatchNorm2d(256)
        self.Relu = nn.ReLU(True)
        # state size. 256 x 7 x 7
        self.ConvTran3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.BN3 = nn.BatchNorm2d(128)
        # state size. 128 x 14 x 14
        self.ConvTran4 = nn.ConvTranspose2d(128,  1, 4, stride=2, padding=1, bias=False)
        # self.ConvTran5 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()
        # state size. 1 x 28 x 28


    def forward(self, input):
        # print("input",input.shape)
        x1 = self.Relu(self.BN1(self.ConvTran1(input)))
        x1 = self.dropout(x1)
        # print("x1",x1.shape)
        x2 = self.Relu(self.BN2(self.ConvTran2(x1)))
        # x2 = self.dropout(x2)
        # print("x2",x2.shape)
        x3 = self.Relu(self.BN3(self.ConvTran3(x2)))
        # x3 = self.dropout(x3)
        # print("x3", x3.shape)
        output = self.tanh(self.ConvTran4(x3))
        # output = self.dropout(output)
        # print(output.shape)
        return output



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # input is (nc) x 28 x 28
        self.conv2d1 = nn.Conv2d(Args.num_c, 64, 5, stride=2, padding=2, bias=False)
        self.Relu = nn.LeakyReLU(0.2, inplace=True)
        # state size. 64 x 15 x 15
        self.conv2d2 = nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=False)
        self.BN2 = nn.BatchNorm2d(128)
        # # state size. 128 x 7 x 7
        self.conv2d3 = nn.Conv2d(128, 256, 7, stride=1, padding=0, bias=False)
        self.BN3 = nn.BatchNorm2d(256)
        # state size. 256 x 4 x 4
        # self.conv2d4 = nn.Conv2d(256, 512, 4, stride=1, padding=0, bias=False)
        self.dense = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # print(input.shape)
        x1 = self.Relu(self.conv2d1(input))
        # print("x1",x1.shape)
        x2 = self.Relu(self.conv2d2(x1))
        # x2 = self.Relu(self.BN2(self.conv2d2(x1)))
        # print("x2",x2.shape)
        # x3 = self.Relu(self.BN3(self.conv2d3(x2)))
        x3 = self.Relu(self.conv2d3(x2))
        # print("x3",x3.shape)
        # x4 = self.conv2d4(x3)
        # print("x4",x4.shape)
        x3 = torch.squeeze(x3)
        # print("x3", x3.shape)
        output = self.sigmoid(self.dense(x3))
        # print(output.shape)
        return output

