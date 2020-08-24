import torch
import torch.nn as nn

import torchsummary

# direct implementation of: http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf
class Generator(torch.nn.Module):
    def __init__(self, latentSize=200, leak=0.0):
        super(Generator, self).__init__()

        self.latentSize = latentSize

        if leak > 0.0:
            self.useLeakyRelu = True
            self.leak = leak
        else:
            self.leak=0.0
            self.useLeakyRelu = False

        self.model = nn.Sequential(
            nn.ConvTranspose3d(self.latentSize,512,4,2,0),
            self._normalization(512),
            self._activation(), 

            nn.ConvTranspose3d(512, 256, 4, 2, 1),
            self._normalization(256),
            self._activation(), 

            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            self._normalization(128),
            self._activation(), 

            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            self._normalization(64),
            self._activation(), 

            nn.ConvTranspose3d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def _normalization(self, numFeatures:int):
        return nn.BatchNorm3d(numFeatures)

    def _activation(self):
        return nn.ReLU() if not self.useLeakyRelu else nn.LeakyReLU(self.leak)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1, 1)  
        return self.model(x)

class Discriminator(torch.nn.Module):
    def __init__(self, voxelSize = 64):
        super(Discriminator, self).__init__()

        self.useLeakyRelu = True
        self.leak = 0.2
        self.dropoutRate = 0.5
        self.voxelSize = 64

        self.model = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2 ,1),
            self._normalization(64),
            self._activation(),
            nn.Dropout3d(0.5),

            nn.Conv3d(64, 128, 4, 2, 1),
            self._normalization(128),
            self._activation(),
            nn.Dropout3d(0.5),

            nn.Conv3d(128, 256, 4, 2, 1),
            self._normalization(256),
            self._activation(),
            nn.Dropout3d(0.5),

            nn.Conv3d(256, 512, 4, 2, 1),
            self._normalization(512),
            self._activation(),
            nn.Dropout3d(0.5),

            nn.Conv3d(512, 1, 4, 2, 0),
            nn.Sigmoid(),
        )
    
    def _normalization(self, numFeatures:int):
        return nn.BatchNorm3d(numFeatures)

    def _activation(self):
        return nn.ReLU() if not self.useLeakyRelu else nn.LeakyReLU(self.leak)

    def forward(self, x):
        x = x.view(-1, 1, self.voxelSize, self.voxelSize, self.voxelSize)
        y = self.model(x)
        return y.view(-1, y.size(1))

# direct implementation of: https://arxiv.org/pdf/1706.05170.pdf
class Projector(torch.nn.Module):
    def __init__(self, latentSize = 200, voxelSize=64):
        super(Projector, self).__init__()

        self.leak = 0.2
        self.useLeakyRelu = True
        self.latentSize = latentSize
        self.voxelSize = 64

        self.model = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2 ,1),
            self._normalization(64),
            self._activation(),

            nn.Conv3d(64, 128, 4, 2, 1),
            self._normalization(128),
            self._activation(),

            nn.Conv3d(128, 256, 4, 2, 1),
            self._normalization(256),
            self._activation(),

            nn.Conv3d(256, 512, 4, 2, 1),
            self._normalization(512),
            self._activation(),

            nn.Conv3d(512, self.latentSize, 4, 2, 0),
            nn.Sigmoid()
        )

    def _normalization(self, numFeatures:int):
        return nn.BatchNorm3d(numFeatures)

    def _activation(self):
        return nn.ReLU() if not self.useLeakyRelu else nn.LeakyReLU(self.leak)

    def forward(self, x):
        y = self.model(x)
        return y.view(-1, y.size(1))

# simple test to catch any pytorch asserts :) 
# also print all model stats!
if __name__ == "__main__":

    batchSize = 16
    latentSize = 200

    G = Generator(latentSize).cuda(0)
    D = Discriminator().cuda(0)
    P = Projector().cuda(0)

    G = torch.nn.DataParallel(G, device_ids=[0,1])
    D = torch.nn.DataParallel(D, device_ids=[0,1])
    P = torch.nn.DataParallel(P, device_ids=[0,1])

    # latent vector (random!)
    z = torch.autograd.Variable(torch.rand(batchSize,latentSize,1,1,1)).cuda(1)

    # pass to generator, yielding Voxel
    X = G(z)

    # pass to discriminator
    D_X = D(X)
    # and projector
    P_X = P(X)

    # verify shapes
    print(X.shape, D_X.shape, P_X.shape)   # (BATCH, 1, 64, 64, 64) (16, 1) (16, 200)

    print("######## GENERATOR SUMMARY ########")
    torchsummary.summary(G, (latentSize, 1, 1, 1))
    print("######## DISCRIMINATOR SUMMARY #######")
    torchsummary.summary(D, (1, 64, 64, 64))
    print("######## PROJECTOR SUMMARY ########")
    torchsummary.summary(P, (1, 64, 64, 64))