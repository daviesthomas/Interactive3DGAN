import argparse
import json
import os
import torch 
from tqdm import tqdm 
import pickle 

import kaolin as kal 

from nets import Generator, Discriminator

class GAN3DTrainer(object):
    def __init__(self, logDir, printEvery=1, resume=False):
        super(GAN3DTrainer, self).__init__()

        self.logDir = logDir

        self.currentEpoch = 0

        self.trainStats = {
            'lossG': [],
            'lossD': [],
            'accG': [],
            'accD': []
        }

        self.printEvery = printEvery

        self.G = Generator()
        self.D = Discriminator()

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')

            self.G = self.G.to(self.device)
            self.D = self.D.to(self.device)

            # parallelize models on both devices, splitting input on batch dimension
            self.G = torch.nn.DataParallel(self.G, device_ids=[0,1])
            self.D = torch.nn.DataParallel(self.D, device_ids=[0,1])

        # optim params direct from paper
        self.optimG = torch.optim.Adam(
            self.G.parameters(), 
            lr = 0.0025, 
            betas= (0.5, 0.999)
        )

        self.optimD = torch.optim.Adam(
            self.D.parameters(), 
            lr=0.00005, 
            betas= (0.5,0.999)
        )

        if resume:
            self.load()
    
    def train(self, trainData : torch.utils.data.DataLoader):
        epochLoss = 0.0
        numBatches = 0
        
        self.G.train()
        self.D.train()

        for i, sample in enumerate(tqdm(trainData)):
            data = sample['data']

            self.optimG.zero_grad()
            self.G.zero_grad()

            self.optimD.zero_grad()
            self.D.zero_grad()

            realVoxels = torch.zeros(data['62'].shape[0], 64, 64, 64).to(self.device)
            realVoxels[:, 1:-1, 1:-1, 1:-1] = data['62'].to(self.device)

            # discriminator train
            z = torch.normal(
                torch.zeros(data['62'].shape[0], 200), 
                torch.ones(data['62'].shape[0],200) * 0.33
            ).to(self.device)

            fakeVoxels = self.G(z)
            fakeD = self.D(fakeVoxels)
            realD = self.D(realVoxels)

            lossD = -torch.mean(torch.log(realD) + torch.log(1. - fakeD))
            accD = ((realD >= .5).float().mean() + (fakeD < .5).float().mean()) / 2.
            accG = (fakeD > .5).float().mean()

            # only train if Disc wrong enough :)
            if accD < .8:
                self.D.zero_grad()
                lossD.backward()
                self.optimD.step()

            # gen train
            z = torch.normal(
                torch.zeros(data['62'].shape[0], 200), 
                torch.ones(data['62'].shape[0],200) * 0.33
            ).to(self.device)

            fakeVoxels = self.G(z)
            fakeD = self.D(fakeVoxels)
            lossG = -torch.mean(torch.log(fakeD))

            self.D.zero_grad()
            self.G.zero_grad()
            lossG.backward()
            self.optimG.step()

            #log
            numBatches += 1
            if i % self.printEvery == 0:
                tqdm.write(f'[TRAIN] Epoch {self.currentEpoch:03d}, Batch {i:03d}: '
                           f'gen: {float(accG.item()):2.3f}, dis = {float(accD.item()):2.3f}')
                
                #self.trainStats['lossG'].append(lossG.to(torch.device('cpu')))
                #self.trainStats['lossD'].append(lossD.to(torch.device('cpu')))
                #self.trainStats['accG'].append(accG.to(torch.device('cpu')))
                #self.trainStats['accD'].append(accD.to(torch.device('cpu')))

        #self.trainLoss.append(epochLoss)
        self.currentEpoch += 1

    def save(self):
        logTable = {
            'epoch': self.currentEpoch
        }

        torch.save(self.G.state_dict(), os.path.join(self.logDir, 'generator.pth'))
        torch.save(self.D.state_dict(), os.path.join(self.logDir, 'discrim.pth'))
        torch.save(self.optimG.state_dict(), os.path.join(self.logDir, 'optimG.pth'))
        torch.save(self.optimD.state_dict(), os.path.join(self.logDir, 'optimD.pth'))

        with open(os.path.join(self.logDir, 'recent.log'), 'w') as f:
            f.write(json.dumps(logTable))

        pickle.dump(
            self.trainStats, 
            open(os.path.join(self.logDir,'trainStats.pkl'), 'wb')
        )

        tqdm.write('======== SAVED RECENT MODEL ========')

    def load(self):
        self.G.load_state_dict(torch.load(os.path.join(self.logDir, 'generator.pth')))
        self.D.load_state_dict(torch.load(os.path.join(self.logDir, 'discrim.pth')))
        self.optimG.load_state_dict(torch.load(os.path.join(self.logDir, 'optimG.pth')))
        self.optimD.load_state_dict(torch.load(os.path.join(self.logDir, 'optimD.pth')))
    
        with open(os.path.join(self.logDir, 'recent.log'), 'r') as f:
            runData = json.load(f)

        self.trainStats = pickle.load(
            open(os.path.join(self.logDir,'trainStats.pkl'), 'rb')
        )

        self.currentEpoch = runData['epoch']

        