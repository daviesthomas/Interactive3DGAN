import argparse
import json
import os
import torch 
from tqdm import tqdm 
import pickle 

import kaolin as kal 
from torch.utils.tensorboard import SummaryWriter

from nets import Generator, Discriminator, Projector

class GAN3DTrainer(object):
    def __init__(self, logDir, printEvery=1, resume=False, useTensorboard=True):
        super(GAN3DTrainer, self).__init__()

        self.logDir = logDir

        self.currentEpoch = 0
        self.totalBatches = 0

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

        self.useTensorboard = useTensorboard
        self.tensorGraphInitialized = False
        self.writer = None
        if useTensorboard:
            self.writer = SummaryWriter(os.path.join(self.logDir,'tensorboard'))

    
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
                torch.ones(data['62'].shape[0], 200) * 0.33
            ).to(self.device)

            fakeVoxels = self.G(z)
            fakeD = self.D(fakeVoxels)

            # https://arxiv.org/pdf/1706.05170.pdf (IV. Methods, A. Training the gen model)
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
                
                if (self.useTensorboard):   
                    self.writer.add_scalar('GenLoss/train', lossG, numBatches + self.totalBatches)
                    self.writer.add_scalar('DisLoss/train', lossD, numBatches + self.totalBatches)
                    self.writer.add_scalar('GenAcc/train', accG, numBatches + self.totalBatches)
                    self.writer.add_scalar('DisAcc/train', accD, numBatches + self.totalBatches)
                    self.writer.flush()

                    if not self.tensorGraphInitialized:   
                        #TODO: why can't I push graph? 
                        tempZ = torch.autograd.Variable(torch.rand(data['62'].shape[0],200,1,1,1)).cuda(1)             
                        self.writer.add_graph(self.G.module, tempZ)
                        self.writer.flush()

                        self.writer.add_graph(self.D.module, fakeVoxels)
                        self.writer.flush()
                        
                        self.tensorGraphInitialized = True

        #self.trainLoss.append(epochLoss)
        self.currentEpoch += 1
        self.totalBatches += numBatches

    def save(self):
        logTable = {
            'epoch': self.currentEpoch,
            'totalBatches': self.totalBatches
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
        self.totalBatches = runData['totalBatches']

#essentially training as an Autoencoder with fixed decoder
class ProjectorTrainer(object):
    def __init__(self, logDir, printEvery=1, resume=False, lossRatio = 0.0, useTensorboard=True):
        super().__init__()

        self.printEvery = printEvery
        self.logDir = logDir
        self.lossRatio = lossRatio  # (1-a)*dissimLoss + a*realismLoss
        self.currentEpoch = 0
        self.totalBatches = 0

        self.P = Projector()
        # pre-trained G and D !
        self.G = Generator()    
        self.D = Discriminator()

        # once hook is attached, activations will be pushed to self.activations
        self.D.attachLayerHook(self.D.layer3)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')

            self.G = self.G.to(self.device)
            self.D = self.D.to(self.device)
            self.P = self.P.to(self.device)

            # parallelize models on both devices, splitting input on batch dimension
            self.G = torch.nn.DataParallel(self.G, device_ids=[0,1])
            self.D = torch.nn.DataParallel(self.D, device_ids=[0,1])
            self.P = torch.nn.DataParallel(self.P, device_ids=[0,1])

        self.optim = torch.optim.Adam(
            self.P.parameters(), 
            lr=0.0005, 
            betas= (0.5,0.999)
        )

        self.load(resume=resume)

        self.useTensorboard = useTensorboard
        self.tensorGraphInitialized = False
        self.writer = None
        if useTensorboard:
            self.writer = SummaryWriter(os.path.join(self.logDir,'tensorboard'))

    ''' load generator and discriminator and (maybe) projector weights '''
    def load(self, resume=False):
        self.G.load_state_dict(torch.load(os.path.join(self.logDir, 'generator.pth')))
        self.D.load_state_dict(torch.load(os.path.join(self.logDir, 'discrim.pth')))
        
        if (resume):
            self.P.load_state_dict(torch.load(os.path.join(self.logDir, 'projector.pth')))
            self.optim.load_state_dict(torch.load(os.path.join(self.logDir, 'optimP.pth')))
            
            with open(os.path.join(self.logDir, 'recent-projector.log'), 'r') as f:
                runData = json.load(f)

            self.currentEpoch = runData['epoch']
            self.totalBatches = runData['totalBatches']


    def save(self):
        logTable = {
            'epoch': self.currentEpoch,
            'totalBatches': self.totalBatches
        }

        torch.save(self.P.state_dict(), os.path.join(self.logDir, 'projector.pth'))
        torch.save(self.optim.state_dict(), os.path.join(self.logDir, 'optimP.pth'))

        with open(os.path.join(self.logDir, 'recent-projector.log'), 'w') as f:
            f.write(json.dumps(logTable))

        tqdm.write('======== SAVED RECENT MODEL ========')

    def train(self, trainData):
        numBatches = 0
        self.P.train()

        for i, sample in enumerate(tqdm(trainData)):
            data = sample['data']

            self.optim.zero_grad()
            self.P.zero_grad()

            inputVoxels = torch.zeros(data['62'].shape[0], 64, 64, 64).to(self.device)
            inputVoxels[:, 1:-1, 1:-1, 1:-1] = data['62'].to(self.device)

            #TODO: randomly drop 50% of voxels.

            z = self.P(inputVoxels)

            outputVoxels = self.G(z)
            
            ### Realism loss: run through the GAN, use output as "Real" vs Not
            genD = self.D(outputVoxels)
            realD = self.D(inputVoxels)
            realismLoss = -torch.mean(torch.log(realD) + torch.log(1. - genD))
            
            ### Dissimilarity loss

            # Fetch layer 3 activations ("We specifically
            # select the output of the 256 × 8 × 8 × 8 layer")
            acts = self.D.module.activations

            # NOTE: assumes 2 GPUs...
            actGen = torch.nn.parallel.gather(acts[:2], 'cuda:0')
            actReal = torch.nn.parallel.gather(acts[-2:], 'cuda:0')
            self.D.module.activations = []

            dissimilarityLoss = torch.mean(torch.abs(actGen - actReal))
            
            loss = dissimilarityLoss*(1.0 - self.lossRatio) + realismLoss*(self.lossRatio) 

            self.P.zero_grad()
            loss.backward()
            self.optim.step()

            #log
            numBatches += 1
            if i % self.printEvery == 0:
                tqdm.write(f'[TRAIN] Epoch {self.currentEpoch:03d}, Batch {i:03d}: '
                           f'Dissim Loss: {float(dissimilarityLoss.item()):2.3f}, Realism = {float(realismLoss.item()):2.3f}')
                
                if (self.useTensorboard):   
                    self.writer.add_scalar('DissimLoss/train', dissimilarityLoss, numBatches + self.totalBatches)
                    self.writer.add_scalar('RealismLoss/train', realismLoss, numBatches + self.totalBatches)
                    self.writer.add_scalar('Loss/train',loss , numBatches + self.totalBatches)
                    self.writer.flush()

                    if not self.tensorGraphInitialized:   
                        self.writer.add_graph(self.P.module, torch.ones(inputVoxels.size))
                        self.writer.flush()   
 
                        self.writer.add_graph(self.G.module, torch.ones(z.size))
                        self.writer.flush()

                        self.writer.add_graph(self.D.module, torch.ones(outputVoxels.size))
                        self.writer.flush()
                        
                        self.tensorGraphInitialized = True

        #self.trainLoss.append(epochLoss)
        self.currentEpoch += 1
        self.totalBatches += numBatches




        