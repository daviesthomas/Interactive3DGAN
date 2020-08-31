# code to initiate training OR start interactive client

import kaolin as kal
import torch
from tqdm import tqdm
import os
import argparse

from nets import Generator, Projector
from trainer import GAN3DTrainer, ProjectorTrainer

def loadDataset(meshPath, voxelPath, batchSize):
    dataSet = kal.datasets.shapenet.ShapeNet_Voxels(
        meshPath,
        voxelPath,
        categories=['chair'],
        resolutions=[62]    # we default to 64 voxel size
    )

    dataLoader = torch.utils.data.DataLoader(
        dataSet, 
        batch_size= batchSize, 
        shuffle=True, 
        num_workers=8
    )

    return dataLoader

def voxToMesh(voxels):
    model = kal.transforms.voxelfunc.max_connected(voxels, 0.5)
    verts, faces = kal.conversions.voxelgrid_to_quadmesh(model)
    mesh = kal.rep.QuadMesh.from_tensors(verts,faces)
    #mesh.laplacian_smoothing(iterations=3)
    return mesh

def main():
    argparser = argparse.ArgumentParser(description='Implementation of interactive 3d modelling with GANs!')
    argparser.add_argument('--mode','-m',type=str, choices=['trainGAN','testGAN','trainPROJ','testPROJ','viewData','play'], required=True)
    argparser.add_argument('--resume', action='store_true')
    argparser.add_argument('--shapenetMeshPath', '-mp', type=str, default='/raid/thomas/data/meshDatasets/shapenet/ShapeNetCore.v1')
    argparser.add_argument('--shapenetVoxelPath','-vp', type=str, default='/raid/thomas/data/meshDatasets/shapenet/ShapeNetCore.v1.Voxels')
    argparser.add_argument('--logPath', '-l', type=str, default='/raid/thomas/InteractiveGANExperiments/base3DGAN')
    argparser.add_argument('--batchSize', '-bs', type=int, default=50)
    argparser.add_argument('--logEvery', type=int, default=10)
    argparser.add_argument('--saveEvery', type=int, default=10)
    argparser.add_argument('--epochs', '-e', type=int, default=40)
    argparser.add_argument('--useTensorboard', action='store_true')
    
    args = argparser.parse_args()

    device = torch.device('cuda:0')

    if (args.mode == 'trainGAN' or args.mode == 'trainPROJ'):
        trainDataLoader = loadDataset(
            args.shapenetMeshPath,
            args.shapenetVoxelPath, 
            args.batchSize
        )

        if (args.mode == 'trainGAN'):  
            trainer = GAN3DTrainer(
                logDir = args.logPath,
                printEvery=args.logEvery, 
                resume=args.resume,
                useTensorboard=args.useTensorboard
            )
        elif (args.mode == 'trainPROJ'):
            trainer = ProjectorTrainer(
                logDir = args.logPath,
                printEvery=args.logEvery,
                resume=args.resume,
                useTensorboard=args.useTensorboard
            )
        
        for epoch in range(args.epochs):
            trainer.train(trainDataLoader)

            if epoch % args.saveEvery == 0:
                trainer.save()

    elif (args.mode == 'testGAN'):
        G = Generator().to(device)
        G = torch.nn.DataParallel(G, device_ids=[0,1])

        G.load_state_dict(torch.load(os.path.join(args.logPath, 'generator.pth')))
        G.eval()

        z = torch.normal(
            torch.zeros(args.batchSize, 200), 
            torch.ones(args.batchSize, 200) * .33).to(device)

        fake_voxels = G(z)[:,0]

        for i, model in enumerate(fake_voxels):
            model = model[:-2,:-2,:-2]
            model = kal.transforms.voxelfunc.max_connected(model, 0.5)
            verts, faces = kal.conversions.voxelgrid_to_quadmesh(model)
            mesh = kal.rep.QuadMesh.from_tensors(verts,faces)
            #mesh.laplacian_smoothing(iterations=3)
            mesh.show()

    elif (args.mode == 'testPROJ'):
        # we choose a random item from dataset. Push through projector then generator and compare!
        P = Projector().to(device)
        G = Generator().to(device)

        G = torch.nn.DataParallel(G, device_ids=[0,1])
        G.load_state_dict(torch.load(os.path.join(args.logPath, 'generator.pth')))

        P = torch.nn.DataParallel(P, device_ids=[0,1])
        P.load_state_dict(torch.load(os.path.join(args.logPath, 'projector.pth')))

        P.eval()
        G.eval()

        dataLoader = loadDataset(
            args.shapenetMeshPath,
            args.shapenetVoxelPath, 
            args.batchSize
        )

        for i,sample in enumerate(dataLoader):
            inputVoxels = torch.zeros(sample['data']['62'].shape[0], 64, 64, 64).to(device)
            inputVoxels[:, 1:-1, 1:-1, 1:-1] = sample['data']['62'].to(device)

            z = P(inputVoxels)
            outputVoxels = G(z).squeeze(dim=1)

            for i, originalVoxels in enumerate(inputVoxels):
                print("meshing")
                originalMesh = voxToMesh(originalVoxels)
                generatedMesh = voxToMesh(outputVoxels[i])

                originalMesh.show()
                generatedMesh.show()



    elif (args.mode == 'viewData'):
        dataLoader = loadDataset(
            args.shapenetMeshPath,
            args.shapenetVoxelPath, 
            args.batchSize
        )

        for i,sample in enumerate(dataLoader):
            shapes = sample['data']['62']

            for voxels in shapes:
                voxels = voxels[:-2,:-2,:-2]

                mesh = voxToMesh(voxels)
                mesh.show()




main()