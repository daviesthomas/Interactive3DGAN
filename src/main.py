# code to initiate training OR start interactive client

import kaolin as kal
import torch
from tqdm import tqdm
import os
import argparse

from nets import Generator
from trainer import GAN3DTrainer

def main():
    argparser = argparse.ArgumentParser(description='Implementation of interactive 3d modelling with GANs!')
    argparser.add_argument('--mode','-m',type=str, choices=['trainGAN','testGAN','trainPROJ','viewData','play'], required=True)
    argparser.add_argument('--resume', action='store_true')
    argparser.add_argument('--shapenetMeshPath', '-mp', type=str, default='/raid/thomas/data/meshDatasets/shapenet/ShapeNetCore.v1')
    argparser.add_argument('--shapenetVoxelPath','-vp', type=str, default='/raid/thomas/data/meshDatasets/shapenet/ShapeNetCore.v1.Voxels')
    argparser.add_argument('--logPath', '-l', type=str, default='/raid/thomas/InteractiveGANExperiments/base3DGAN')
    argparser.add_argument('--batchSize', '-bs', type=int, default=50)
    argparser.add_argument('--logEvery', type=int, default=10)
    argparser.add_argument('--saveEvery', type=int, default=10)
    argparser.add_argument('--epochs', '-e', type=int, default=40)
    
    args = argparser.parse_args()

    device = torch.device('cuda:0')

    if (args.mode == 'trainGAN' or args.mode == 'trainPROJ'):
        trainDataSet = kal.datasets.shapenet.ShapeNet_Voxels(
            args.shapenetMeshPath,
            args.shapenetVoxelPath,
            categories=['chair'],
            resolutions=[62]    # we default to 64 voxel size
        )

        trainDataLoader = torch.utils.data.DataLoader(
            trainDataSet, 
            batch_size= args.batchSize, 
            shuffle=True, 
            num_workers=8
        )

        if (args.mode == 'trainGAN'):  
            trainer = GAN3DTrainer(
                logDir = args.logPath,
                printEvery=args.logEvery, 
                resume=args.resume
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
    elif (args.mode == 'viewData'):
        trainDataSet = kal.datasets.shapenet.ShapeNet_Voxels(
            args.shapenetMeshPath,
            args.shapenetVoxelPath,
            categories=['chair'],
            resolutions=[62]    # we default to 64 voxel size
        )

        trainDataLoader = torch.utils.data.DataLoader(
            trainDataSet, 
            batch_size= args.batchSize, 
            shuffle=True, 
            num_workers=8
        )

        for i,sample in enumerate(trainDataLoader):
            shapes = sample['data']['62']

            for voxels in shapes:
                voxels = voxels[:-2,:-2,:-2]

                model = kal.transforms.voxelfunc.max_connected(voxels, 0.5)
                verts, faces = kal.conversions.voxelgrid_to_quadmesh(model)
                mesh = kal.rep.QuadMesh.from_tensors(verts,faces)
                #mesh.laplacian_smoothing(iterations=3)
                mesh.show()


main()