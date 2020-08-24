# Interactive3DGAN
Implementation of the paper: Interactive 3D Modeling with a Generative Adversarial Network

## Dependencies
- pytorch
- kaolin (for convenient data loader)

## How To

First download all the data! We depend on ShapenetCoreV.1 dataset and the R2N2 derivative set.



- ShapeNet rendered images http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- ShapeNet voxelized models http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
- 

### conda quick setup
This conda environment assumes cuda 10.2 installed, and intalls torch 1.6 (or latest at time of reading). Simply force versions if required...

  conda create -name interactive3DGan pytorch torchvision cudatoolkit=10.1 pytorch3d trimesh scipy tqdm matplotlib networkx pyglet -c pytorch,conda-forge

#### PPTK Dependency
Annoyingly there is a dependency on pptk within kaolin (many hidden dependencies it seems...). PPTK is not supported on python 3.8 since no wheel is uploaded onto pypi. We must instead manually install. Luckily we can simply download the "3.7" wheel and rename to "3.8" and install directly :) 

Download from: https://pypi.org/project/pptk/#modal-close

rename from 'pptk-0.1.0-cp37-none-manylinux1_x86_64.whl' to 'pptk-0.1.0-cp38-none-manylinux1_x86_64.whl'

install with ''' pip install ./pptk-0.1.0-cp38-none-manylinux1_x86_64.whl '''