## Install

### Environment setup

```bash
# clone the repository
git clone https://github.com/FrancescoGentile/detrex.git
cd detrex

# create a virtual environment (here with conda but any other tool is fine)
conda create -p .venv python=3.10 # python 3.8 is the minimum required version
conda activate .venv/

# install torch and torchvision (torch>=1.10 is required)
# note: it is mandatory to use the torch and cuda versions
# that were used to compile the detrex package (see below)
pip install torch torchvision # to install the latest version (at the time of writing 2.3.0 with cuda 12.1)

# install gdown to download the wheels and the dataset
pip install gdown
```

### Install the package

```bash
# download the wheels
gdown --folder https://drive.google.com/drive/folders/1gS9lsHIqh6ekExJH5ac4BQD3aBrBUp7Q
```

The structure of wheel folder is the following:

```
whl/
├── pt{torch_version}cu{cuda_version}/
│   ├── detrex-{version}-cp{python_version}-cp{python_version}m-{platform}.whl
```

where:
- `{torch_version}` is the version of torch used to compile the package, for example if the package was compiled with torch 2.3.0, then `{torch_version}` is 23 (only the major and minor version are considered)
- `{cuda_version}` is the version of cuda used to compile the package, for example if the package was compiled with cuda 12.1, then `{cuda_version}` is 121 (only the major and minor version are considered)
- `{version}` is the version of the package
- `{python_version}` is the python version, for example 310
- `{platform}` is the platform, for example `linux_x86_64`

As stated above, you need to install the wheel that was compiled with the same torch and cuda versions that you have installed in your environment.

```bash
# install the package
pip install /path/to/the/wheel.whl

# newer versions of pip may give problems when installing detectron2 (a dependency of detrex)
# since it must be built using legacy build tools. If you encounter this problem, it should be
# solved by installing the `wheel` package and then installing the wheel using no-build-isolation
pip install wheel
pip install /path/to/the/wheel.whl --no-build-isolation
```

At the moment of writing the only available wheel is `detrex-0.5.0-cp310-cp310m-linux_x86_64.whl` compiled with torch 2.3.0 and cuda 12.1.

## Download the dataset

```bash
mkdir datasets
cd datasets 
gdown --folder https://drive.google.com/drive/folders/164XPo2BSmR3ZiyK997rG4uzp1CffOINy
cd vg
unzip images.zip -d .
rm images.zip
cd ../../
```

## Run the code

You can train most of the models using the `train_net.py` script under the `tools` folder. Some models, like DINO, requires a specific training script that can be found under the `projects/model_name` folder. The arguments of the scripts are the following:

```bash
# use --help to see all the arguments of the script
python tools/train_net.py --config-file vg/configs/{config_file}

# if the model requires a specific training script
python projects/{model_name}/train_net.py --config-file vg/configs/{config_file}
```

The provided configuration files are the following (all are DINO models):
- vg/configs/dino_r50_4scale_24ep.py
- vg/configs/dino_swin_large_224_4scale.py
- vg/configs/dino_swin_large_384_5scale.py

All such configuration files initialize the model with the weights of the corresponding model pre-trained on COCO. To download such weights, see under `projects/dino/README.md` for the links. You can change the path to the weights in the configuration file by setting `train.init_checkpoint` to the path of the weights.

## Compiling the package from source

If no wheel is available for your configuration (torch and cuda versions, python version or platform), you can compile the package from source using the following instructions.

First, clone the repository and create a virtual environment with the correct torch and cuda versions.

```bash
# clone the repository
git clone https://github.com/FrancescoGentile/detrex.git
cd detrex

# create a virtual environment using conda
conda create -p .venv python={python_version} # python 3.8 is the minimum required version
conda activate .venv/

# install torch with the version you want to compile the package with
# for versions of torch<2.0.0 see the official pytorch website
# for versions of torch>=2.0.0
pip install torch=={torch_version} --index-url https://download.pytorch.org/whl/cu{cuda_version}

# install the cuda-toolkit of the version matching the torch version
conda install cuda-toolkit -c nvidia/label/cuda-{cuda_version}
# for example, to compile the package with cuda 12.1
conda install cuda-toolkit -c nvidia/label/cuda-12.1.0

# make sure that the gcc version is compatible with the cuda version
# if not, install the correct version of gcc
# at the time of writing, for cuda 12.1 it is sufficient to install cxx-compiler
conda install cxx-compiler

# install the tools needed to build the package
pip install -U setuptools wheel build ninja # ninja is optional but recommended
```

Then set the environment the following environment variables:
- `CUDA_HOME` should point to the cuda installation directory. If you followed the instructions above it should be `/path/to/.venv/`
- `TORCH_CUDA_ARCH_LIST` to the list of architectures you want to compile the package for. The list of the architectures supported by the installed compiler can be found by running `nvcc --list-gpu-arch`. For example, to support all gpu architectures from sm_50 to sm_90, set `TORCH_CUDA_ARCH_LIST` to "5.0 5.2 5.3 6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 6.7 8.9 9.0".
- `FORCE_CUDA` to "1" to force the compilation of the package with cuda support.
- `MAX_JOBS` to the number of jobs you want to use to compile the package if you use ninja (note: each ninja job requires a lot of memory, so be careful with this parameter).

Once the environment is set, you can compile the package with the following command:

```bash
# no-isolation is used here to avoid creating a temporary environment for the build,
# in this way the build process will use the current environment
python -m build --wheel --no-isolation
```

If everything goes well, you will find the wheel in the `dist` folder.
