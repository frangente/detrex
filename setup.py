# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Modified from
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/setup.py
# https://github.com/facebookresearch/detectron2/blob/main/setup.py
# https://github.com/open-mmlab/mmdetection/blob/master/setup.py
# https://github.com/Oneflow-Inc/libai/blob/main/setup.py
# ------------------------------------------------------------------------------------------------

import os
import shutil
from pathlib import Path
from typing import List

import torch
from setuptools import Extension, setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_extensions() -> List[Extension]:
    root_dir = Path(__file__).parent
    extensions_dir = root_dir / "detrex" / "layers" / "csrc"

    sources = list(extensions_dir.rglob("*.cpp"))
    source_cuda = list(extensions_dir.rglob("*.cu"))

    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    if use_cuda or force_cuda:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []
        return None

    # make sources relative to the root directory
    sources = [str(s.relative_to(root_dir)) for s in sources]
    include_dirs = [str(extensions_dir.relative_to(root_dir))]

    ext_modules = [
        extension(
            "detrex._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def get_detrex_configs() -> List[str]:
    """Returns a list of configs to include in package for model zoo."""
    root_dir = Path(__file__).parent
    source_configs_dir = root_dir / "configs"
    destination = root_dir / "detrex" / "config" / "configs"

    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if source_configs_dir.exists():
        if destination.is_symlink():
            destination.unlink()
        elif destination.is_dir():
            shutil.rmtree(destination)

    if not destination.exists():
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = list(source_configs_dir.rglob("*.py"))
    return [str(p) for p in config_paths]


def write_version() -> None:
    # read version from pyproject.toml
    pyproject_toml = Path(__file__).parent / "pyproject.toml"
    with open(pyproject_toml, "r") as f:
        lines = f.readlines()
    version = None
    for line in lines:
        if "version" in line:
            version = line.split("=")[1].strip().replace('"', "")
            break
    if version is None:
        msg = "Version not found in pyproject.toml"
        raise RuntimeError(msg)

    # get torch version
    torch_major, torch_minor = torch.__version__.split(".")[:2]
    torch_version = f"pt{torch_major}{torch_minor}"

    # get cuda version
    cuda_version = torch.version.cuda
    if cuda_version is None:
        cuda_version = "cpu"
    else:
        cuda_version = f"cu{cuda_version.replace('.', '')}"

    version = f"{version}+{torch_version}{cuda_version}"
    print(version)

    # write version to _version.py
    version_file = Path(__file__).parent / "detrex" / "_version.py"
    with open(version_file, "w") as f:
        f.write(f'__version__ = "{version}"')


if __name__ == "__main__":
    write_version()

    setup(
        ext_modules=get_extensions(),
        cmdclass={"build_ext": cpp_extension.BuildExtension},
        package_data={"detrex.config": get_detrex_configs()},
    )
