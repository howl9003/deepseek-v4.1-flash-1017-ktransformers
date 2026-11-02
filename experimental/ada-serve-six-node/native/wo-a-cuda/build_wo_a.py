"""Build one wo_a_parts variant, CPU only, with the recipe of the pinned native kernels (phase-g-g0-native worker) (Claude, 1 Nov 2026).
argv: extension name, BPC, STAGES. Prints the binary and its sha."""
import hashlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
name, bpc, stages = sys.argv[1], sys.argv[2], sys.argv[3]
nvcc = Path('/usr/local/cuda-13.3/bin/nvcc')
env = dict(os.environ, CUDA_HOME=str(nvcc.parent.parent), PATH=str(Path(sys.prefix) / 'bin') + ':' + str(nvcc.parent) + ':' + os.environ['PATH'])
torch_root = Path(torch.__file__).parent
includes = [nvcc.parent.parent / 'include', torch_root / 'include', torch_root / 'include/torch/csrc/api/include', Path(sysconfig.get_paths()['include']), torch_root.parent / 'nvidia/cu13/include']
folder = HERE / 'build'
folder.mkdir(exist_ok=True)
obj, so = folder / (name + '.o'), folder / (name + '.so')
subprocess.run([nvcc, '-c', HERE / 'wo_a_parts.cu', '-o', obj, '-O3', '-std=c++20', '-gencode=arch=compute_89,code=sm_89', '-fmad=false', '--expt-relaxed-constexpr',
                '--compiler-options=-fPIC', '-DTORCH_EXTENSION_NAME=' + name, '-DTORCH_API_INCLUDE_EXTENSION_H', '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
                '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_BFLOAT16_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__',
                '-DWO_A_BPC=' + bpc, '-DWO_A_STAGES=' + stages] + ['-I' + str(p) for p in includes], check=True, env=env)
subprocess.run(['g++', '-shared', obj, '-L' + str(torch_root / 'lib'), '-lc10', '-lc10_cuda', '-ltorch_cpu', '-ltorch_cuda', '-ltorch_python', '-ltorch',
                '-L' + str(nvcc.parent.parent / 'lib64'), '-lcudart', '-Wl,-rpath,' + str(torch_root / 'lib'), '-o', so], check=True, env=env)
print(so, hashlib.sha256(so.read_bytes()).hexdigest())
