import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp') + glob.glob('*.cu')

setup(
    name='tri_interpolate',
    ext_modules=[
        CUDAExtension(
            name = 'tri_interpolate',
            sources = sources,
            include_dirs = include_dirs,
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })