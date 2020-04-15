from setuptools import setup, Extension
from torch.utils import cpp_extension

# setup(name='dp_cpp',
#       ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(
    name='dp_cuda',
    ext_modules=[
        cpp_extension.CUDAExtension('dp_cuda', [
            'dp_cuda.cpp',
            'dp_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })
setup(
    name='dp_ne_cuda',
    ext_modules=[
        cpp_extension.CUDAExtension('dp_ne_cuda', [
            'dp_ne_cuda.cpp',
            'dp_ne_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })
# setup(
#     name='dp_ne_rank1_cuda',
#     ext_modules=[
#         cpp_extension.CUDAExtension('dp_ne_rank1_cuda', [
#             'dp_ne_rank1_cuda.cpp',
#             'dp_ne_rank1_cuda_kernel.cu',
#         ])
#     ],
#     cmdclass={
#         'build_ext': cpp_extension.BuildExtension
#     })
