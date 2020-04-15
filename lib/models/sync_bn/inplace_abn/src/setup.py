from setuptools import setup, Extension
from torch.utils import cpp_extension

# setup(name='dp_cpp',
#       ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(
    name='inplace_abn',
    ext_modules=[
        cpp_extension.CUDAExtension('inplace_abn', [
            'inplace_abn.cpp',
            'inplace_abn_cpu.cpp',
            'inplace_abn_cuda.cu',],
            extra_compile_args={'nvcc': ["--expt-extended-lambda"],
                                'cxx': ["-O3"]}
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })
