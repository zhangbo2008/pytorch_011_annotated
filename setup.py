#encoding:utf-8
from setuptools import setup, Extension, distutils, Command, find_packages
import setuptools.command.build_ext
import setuptools.command.install
import distutils.command.build
import distutils.command.clean
import platform
import subprocess
import shutil
import sys
import os
# setup.py命令说明.
# TODO: make this more robust
WITH_CUDA = os.path.exists('/Developer/NVIDIA/CUDA-7.5/include') or os.path.exists('/usr/local/cuda/include')
DEBUG = True

class build_deps(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from tools.nnwrap import generate_wrappers as generate_nn_wrappers
        build_all_cmd = ['bash', 'torch/lib/build_all.sh'] #
        if WITH_CUDA:
            build_all_cmd += ['--with-cuda']
        if subprocess.call(build_all_cmd) != 0:# 运行cmd命令.
            sys.exit(1)
        generate_nn_wrappers()


class build_module(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self.run_command('build_py')
        self.run_command('build_ext')

# 打包csrc,让他可以被python调用
class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        # cwrap depends on pyyaml, so we can't import it earlier
        from tools.cwrap import cwrap
        from tools.cwrap.plugins.THPPlugin import THPPlugin
        from tools.cwrap.plugins.THPLongArgsPlugin import THPLongArgsPlugin
        from tools.cwrap.plugins.ArgcountSortPlugin import ArgcountSortPlugin
        from tools.cwrap.plugins.AutoGPU import AutoGPU  # 下行用TensorMethods.cwrap来生成TensorMethods.cpp文件.这个文件可以被python调用. 参考:https://blog.csdn.net/qq_43861133/article/details/121508818
        print(11111111111)
        cwrap('torch/csrc/generic/TensorMethods.cwrap', plugins=[
            THPLongArgsPlugin(), THPPlugin(), ArgcountSortPlugin(), AutoGPU()
        ])
        # It's an old-style class in Python 2.7...
        print(222222222222222222222)
        setuptools.command.build_ext.build_ext.run(self)
        print(3333333333333)



class build(distutils.command.build.build):
    sub_commands = [
        ('build_deps', lambda self: True),
    ] + distutils.command.build.build.sub_commands


class install(setuptools.command.install.install):
    def run(self):
        if not self.skip_build:
            self.run_command('build_deps')
        setuptools.command.install.install.run(self)


class clean(distutils.command.clean.clean):
    def run(self):
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            for glob in filter(bool, ignores.split('\n')):
                shutil.rmtree(glob, ignore_errors=True)
        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)



################################################################################
# Configure compile flags
################################################################################

include_dirs = []
extra_link_args = []
extra_compile_args = ['-std=c++11']

cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, "torch", "lib")

tmp_install_path = lib_path + "/tmp_install"
include_dirs += [
    cwd,
    os.path.join(cwd, "torch", "csrc"),
    tmp_install_path + "/include",
    tmp_install_path + "/include/TH",
]

extra_link_args.append('-L' + lib_path)

main_libraries = ['TH']
main_sources = [
    "torch/csrc/Module.cpp",
    "torch/csrc/Generator.cpp",
    "torch/csrc/Tensor.cpp",
    "torch/csrc/Storage.cpp",
    "torch/csrc/utils.cpp",
    "torch/csrc/serialization.cpp",
]  #这些文件是核心代码.会编译成上面th文件.

try:
    import numpy as np
    include_dirs += [np.get_include()]
    main_sources += ["torch/csrc/numpy.cpp"]
    extra_compile_args += ['-DWITH_NUMPY']
except ImportError:
    pass


if DEBUG:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g']


def make_relative_rpath(path):
    if platform.system() == 'Darwin':
        return '-Wl,-rpath,@loader_path/' + path
    else:
        return '-Wl,-rpath,$ORIGIN/' + path

################################################################################
# Declare extensions and package
################################################################################

extensions = []
packages = find_packages(exclude=('tools.*', 'torch.cuda', 'torch.legacy.cunn'))

C = Extension("torch._C",
    libraries=main_libraries,   # 加入库包TH.so
    sources=main_sources,
    language='c++',
    extra_compile_args=extra_compile_args,
    include_dirs=include_dirs, # 添加头文件.
    extra_link_args=extra_link_args + [make_relative_rpath('lib')]
)  # 这部分的源码在/usr/lib/python2.7/distutils/extension.py link against是链接的意思.
extensions.append(C)
#加入神经网络到库包.
THNN = Extension("torch._thnn._THNN",
    libraries=['TH', 'THNN'],
    sources=['torch/csrc/nn/THNN.cpp'],
    language='c++',
    extra_compile_args=extra_compile_args,
    include_dirs=include_dirs,
    extra_link_args=extra_link_args + [make_relative_rpath('../lib')]
)
extensions.append(THNN)
#=============最后把extensions都放到torch里面.
setup(name="torch", version="0.1",
    ext_modules=extensions,
    cmdclass = {
        'build': build,
        'build_ext': build_ext,
        'build_deps': build_deps,
        'build_module': build_module,
        'install': install,
        'clean': clean,
    },
    packages=packages,
    package_data={'torch': ['lib/*.so*', 'lib/*.h']}, # 打包其他文件.
    install_requires=['pyyaml'],
)
#-package_dir 告诉setuptools哪些目录下的文件被映射到哪个源码包。一个例子：package_dir = {'': 'lib'}，表示“root package”中的模块都在lib 目录中。

# python调用c方法:https://blog.csdn.net/qq_43861133/article/details/121508818 里面第四个方法.