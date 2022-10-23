from distutils.core import setup, Extension

demo = Extension('demo', sources = ['demo.c'])

setup(name = 'C extension module', version = '1.0', description = 'This is a demo', ext_modules = [demo])