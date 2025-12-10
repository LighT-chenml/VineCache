from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
# extensions = [
#     Extension('EvLFU', ['EvLFU.pyx', 'evlfu_v2.cpp'],
#               extra_compile_args=['-std=c++11'],
#               language='c++'
#               ),
# ]

# setup(
#     ext_modules=cythonize(extensions),
#     extra_compile_args=["-w", '-g'],
#     extra_compile_args=["-O3"],
# )

extensions = [Extension("VineCache_Cython", ["VineCache_Cython.pyx", "src_VineCache_Cython.cpp"], language='c++',)]
ext_modules = cythonize(extensions, language_level=3)

setup(cmdclass = {'build_ext': build_ext}, ext_modules = ext_modules, extra_compile_args=["-w", "-g", "-O3"])