from distutils import sysconfig
from setuptools import setup, Extension
import os

cpp_path = './src/lib'


cpp_args = ['-std=c++11'] 

ext_modules = [
    Extension(
    'hdp_funcs',
        [os.path.join(cpp_path,'hdp_funcs.cpp')],
        include_dirs=[os.path.join(cpp_path,'pybind11/include'),
                      os.path.join(cpp_path,'eigen')],
    language='c++',
    extra_compile_args = cpp_args,
    ),
    Extension(
    'hdp_preproc',
        [os.path.join(cpp_path,'hdp_preproc.cpp')],
        include_dirs=[os.path.join(cpp_path,'pybind11/include'),
                      os.path.join(cpp_path,'eigen')],
    language='c++',
    extra_compile_args = cpp_args,
    ),
]

setup(name='hdp',
      version='0.1.0',
      description='Implementation of Hierarchical LDA',
      url='http://github.com/datadiarist/hplda',
      author='Andrew and Eduardo',
      author_email='andrew.carr@duke.edu',
      license='MIT',
      classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Text Processing :: Linguistic',
          'Operating System :: OS Independent'
      ],
      keywords='topic non-parametric modeling nlp',
      python_requires = '>=3.0',
      package_dir = {'': 'src'},
      packages=['hdp'],
      zip_safe=False,
      install_requires=[
          'pybind11', 
          'pandas', 
          'scipy', 
          'numpy',
          'nltk'],
      ext_modules=ext_modules,
      test_suite='nose.collector',
      tests_require=['nose']
      )
