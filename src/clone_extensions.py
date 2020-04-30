#!/usr/bin/env python
# coding: utf-8

import os

# Set up paths
lib_path = './lib'
modules ={'pybind11': 'https://github.com/pybind/pybind11.git',
         'eigen': 'https://gitlab.com/libeigen/eigen.git'}

# clone paths to lib directory
for k, g in modules.items():
    m_path = os.path.join(lib_path, k)
    
    if not os.path.exists(m_path):
        clone_cmd = ' '.join(['git clone', g, m_path])
        os.system(clone_cmd)
    else:
        os.system(' '.join(['echo',k, 'dependency already cloned in', lib_path]))
