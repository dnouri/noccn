import sys


def add_path():
    sys.path.append('.')

add_path()


import convnet
import data
import gpumodel
import options
import shownet
