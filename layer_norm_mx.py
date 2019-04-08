import mxnet as mx
import numpy as np


ctx = mx.gpu(0)
dtype = np.float32
eps = 1E-5
n_repeats = 5


