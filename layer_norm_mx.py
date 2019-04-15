import mxnet as mx
from mxnet.gluon import nn
import numpy as np
import numpy.testing as npt
import time
import pandas as pd

ctx = mx.gpu(0)
dtype = np.float32
eps = 1E-5
n_repeats = 5

candidate_B = [128, 128 * 32, 128 * 64, 128 * 128] # The result of apex will be wrong when B >= 128 * 512
candidate_C = [128, 256, 512, 768, 1024]
fwd_time_d = {}
bwd_time_d = {}


for key in ['ln']:
    fwd_time_d[key] = pd.DataFrame(np.zeros(shape=(len(candidate_B), len(candidate_C)), dtype=np.float64),
                                   index=candidate_B, columns=candidate_C)
    bwd_time_d[key] = pd.DataFrame(np.zeros(shape=(len(candidate_B), len(candidate_C)), dtype=np.float64),
                                   index=candidate_B, columns=candidate_C)

for B in candidate_B:
    for C in candidate_C:
        # WarmUp
        for key in ['ln']:
            if key == 'ln':
                ln_layer = nn.LayerNorm(epsilon=eps)
            else:
                raise NotImplementedError
            ln_layer.initialize(ctx=ctx)
            for _ in range(2):
                in_data = mx.nd.random.normal(shape=(B, C), ctx=ctx, dtype=dtype)
                out_data = in_data * in_data
                out_data = ln_layer(out_data)
                npy_out_data = out_data.asnumpy()
            mx.nd.waitall()
            fwd_time = 0
            bwd_time = 0
            for _ in range(n_repeats):
                in_data = mx.nd.random.normal(shape=(B, C), ctx=ctx, dtype=dtype)
                ograd = mx.nd.random.normal(shape=(B, C), ctx=ctx, dtype=dtype)
                npy_in_data = in_data.asnumpy()
                gt_out = (npy_in_data - npy_in_data.mean(axis=-1, keepdims=True)) \
                         / np.sqrt(npy_in_data.var(axis=-1, keepdims=True) + eps)
                mx.nd.waitall()
                # Profile Forward + Backward
                with mx.autograd.record():
                    start = time.time()
                    out_data = ln_layer(in_data)
                    mx.nd.waitall()
                    fwd_time += time.time() - start
                    start = time.time()
                    out_data.backward(ograd)
                    mx.nd.waitall()
                    bwd_time += time.time() - start
                npt.assert_allclose(out_data.asnumpy(), gt_out, 1E-5, 1E-5)
            fwd_time_d[key].at[B, C] = fwd_time / n_repeats * 1000000
            bwd_time_d[key].at[B, C] = bwd_time / n_repeats * 1000000
            print('B={}, C={}'.format(B, C))
            print('LayeNorm = {}'.format(key))
            print('   fwd = {} ms, bwd = {} ms'.format(fwd_time / n_repeats * 1000000,
                                                       bwd_time / n_repeats * 1000000))

print('MXNet LayerNorm Forward:')
print(fwd_time_d['ln'])

print('MXNet LayerNorm Backward:')
print(bwd_time_d['ln'])