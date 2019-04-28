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

candidate_B = [128 * 32]#[128, 128 * 32, 128 * 64, 128 * 128]
candidate_C = [256] #[32, 64, 128, 256, 512, 768, 1024]
fwd_time_d = {}
bwd_time_d = {}



def npy_ln_grad(in_data, ograd, eps, gamma):
    data_mean = in_data.mean(axis=-1, keepdims=True)
    data_std = np.sqrt(in_data.var(axis=-1, keepdims=True) + eps)
    centered_data = (in_data - data_mean) / data_std
    gamma_grad = (centered_data * ograd).sum(axis=0)
    beta_grad = ograd.sum(axis=0)
    w = ograd * gamma / data_std
    in_data_grad = w - w.mean(axis=-1, keepdims=True) - centered_data * (w * centered_data).mean(axis=-1, keepdims=True)
    return in_data_grad, gamma_grad, beta_grad


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
            ln_layer.hybridize()
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
                gt_in_data_grad, gt_gamma_grad, gt_beta_grad =\
                    npy_ln_grad(in_data.asnumpy(), ograd.asnumpy(), eps, ln_layer.params.get('gamma').data().asnumpy())
                mx.nd.waitall()
                # Profile Forward + Backward
                with mx.autograd.record():
                    mx.nd.waitall()
                    start = time.time()
                    out_data = ln_layer(in_data)
                    out_data.wait_to_read()
                    fwd_time += time.time() - start
                    mx.nd.waitall()
                    start = time.time()
                    out_data.backward(ograd)
                    mx.nd.waitall()
                    bwd_time += time.time() - start
                npt.assert_allclose(out_data.asnumpy(), gt_out, 1E-5, 1E-5)
                npt.assert_allclose(in_data.grad.asnumpy(), gt_in_data_grad, 1E-5, 1E-5)
                npt.assert_allclose(ln_layer.params.get('gamma').data().grad.asnumpy(), gt_gamma_grad, 1E-5, 1E-5)
                npt.assert_allclose(ln_layer.params.get('beta').data().grad.asnumpy(), gt_beta_grad, 1E-5, 1E-5)
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