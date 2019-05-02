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


def nd_layer_norm(data, gamma, beta, axis, eps):
    nd_mean = mx.nd.mean(data, axis=-1, keepdims=True)
    nd_var = mx.nd.mean(mx.nd.square(mx.nd.broadcast_minus(data, nd_mean)), axis=axis, keepdims=True)
    nd_std = mx.nd.sqrt(nd_var + eps)
    centered_data = mx.nd.broadcast_div(mx.nd.broadcast_minus(data, nd_mean), nd_std)
    out = mx.nd.broadcast_add(mx.nd.broadcast_mul(centered_data, gamma), beta)
    return out, nd_mean, nd_std

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
            # if key == 'ln':
            #     ln_layer = nn.LayerNorm(epsilon=eps)
            # else:
            #     raise NotImplementedError
            # ln_layer.hybridize()
            # ln_layer.initialize(ctx=ctx)
            for _ in range(2):
                in_data = mx.nd.random.normal(shape=(B, C), ctx=ctx, dtype=dtype)
                nd_gamma = mx.nd.ones(shape=(C,), ctx=ctx, dtype=dtype)
                nd_beta = mx.nd.zeros(shape=(C,), ctx=ctx, dtype=dtype)
                out_data = in_data * in_data
                #out_data = ln_layer(out_data)
                out_data = mx.nd.LayerNorm(out_data, gamma=nd_gamma, beta=nd_beta, axis=-1, eps=eps)
                npy_out_data = out_data.asnumpy()
            mx.nd.waitall()
            fwd_time = 0
            bwd_time = 0
            for _ in range(n_repeats):
                in_data = mx.nd.random.normal(shape=(B, C), ctx=ctx, dtype=dtype)
                ograd = mx.nd.random.normal(shape=(B, C), ctx=ctx, dtype=dtype)
                nd_gamma = mx.nd.ones(shape=(C,), ctx=ctx, dtype=dtype)
                nd_beta = mx.nd.zeros(shape=(C,), ctx=ctx, dtype=dtype)
                npy_in_data = in_data.asnumpy()
                gt_out = (npy_in_data - npy_in_data.mean(axis=-1, keepdims=True)) \
                         / np.sqrt(npy_in_data.var(axis=-1, keepdims=True) + eps)
                # gt_in_data_grad, gt_gamma_grad, gt_beta_grad =\
                #     npy_ln_grad(in_data.asnumpy(), ograd.asnumpy(), eps, ln_layer.params.get('gamma').data().asnumpy())
                gt_in_data_grad, gt_gamma_grad, gt_beta_grad = \
                    npy_ln_grad(in_data.asnumpy(), ograd.asnumpy(), eps, nd_gamma.asnumpy())
                mx.nd.waitall()
                in_data.attach_grad()
                nd_gamma.attach_grad()
                nd_beta.attach_grad()
                mx.nd.waitall()
                # Profile Forward + Backward
                with mx.autograd.record():
                    mx.nd.waitall()
                    start = time.time()
                    # out_data = ln_layer(in_data)
                    out_data, mean_val, std_val = mx.nd.LayerNorm(in_data, gamma=nd_gamma, beta=nd_beta, axis=-1, eps=eps, output_mean_var=True)
                    #out_data, mean_val, std_val = nd_layer_norm(in_data, gamma=nd_gamma, beta=nd_beta, axis=-1, eps=eps)
                    out_data.wait_to_read()
                    fwd_time += time.time() - start
                    mx.nd.waitall()
                    start = time.time()
                    out_data.backward(ograd)
                    mx.nd.waitall()
                    bwd_time += time.time() - start
                # Debug
                npy_gamma = nd_gamma.asnumpy()
                npy_beta = nd_beta.asnumpy()
                npy_mean = npy_in_data.mean(axis=-1, keepdims=True)
                npy_std = np.sqrt(npy_in_data.var(axis=-1, keepdims=True) + eps)
                npy_ograd = ograd.asnumpy()

                mx_in_data_grad = in_data.grad.asnumpy()
                mx_gamma_grad = nd_gamma.grad.asnumpy()
                mx_beta_grad = nd_beta.grad.asnumpy()
                npt.assert_allclose(mean_val.asnumpy()[:, 0], npy_in_data.mean(axis=-1), 1E-5, 1E-5)
                npt.assert_allclose(std_val.asnumpy()[:, 0], np.sqrt(npy_in_data.var(axis=-1) + eps), 1E-5, 1E-5)
                npt.assert_allclose(out_data.asnumpy(), gt_out, 1E-5, 1E-5)
                for i in range(B):
                    npt.assert_allclose(mx_in_data_grad[i, :], gt_in_data_grad[i, :], 1E-5, 1E-5)
                # npt.assert_allclose(ln_layer.params.get('gamma').data().grad.asnumpy(), gt_gamma_grad, 1E-5, 1E-5)
                # npt.assert_allclose(ln_layer.params.get('beta').data().grad.asnumpy(), gt_beta_grad, 1E-5, 1E-5)
                npt.assert_allclose(nd_gamma.grad.asnumpy(), gt_gamma_grad, 1E-4, 1E-4)
                npt.assert_allclose(nd_beta.grad.asnumpy(), gt_beta_grad, 1E-3, 1E-3)
            fwd_time_d[key].at[B, C] = fwd_time / n_repeats * 1000000
            bwd_time_d[key].at[B, C] = bwd_time / n_repeats * 1000000
            print('B={}, C={}'.format(B, C))
            print('LayeNorm = {}'.format(key))
            print('   fwd = {} us, bwd = {} us'.format(fwd_time / n_repeats * 1000000,
                                                       bwd_time / n_repeats * 1000000))

print('MXNet LayerNorm Forward:')
print(fwd_time_d['ln'])

print('MXNet LayerNorm Backward:')
print(bwd_time_d['ln'])