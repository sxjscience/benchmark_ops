import mxnet as mx
from mxnet.gluon import nn
import numpy as np
import numpy.testing as npt
import time
import pandas as pd
import argparse
import io
import re



np.random.seed(123)
mx.random.seed(123)


def parse_ctx(ctx_args):
    ctx = re.findall('([a-z]+)(\d*)', ctx_args)
    ctx = [(device, int(num)) if len(num) > 0 else (device, 0) for device, num in ctx]
    ctx = [mx.Context(*ele) for ele in ctx]
    return ctx


parser = argparse.ArgumentParser(description='Profile LayerNorm using MXNet.')
parser.add_argument('--use_gpu', default=1, type=int, help='Whether to use gpu')
parser.add_argument('--nbatch', default=128 * 32, type=int, help='The number of batches for testing')
parser.add_argument('--nchannel', default=1024, type=int, help='The number of channels for testing')
parser.add_argument('--eps', default=1E-5, type=float, help='The eps of layer normalization')
parser.add_argument('--nrepeat', default=5, type=int, help='Number to repeat the ')
parser.add_argument('--dtype', default='float32', type=str, help='The data type to use')
parser.add_argument('--profile', action='store_true', help='If set, profile the code use the build-in profiler of mxnet')



args = parser.parse_args()
if args.dtype == 'float32':
    dtype = np.float32
elif args.dtype == 'float64':
    dtype = np.float64
elif args.dtype == 'float16':
    dtype = np.float16
else:
    raise NotImplementedError

if args.use_gpu:
    ctx = mx.gpu(0)
else:
    ctx = mx.cpu()


if args.profile:
    import os
    from mxnet import profiler
    # import cProfile
    # import pstats
    # def f8(x):
    #     ret = "%8.3f" % x
    #     if ret != '   0.000':
    #         return ret
    #     return "%6dµs" % (x * 1000000)
    #
    #
    # pstats.f8 = f8

    profiler.set_config(profile_all=True, aggregate_stats=True, filename='profile_output.json')
    os.environ['MXNET_EXEC_BULK_EXEC_INFERENCE'] = '0'
    os.environ['MXNET_EXEC_BULK_EXEC_TRAIN'] = '0'
    os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN'] = '0'


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


def check_ln_speed(nbatch, nchannel, eps, nrepeat):
    fwd_check_eps = 1E-1 if dtype == np.float16 else 1E-4
    bwd_check_eps = 1E-1 if dtype == np.float16 else 1E-3
    B, C = nbatch, nchannel
    for _ in range(2):
        in_data = mx.nd.random.normal(shape=(B, C), ctx=ctx, dtype=dtype)
        out_data = in_data * in_data
        npy_out_data = out_data.asnumpy()
    mx.nd.waitall()
    fwd_time = 0
    bwd_time = 0
    if args.profile:
        profiler.set_state('run')
        profiler.pause()
    for i in range(nrepeat + 1):
        in_data = mx.nd.random.normal(shape=(B, C), ctx=ctx, dtype=dtype)
        ograd = mx.nd.random.normal(shape=(B, C), ctx=ctx, dtype=dtype)
        nd_gamma = mx.nd.ones(shape=(C,), ctx=ctx, dtype=dtype)
        nd_beta = mx.nd.zeros(shape=(C,), ctx=ctx, dtype=dtype)
        npy_in_data = in_data.asnumpy().astype(np.float64)
        gt_out = (npy_in_data - npy_in_data.mean(axis=-1, keepdims=True)) \
                 / np.sqrt(npy_in_data.var(axis=-1, keepdims=True) + eps)
        gt_in_data_grad, gt_gamma_grad, gt_beta_grad = \
            npy_ln_grad(npy_in_data, ograd.asnumpy().astype(np.float64), eps, nd_gamma.asnumpy().astype(np.float64))
        mx.nd.waitall()
        in_data.attach_grad()
        nd_gamma.attach_grad()
        nd_beta.attach_grad()
        _no_use = nd_gamma.asnumpy()
        _no_use = nd_beta.asnumpy()
        mx.nd.waitall()
        # Profile Forward + Backward
        with mx.autograd.record():
            mx.nd.waitall()
            if args.profile and i > 0:
                profiler.resume()
            start = time.time()
            out_data, mean_val, std_val = mx.nd.LayerNorm(in_data, gamma=nd_gamma, beta=nd_beta, axis=-1, eps=eps,
                                                          output_mean_var=True)
            out_data.wait_to_read()
            if i > 0:
                fwd_time += time.time() - start
            mx.nd.waitall()
            start = time.time()
            out_data.backward(ograd)
            mx.nd.waitall()
            if args.profile and i > 0:
                profiler.pause()
            if i > 0:
                bwd_time += time.time() - start
        mx_in_data_grad = in_data.grad.asnumpy()
        mx_gamma_grad = nd_gamma.grad.asnumpy()
        mx_beta_grad = nd_beta.grad.asnumpy()
        npt.assert_allclose(mean_val.asnumpy()[:, 0], npy_in_data.mean(axis=-1).astype(dtype), fwd_check_eps, fwd_check_eps)
        npt.assert_allclose(std_val.asnumpy()[:, 0], np.sqrt(npy_in_data.var(axis=-1) + eps).astype(dtype), fwd_check_eps, fwd_check_eps)
        npt.assert_allclose(out_data.asnumpy(), gt_out.astype(dtype), fwd_check_eps, fwd_check_eps)
        for i in range(B):
            npt.assert_allclose(mx_in_data_grad[i, :], gt_in_data_grad[i, :].astype(dtype), fwd_check_eps, fwd_check_eps)
        npt.assert_allclose(mx_gamma_grad, gt_gamma_grad.astype(dtype), bwd_check_eps, bwd_check_eps)
        npt.assert_allclose(mx_beta_grad, gt_beta_grad.astype(dtype), bwd_check_eps, bwd_check_eps)
    if args.profile:
        profiler.set_state('stop')
    return fwd_time / nrepeat * 1000000, bwd_time / nrepeat * 1000000

fwd_time, bwd_time = check_ln_speed(args.nbatch, args.nchannel, args.eps, args.nrepeat)

print('Forward: {}us, Backward: {}us'.format(fwd_time, bwd_time))
if args.profile:
    print(profiler.dumps())
    profiler.dump()
