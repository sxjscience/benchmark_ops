import torch as th
from torch import nn
import numpy as np
import time
import numpy.testing as npt
import pandas as pd
import argparse
from apex.normalization.fused_layer_norm import FusedLayerNorm



def npy_ln_grad(in_data, ograd, eps, gamma):
    data_mean = in_data.mean(axis=-1, keepdims=True)
    data_std = np.sqrt(in_data.var(axis=-1, keepdims=True) + eps)
    centered_data = (in_data - data_mean) / data_std
    gamma_grad = (centered_data * ograd).sum(axis=0)
    beta_grad = ograd.sum(axis=0)
    w = ograd * gamma / data_std
    in_data_grad = w - w.mean(axis=-1, keepdims=True) - centered_data * (w * centered_data).mean(axis=-1, keepdims=True)
    return in_data_grad, gamma_grad, beta_grad



th.manual_seed(123)
np.random.seed(123)


parser = argparse.ArgumentParser(description='Profile LayerNorm using MXNet.')
parser.add_argument('--use_gpu', default=1, type=int, help='Whether to use gpu')
parser.add_argument('--nbatch', default=128 * 32, type=int, help='The number of batches for testing')
parser.add_argument('--nchannel', default=1024, type=int, help='The number of channels for testing')
parser.add_argument('--eps', default=1E-5, type=float, help='The eps of layer normalization')
parser.add_argument('--nrepeat', default=5, type=int, help='Number to repeat the ')
parser.add_argument('--dtype', default='float32', type=str, help='The data type to use')
parser.add_argument('--apex', action='store_true', help='If set, profile the code use the build-in profiler of mxnet')
parser.add_argument('--profile', action='store_true', help='If set, profile the code use the build-in profiler of mxnet')
args = parser.parse_args()


if args.use_gpu:
    device = th.device('cuda:0')
else:
    device = th.device('cpu')
if args.dtype == 'float32':
    dtype = th.float32
elif args.dtype == 'float64':
    dtype = th.float64
elif args.dtype == 'float16':
    dtype = th.float16
else:
    raise NotImplementedError

def check_ln_speed(use_apex, nbatch, nchannel, eps, nrepeat):
    B, C = nbatch, nchannel
    # WarmUp
    for _ in range(2):
        in_data = th.randn(B, C, device=device, dtype=dtype)
        out_data = in_data * in_data
        npy_out_data = out_data.cpu().numpy()
    if not use_apex:
        layer = nn.LayerNorm(in_data.size()[1:], eps=eps)
    else:
        layer = FusedLayerNorm(in_data.size()[1:], eps=eps)
    if args.use_gpu:
        layer.cuda(device)
    if dtype == th.float16:
        layer.half()
    th.cuda.synchronize()
    fwd_time = 0
    bwd_time = 0
    for i in range(nrepeat):
        in_data = th.randn(B, C, device=device, dtype=dtype, requires_grad=True)
        ograd = th.randn(B, C, device=device, dtype=dtype)
        npy_in_data = in_data.cpu().detach().numpy()
        gt_out = (npy_in_data - npy_in_data.mean(axis=-1, keepdims=True)) \
                 / np.sqrt(npy_in_data.var(axis=-1, keepdims=True) + eps)
        th.cuda.synchronize()

        # Profile Forward + Backward
        with th.enable_grad():
            th.cuda.synchronize()
            start = time.time()
            out_data = layer(in_data)
            th.cuda.synchronize()
            fwd_time += time.time() - start
            start = time.time()
            out_data.backward([ograd])
            th.cuda.synchronize()
            bwd_time += time.time() - start
        npy_th_out_data = out_data.cpu().detach().numpy()
        npt.assert_allclose(npy_th_out_data, gt_out, 1E-5, 1E-5)

    return fwd_time / nrepeat * 1000000, bwd_time / nrepeat * 1000000
fwd_time, bwd_time = check_ln_speed(args.apex, args.nbatch, args.nchannel, args.eps, args.nrepeat)

print('Forward: {}us, Backward: {}us'.format(fwd_time, bwd_time))
