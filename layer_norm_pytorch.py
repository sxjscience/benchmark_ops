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


eps = 1E-5
n_repeats = 5


candidate_B = [128 * 32]#[128, 128 * 32, 128 * 64, 128 * 128] # The result of apex is wrong when B >= 128 * 512
candidate_C = [256]#[32, 64, 128, 256, 512, 768, 1024]
fwd_only_time_d = {}
fwd_time_d = {}
bwd_time_d = {}

for key in ['th', 'apex']:
    fwd_only_time_d[key] = pd.DataFrame(np.zeros(shape=(len(candidate_B), len(candidate_C)), dtype=np.float64),
                                        index=candidate_B, columns=candidate_C)
    fwd_time_d[key] = pd.DataFrame(np.zeros(shape=(len(candidate_B), len(candidate_C)), dtype=np.float64),
                                   index=candidate_B, columns=candidate_C)
    bwd_time_d[key] = pd.DataFrame(np.zeros(shape=(len(candidate_B), len(candidate_C)), dtype=np.float64),
                                   index=candidate_B, columns=candidate_C)
for B in candidate_B:
    for C in candidate_C:
        for key in ['th', 'apex']:
            # WarmUp
            for _ in range(2):
                in_data = th.randn(B, C, device=device, dtype=dtype)
                out_data = in_data * in_data
                npy_out_data = out_data.cpu().numpy()
            if key == 'th':
                layer = nn.LayerNorm(in_data.size()[1:], eps=eps)
            elif key == 'apex':
                layer = FusedLayerNorm(in_data.size()[1:], eps=eps)
            else:
                raise NotImplementedError
            layer.cuda(device)
            if dtype == th.float16:
                layer.half()
            th.cuda.synchronize()
            fwd_only_time = 0
            fwd_time = 0
            bwd_time = 0
            for _ in range(n_repeats):
                in_data = th.randn(B, C, device=device, dtype=dtype, requires_grad=True)
                ograd = th.randn(B, C, device=device, dtype=dtype)
                npy_in_data = in_data.cpu().detach().numpy()
                gt_out = (npy_in_data - npy_in_data.mean(axis=-1, keepdims=True))\
                          / np.sqrt(npy_in_data.var(axis=-1, keepdims=True) + eps)
                th.cuda.synchronize()

                # Profile Forward-only
                start = time.time()
                with th.no_grad():
                    out_data = layer(in_data)
                th.cuda.synchronize()
                fwd_only_time += time.time() - start
                npy_th_out_data = out_data.cpu().numpy()
                npt.assert_allclose(npy_th_out_data, gt_out, 1E-5, 1E-5)

                # Profile Forward + Backward
                with th.enable_grad():
                    start = time.time()
                    out_data = layer(in_data)
                    th.cuda.synchronize()
                    fwd_time += time.time() - start
                    start = time.time()
                    out_data.backward([ograd])
                    th.cuda.synchronize()
                    bwd_time += time.time() - start
            fwd_only_time_d[key].at[B, C] = fwd_only_time / n_repeats * 1000000
            fwd_time_d[key].at[B, C] = fwd_time / n_repeats * 1000000
            bwd_time_d[key].at[B, C] = bwd_time / n_repeats * 1000000
            print('B={}, C={}'.format(B, C))
            print('LayeNorm = {}'.format(key))
            print('   fwd-only = {} ms, fwd = {} ms, bwd = {} ms'.format(fwd_only_time / n_repeats * 1000000,
                                                                         fwd_time / n_repeats * 1000000,
                                                                         bwd_time / n_repeats * 1000000))
print('PyTorch LayerNorm Forward ')
print(fwd_time_d['th'])

print('PyTorch LayerNorm Backward ')
print(bwd_time_d['th'])

print('Apex LayerNorm Forward')
print(fwd_time_d['apex'])

print('Apex LayerNorm Backward')
print(bwd_time_d['apex'])
