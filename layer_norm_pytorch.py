import torch as th
from torch import nn
import numpy as np
import time
import numpy.testing as npt
from apex.normalization.fused_layer_norm import FusedLayerNorm




device = th.device('cuda:0')
dtype = th.float32
eps = 1E-5
n_repeats = 5


for B in [128*500]:#[128 * 10, 128 * 100, 128 * 500, 128 * 1000]:
    for C in [128]:#[128, 256, 512, 1024, 2048]:
        # WarmUp
        for _ in range(2):
            in_data = th.randn(B, C, device=device, dtype=dtype)
            out_data = in_data * in_data
            npy_out_data = out_data.cpu().numpy()


        # Calculate
        th_ln_fwd_time = 0
        th_ln_bwd_time = 0

        apex_ln_fwd_time = 0
        apex_ln_bwd_time = 0
        for _ in range(n_repeats):
            in_data = th.randn(B, C, device=device, dtype=dtype)
            ograd = th.randn(B, C, device=device, dtype=dtype)
            npy_in_data = in_data.cpu().numpy()
            gt_out = (npy_in_data - npy_in_data.mean(axis=-1, keepdims=True))\
                     / np.sqrt(npy_in_data.var(axis=-1, keepdims=True) + eps)

            th.cuda.synchronize()
            th_ln = nn.LayerNorm(in_data.size()[1:], eps=eps)
            th_ln.cuda(device)
            apex_ln = FusedLayerNorm(in_data.size()[1:], eps=eps)
            apex_ln.cuda(device)
            th.cuda.synchronize()

            # Test for torch LayerNorm
            start = time.time()
            with th.no_grad():
                out_data = th_ln(in_data)
            th.cuda.synchronize()
            th_ln_fwd_time += time.time() - start
            npy_th_ln_out_data = out_data.cpu().numpy()
            npt.assert_allclose(npy_th_ln_out_data, gt_out, 1E-5, 1E-5)

            # Test for apex LayerNorm
            start = time.time()
            with th.no_grad():
                out_data = apex_ln(in_data)
            th.cuda.synchronize()
            apex_ln_fwd_time += time.time() - start
            npy_apex_ln_out_data = out_data.cpu().numpy()
            npt.assert_allclose(npy_apex_ln_out_data, gt_out, 1E-5, 1E-5)

        print('B={}, C={}'.format(B, C))
        print('Torch LayerNorm Time Spent = {} ms'.format(th_ln_fwd_time / n_repeats * 1000))
        print('Torch Apex LayerNorm Time Spent = {} ms'.format(apex_ln_fwd_time / n_repeats * 1000))
