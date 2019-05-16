import subprocess
from nvprof_parser import parse_nvprof_out
import pandas as pd
import re

LARGE_BERT_SHAPES = [(128 * 32, 768), (128 * 32, 1024), (128 * 64, 768), (128 * 64, 1024)]
TEST_BATCH_L = [128, 128 * 20, 128 * 32, 128 * 64, 128 * 128]
TEST_CHANNEL_L = [32, 64, 128, 256, 512, 768, 1024]
NVPROF_EXE = 'nvprof'
PYTHON_EXE = 'python3'
N_REPEAT = 3
EPS = 1E-5
CTX = 'gpu0'
DTYPE = 'float32'
TIME_R = r'\d+\.?\d*'
LN_OUT_REG = r'Forward: ({})us, Backward: ({})us'.format(TIME_R, TIME_R)
MX_FWD_KEYWORD = 'LayerNormFusedForwardKernel'
MX_BWD_DATA_KEYWORD = 'LayerNormFusedBackwardKernel_Data'
MX_BWD_GAMMA_BETA_KEYWORD = ['LayerNormFusedBackwardKernel_PartGammaBeta', 'LayerNormFusedBackwardKernel_GammaBeta']


def test_speed(codebase, test_batch_l, test_channel_l, eps, ctx, dtype, fwd_keyword,
               bwd_data_keyword, bwd_gamma_beta_keyword, profile_nv):
    for nbatch in test_batch_l:
        for nchannel in test_channel_l:
            if codebase == 'mxnet':
                ret = subprocess.run([NVPROF_EXE, PYTHON_EXE, 'layer_norm_mx.py', '--ctx', str(ctx), '--nbatch', str(nchannel),
                                      '--eps', str(eps), '--dtype', dtype, '--nrepeat', str(N_REPEAT)],
                                     stderr=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
            elif codebase == 'pytorch':
                ret = subprocess.run(
                    [NVPROF_EXE, PYTHON_EXE, 'layer_norm_pytorch.py', '--ctx', str(ctx), '--nbatch', str(nchannel),
                     '--eps', str(eps), '--dtype', dtype, '--nrepeat', str(N_REPEAT), '--apex'],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE)
            else:
                raise NotImplementedError
            runfile_out = ret.stdout.decode('utf-8')
            fwd_time, bwd_time = re.match(LN_OUT_REG, runfile_out).groups()
            fwd_time = float(fwd_time)
            bwd_time = float(bwd_time)
            print(nbatch, nchannel, fwd_time, bwd_time)
            if profile_nv:
                nvprof_result = parse_nvprof_out(ret.stderr.decode('utf-8'))
                _, fwd_runtime, _, _, _ = nvprof_result.fetch_run_time(keyword=fwd_keyword, unit='us')
                fwd_runtime = sum(fwd_runtime)
                _, bwd_data_runtime, _, _, _ = nvprof_result.fetch_run_time(keyword=bwd_data_keyword, unit='us')
                bwd_data_runtime = sum(bwd_data_runtime)
                _, bwd_gamma_beta_runtime, _, _, _ = nvprof_result.fetch_run_time(keyword=bwd_gamma_beta_keyword, unit='us')
                bwd_gamma_beta_runtime = sum(bwd_gamma_beta_runtime)
                print(fwd_runtime, bwd_data_runtime, bwd_gamma_beta_runtime)

test_speed('pytorch', TEST_BATCH_L, TEST_CHANNEL_L, EPS, CTX, DTYPE, MX_FWD_KEYWORD, MX_BWD_DATA_KEYWORD, MX_BWD_GAMMA_BETA_KEYWORD, profile_nv=True)
test_speed('mxnet', TEST_BATCH_L, TEST_CHANNEL_L, EPS, CTX, DTYPE, MX_FWD_KEYWORD, MX_BWD_DATA_KEYWORD, MX_BWD_GAMMA_BETA_KEYWORD, profile_nv=True)
