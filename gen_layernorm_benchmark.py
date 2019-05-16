import subprocess
from nvprof_parser import parse_nvprof_out
import pandas as pd
import re

LARGE_BERT_SHAPES = [(128 * 32, 768), (128 * 32, 1024), (128 * 64, 768), (128 * 64, 1024)]
TEST_BATCH_L = [128, 128 * 20, 128 * 32, 128 * 64, 128 * 128]
TEST_CHANNEL_L = [32, 64, 128, 256, 512, 768, 1024]
NVPROF_EXE = 'nvprof'
PYTHON_EXE = 'python3'
N_REPEAT = 5
EPS = 1E-5
CTX = 'gpu0'
DTYPE = 'float32'
TIME_R = r'\d+\.?\d*'
LN_OUT_REG = r'Forward: {}us, Backward: {}us'.format(TIME_R, TIME_R)

def test_speed(runfile, test_batch_l, test_channel_l, eps, ctx, dtype):
    for nbatch in test_batch_l:
        for nchannel in test_channel_l:
            ret = subprocess.run([NVPROF_EXE, PYTHON_EXE, runfile, '--ctx', str(ctx), '--nbatch', str(nchannel),
                                  '--eps', str(eps), '--dtype', dtype, '--nrepeat', str(N_REPEAT)],
                                 stderr=subprocess.PIPE,
                                 stdout=subprocess.PIPE)
            runfile_out = ret.stdout.decode('utf-8')
            fwd_time, bwd_time = re.match(LN_OUT_REG, runfile_out).groups()
            fwd_time = float(fwd_time)
            bwd_time = float(bwd_time)
            print(nbatch, nchannel, fwd_time, bwd_time)
            nvprof_result = parse_nvprof_out(ret.stderr.decode('utf-8'))
            print(nvprof_result.command, nvprof_result.profile_result)

test_speed('layer_norm_mx.py', TEST_BATCH_L, TEST_CHANNEL_L, EPS, CTX, DTYPE)