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
USE_GPU = 1
DTYPE = 'float32'
TIME_R = r'\d+\.?\d*'
LN_OUT_REG = r'Forward: ({})us, Backward: ({})us'.format(TIME_R, TIME_R)
MX_FWD_KEYWORD = 'LayerNormFusedForwardKernel'
MX_BWD_DATA_KEYWORD = 'LayerNormFusedBackwardKernel_Data'
MX_BWD_GAMMA_BETA_KEYWORD = ['LayerNormFusedBackwardKernel_PartGammaBeta', 'LayerNormFusedBackwardKernel_GammaBeta']


def as_markdown_table(df):
    ret = ''
    # Print header
    ret += ' ' + '|' +  '|'.join([' B={} '.format(ele) for ele in df.columns]) + '\n'
    ret += '---' + '|' +  '|'.join(['---' for ele in df.columns]) + '\n'
    for c in df.index:
        ret += 'C={}'.format(c) + '|' + '|'.join([' {:g} '.format(ele) for ele in df.loc[c, :]]) + '\n'
    return ret


def test_speed(codebase, test_batch_l, test_channel_l, eps, use_gpu, dtype, profile_nv):
    py_time_fwd_df = pd.DataFrame(columns=test_batch_l, index=test_channel_l)
    py_time_bwd_df = pd.DataFrame(columns=test_batch_l, index=test_channel_l)
    nv_time_fwd_df = pd.DataFrame(columns=test_batch_l, index=test_channel_l)
    nv_time_bwd_df = pd.DataFrame(columns=test_batch_l, index=test_channel_l)
    nv_time_bwd_data_df = pd.DataFrame(columns=test_batch_l, index=test_channel_l)
    nv_time_bwd_gamma_beta_df = pd.DataFrame(columns=test_batch_l, index=test_channel_l)
    for nbatch in test_batch_l:
        for nchannel in test_channel_l:
            if codebase == 'mxnet':
                run_args = [PYTHON_EXE, 'layer_norm_mx.py', '--use_gpu', str(use_gpu), '--nbatch', str(nchannel),
                            '--eps', str(eps), '--dtype', dtype, '--nrepeat', str(N_REPEAT)]
                fwd_keyword = 'LayerNormFusedForwardKernel'
                bwd_data_keyword = 'LayerNormFusedBackwardKernel_Data'
                bwd_gamma_beta_keyword = ['LayerNormFusedBackwardKernel_PartGammaBeta', 'LayerNormFusedBackwardKernel_GammaBeta']
            elif codebase == 'pytorch':
                run_args = [PYTHON_EXE, 'layer_norm_pytorch.py', '--use_gpu', str(use_gpu), '--nbatch', str(nchannel),
                            '--eps', str(eps), '--dtype', dtype, '--nrepeat', str(N_REPEAT)]
                fwd_keyword = None
                bwd_data_keyword = None
                bwd_gamma_beta_keyword = None
            elif codebase == 'pytorch_apex':
                run_args = [PYTHON_EXE, 'layer_norm_pytorch.py', '--use_gpu', str(use_gpu), '--nbatch', str(nchannel),
                            '--eps', str(eps), '--dtype', dtype, '--nrepeat', str(N_REPEAT), '--apex']
                fwd_keyword = 'cuApplyLayerNorm'
                bwd_data_keyword = 'cuComputeGradInput'
                bwd_gamma_beta_keyword = ['cuComputePartGradGammaBeta', 'cuComputeGradGammaBeta']
            else:
                raise NotImplementedError
            if profile_nv:
                run_args = [NVPROF_EXE] + run_args
            ret = subprocess.run(run_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            runfile_out = ret.stdout.decode('utf-8')
            fwd_time, bwd_time = re.match(LN_OUT_REG, runfile_out).groups()
            fwd_time = round(fwd_time, 1)
            bwd_time = round(bwd_time, 1)
            print(nbatch, nchannel, fwd_time, bwd_time)
            py_time_fwd_df.loc[nchannel, nbatch] = fwd_time
            py_time_bwd_df.loc[nchannel, nbatch] = bwd_time
            if profile_nv:
                nvprof_result = parse_nvprof_out(ret.stderr.decode('utf-8'))
                _, fwd_runtime, _, _, _ = nvprof_result.fetch_run_time(keyword=fwd_keyword, unit='us')
                fwd_runtime = sum(fwd_runtime)
                _, bwd_data_runtime, _, _, _ = nvprof_result.fetch_run_time(keyword=bwd_data_keyword, unit='us')
                bwd_data_runtime = sum(bwd_data_runtime)
                _, bwd_gamma_beta_runtime, _, _, _ = nvprof_result.fetch_run_time(keyword=bwd_gamma_beta_keyword, unit='us')
                bwd_gamma_beta_runtime = sum(bwd_gamma_beta_runtime)
                print(fwd_runtime, bwd_data_runtime, bwd_gamma_beta_runtime)
                total_bwd_runtime = bwd_data_runtime + bwd_gamma_beta_runtime
                nv_time_fwd_df[nchannel, nbatch] = round(fwd_runtime, 1)
                nv_time_bwd_df[nchannel, nbatch] = round(total_bwd_runtime, 1)
                nv_time_bwd_data_df[nchannel, nbatch] = round(bwd_data_runtime, 1)
                nv_time_bwd_gamma_beta_df[nchannel, nbatch] = round(bwd_gamma_beta_runtime, 1)
    return py_time_fwd_df, py_time_bwd_df, nv_time_fwd_df, nv_time_bwd_data_df, nv_time_bwd_data_df, nv_time_bwd_gamma_beta_df


apex_py_fwd_time, apex_py_bwd_time, apex_nv_fwd_time, apex_nv_bwd_time, apex_nv_bwd_data_time, apex_nv_bwd_gamma_beta_time \
    = test_speed('pytorch_apex', TEST_BATCH_L, TEST_CHANNEL_L, EPS, USE_GPU, DTYPE, profile_nv=True)
print('PyTorch Apex')
print('Forward (python timer)')
print(as_markdown_table(apex_py_fwd_time))
print('Backward (python timer)')
print(as_markdown_table(apex_py_bwd_time))
print('Forward (nvprof timer)')
print(as_markdown_table(apex_nv_fwd_time))
print('Backward (nvprof timer)')
print(as_markdown_table(apex_nv_bwd_time))
print('Backward Data (nvprof timer)')
print(as_markdown_table(apex_nv_bwd_data_time))
print('Backward Gamma & Beta (nvprof timer)')
print(as_markdown_table(apex_nv_bwd_gamma_beta_time))


mx_py_fwd_time, mx_py_bwd_time, mx_nv_fwd_time, mx_nv_bwd_time, mx_nv_bwd_data_time, mx_nv_bwd_gamma_beta_time =\
    test_speed('mxnet', TEST_BATCH_L, TEST_CHANNEL_L, EPS, USE_GPU, DTYPE, profile_nv=True)
