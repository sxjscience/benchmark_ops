import io
import pandas as pd
import re
PERCENTAGE_R = r'\d+\.?\d*\%'
TIME_R = r'\d+\.?\d*[mu]*s'
SPACES_R = r'\s+'
INT_R = r'\d+'
ANY_SEQ_R = r'.+'

NVPROF_BEGIN_PROFILE_REG = r'==(\d+)== NVPROF is profiling process (\d+), command: (.+)'
NVPROF_STAT_REG = SPACES_R + '({})'.format(PERCENTAGE_R) +\
                  SPACES_R + '({})'.format(TIME_R) + \
                  SPACES_R + '({})'.format(INT_R) +\
                  SPACES_R + '({})'.format(TIME_R) +\
                  SPACES_R + '({})'.format(TIME_R) +\
                  SPACES_R + '({})'.format(TIME_R) +\
                  SPACES_R + '({})'.format(ANY_SEQ_R)


class NVProfResult(object):
    def __init__(self, command, profile_result):
        """

        Parameters
        ----------
        command
        result : pd.DataFrame
            Col indexes are 'percentage', 'time', 'calls', 'avg', 'min', 'max', 'name'
        """
        self.command = command
        self.profile_result = profile_result

    def fetch_run_time(self, match_reg, base='us'):
        """

        Returns
        -------
        ncalls : int
        avg_time: float
        min_time: float
        max_time: float
        kernel_name: str
        """
        pass


def parse_nvprof_out(data):
    """ Parse the output of 'nvprof COMMAND'

    Parameters
    ----------
    data : str or bytes


    Returns
    -------
    ret : NVProfResult
    """
    if type(data) == bytes:
        data = data.decode('utf-8')
    lines = data.splitlines()
    command = None
    df = pd.DataFrame(columns=['percentage', 'time', 'calls', 'avg', 'min', 'max', 'name'])
    has_parsed_command = False
    for line in lines:
        if not has_parsed_command:
            match_res = re.match(NVPROF_BEGIN_PROFILE_REG, line)
            if match_res is not None:
                first_id, second_id, command = match_res.groups()
                assert first_id == second_id
                has_parsed_command = True
                continue
        else:
            prefix = ' GPU activities:'
            if line.startswith(prefix):
                line = line[len(prefix):]
            match_res = re.match(NVPROF_STAT_REG, line)
            if match_res is not None:
                d_percentage, d_time, d_ncalls, d_avg, d_min, d_max, d_name = match_res.groups()
                df.loc[df.shape[0]] = [d_percentage, d_time, d_ncalls, d_avg, d_min, d_max, d_name]
    if command is None or df.shape[0] == 0:
        raise ValueError('The input data is not valid!')
    return NVProfResult(command=command, profile_result=df)

