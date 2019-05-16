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


def get_time_in_unit(time_spent, base='us'):
    if base == 'us':
        base_frac = 1E-6
    elif base == 'ms':
        base_frac = 1E-3
    elif base == 's':
        base_frac = 1
    else:
        raise NotImplementedError
    assert time_spent[-1] == 's'
    if time_spent[-2] == 'm':
        return float(time_spent[:-2]) * 1E-3 / base_frac
    elif time_spent[-2] == 'u':
        return float(time_spent[:-2]) * 1E-6 / base_frac
    else:
        return float(time_spent[:-1]) / base_frac


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

    def fetch_run_time(self, keyword, unit='us'):
        """

        Returns
        -------
        ncalls_l : list of int
        avg_time_l : list of float
        min_time_l : list of float
        max_time_l : list of float
        kernel_name_l : list of str
        """
        ncalls_l = []
        avg_time_l = []
        min_time_l = []
        max_time_l = []
        kernel_name_l = []
        if type(keyword) == str:
            for i in range(self.profile_result.shape[0]):
                row = self.profile_result.loc[i]
                if keyword in row['name']:
                    ncalls_l.append(int(row['calls']))
                    avg_time_l.append(get_time_in_unit(row['avg']))
                    min_time_l.append(get_time_in_unit(row['min']))
                    max_time_l.append(get_time_in_unit(row['max']))
                    kernel_name_l.append(row['name'])
        elif type(keyword) == list:
            for sub_keword in keyword:
                sub_ncalls_l, sub_avg_time_l, sub_min_time_l, sub_max_time_l, sub_kernel_name_l =\
                    self.fetch_run_time(sub_keword, unit=unit)
                ncalls_l.extend(sub_ncalls_l)
                avg_time_l.extend(sub_avg_time_l)
                min_time_l.extend(sub_min_time_l)
                max_time_l.extend(sub_max_time_l)
                kernel_name_l.extend(sub_kernel_name_l)
        return ncalls_l, avg_time_l, min_time_l, max_time_l, kernel_name_l


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

