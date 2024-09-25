from os import listdir
from os.path import isfile, isdir
import numpy as np
import argparse

## Read desired input from input-file
def load_inputs(input_file, tuples):
    with open(input_file, 'r') as f:
        lines = np.array( f.readlines() )
    start = np.where(lines == '\n')[0] + 1
    if len(start) > 1: start = start[-1]

    parameters = {}
    for tupl in tuples:
        key, typ = tupl[0], tupl[1]
        key_idx = (np.where(lines == f'\t>> {key}\n')[0] - start)[0]
        if typ == bool:
            parameters[key] = typ(int(lines[key_idx]))
        elif typ == str:
            parameters[key] = typ(lines[key_idx])[:-1]
        else:
            parameters[key] = typ(lines[key_idx])
    return parameters


## Load simulation data
def load_data(d, return_eq = True):
    datafiles = np.array([f for f in listdir(d) if isfile(f'{d}/{f}') and 'data' in f])
    
    file = datafiles[[not 'eq' in datafile for datafile in datafiles]][0]
    with open(f'{d}/{file}', 'r') as f:
        lines = f.readlines()
    splitted_lines = [line.split('\t') for line in lines]
    data = np.array(splitted_lines).astype(float)

    if return_eq:
        eq_file = datafiles[['eq' in datafile for datafile in datafiles]][0]
        with open(f'{d}/{eq_file}', 'r') as f:
            lines = f.readlines()
        splitted_lines = [line.split('\t') for line in lines]
        eq_data = np.array(splitted_lines).astype(float)

        return eq_data, data

    else:
        return data


## Load simulation mutations
def load_muts(d, return_eq = True, return_pars = True):
    mutsfiles = np.array([f for f in listdir(d) if isfile(f'{d}/{f}') and 'mutants' in f])
        
    file = mutsfiles[[not 'eq' in mutsfile for mutsfile in mutsfiles]][0]
    with open(f'{d}/{file}', 'r') as f:
        lines = f.readlines()
    splitted_lines = [line.split('\t') for line in lines]
    muts = np.array(splitted_lines).astype(str)
    for idx in range(len(muts)):
        muts[idx, 1] = muts[idx, 1][:-1]

    if return_eq:
        eq_file = mutsfiles[['eq' in mutsfile for mutsfile in mutsfiles]][0]
        with open(f'{d}/{eq_file}', 'r') as f:
            lines = f.readlines()
        splitted_lines = [line.split('\t') for line in lines]
        eq_muts = np.array(splitted_lines).astype(str)
        for idx in range(len(eq_muts)):
            eq_muts[idx, 1] = eq_muts[idx, 1][:-1]

    if return_pars:
        splitted_d = d.split('_')
        T = float(splitted_d[-2][1:])
        gamma = float(splitted_d[-1][1:])

    if return_eq and return_pars:
        return eq_muts, muts, T, gamma
    elif return_eq:
        return eq_muts, muts
    elif return_pars:
        return muts, T, gamma
    else:
        return muts
