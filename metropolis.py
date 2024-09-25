import os
import argparse
import numpy as np
from time import time

from utils import load_inputs
from my_ratchet_classes_mult_ddG import *


def generate_random_sequence(wt_sequence):
    distmatrix = pd.read_csv('/kaggle/input/ddg-predictor/inputs/inputs/DistPAM1.csv')
    distmatrix = distmatrix.drop(columns = ['Unnamed: 0'])
    residues = tuple(distmatrix.columns)
    
    indexes = np.random.randint(len(residues), size = len(wt_sequence))
    mutant = ''.join([residues[index] for index in indexes])
    if mutant == wt_sequence:
        mutant = generate_random_sequence(wt_sequence)
        return mutant
    else:
        return mutant


def check_parameters(parameters):
    assert np.any([not parameters['casual_ref'], not parameters['casual_start']]), 'There is no sense in using both reference and starting sequence as casual ones. Choose only one of them.'
    if parameters['ref_sequence'] == '': 
        if parameters['casual_ref']: parameters['ref_sequence'] = generate_random_sequence(parameters['wt_sequence'])
        else: parameters['ref_sequence'] = parameters['wt_sequence']
    if parameters['starting_sequence'] == '': 
        if parameters['casual_start']: parameters['starting_sequence'] = generate_random_sequence(parameters['wt_sequence'])
        else: parameters['starting_sequence'] = parameters['ref_sequence']
    parameters['results_dir'] = f"{parameters['results_dir']}/{parameters['protein_name']}_T{str(parameters['T'])}_k{str(parameters['k'])}/s{parameters['seed']}"
    return parameters


def main(args):
    t0 = time.time()
    print(f'PID: {os.getpid()}\n')

    print('Defining wild-type sequence and simulation parameters...')
    tuples = [
        ('protein_name', str),
        ('wt_sequence', str),
        ('ref_sequence', str),
        ('starting_sequence', str),
        ('casual_ref', bool),
        ('casual_start', bool),
        ('metr_mutations', int),
        ('eq_mutations', int),
        ('T', float),
        ('k', float),
        ('seed', int),
        ('unique_length', int),
        ('results_dir', str),
        ('restart_bool', bool),
        ('device', int),
        ('distance_threshold', float),
        ('mutations_number', int)
    ]
    parameters = load_inputs(args.input_file, tuples)
    parameters = check_parameters(parameters)

    if parameters['casual_ref']: print(f'Casual reference sequence: {parameters["casual_ref"]}')
    elif parameters['casual_start']: print(f'Casual starting sequence: {parameters["casual_start"]}')

    print('Initiating algorithm...\n')
    algorithm = Mutation_class(
            wt_sequence = parameters['wt_sequence'],
            ref_sequence = parameters['ref_sequence'],
            starting_sequence = parameters['starting_sequence'],
            metr_mutations = parameters['metr_mutations'],
            eq_mutations = parameters['eq_mutations'],
            T = parameters['T'],
            k = parameters['k'],
            seed = parameters['seed'],
            unique_length = parameters['unique_length'],
            results_dir = parameters['results_dir'],
            restart_bool = parameters['restart_bool'],
            device = parameters['device'],
            distance_threshold = parameters['distance_threshold'],
            number_mutations = parameters['mutations_number']
    )
    algorithm.print_status()
    
    print('Starting the simulation...')
    t1 = time.time()
    if parameters['eq_mutations'] > 0:
        print(f'- Equilibration phase...')
        algorithm.metropolis(equilibration = True)
    print('- Metropolis...')
    algorithm.metropolis()
    print('Done!')
    print('Execution time', time.time()-t0)
    print('Execution time metropolis', time.time()-t1)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type = str,
        default = "/kaggle/input/ddg-predictor/inputs/inputs/inputs.txt",
        help = "str variable, input file used to load parameters values. Default: 'inputs/inputs.txt'."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
