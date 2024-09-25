#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.special import softmax
from jax.tree_util import tree_map
import os, glob, sys
from subprocess import call
from os import listdir
from os.path import isfile, isdir
import argparse
import time

import nvidia_smi
from numba import njit

import torch
from transformers import AutoTokenizer, EsmModel, EsmForProteinFolding, QuantoConfig, EsmForMaskedLM
import numpy as np
import os
from ddg_predictor import *


### -------------------------------------- BASIC ALGORITHM ------------------------------------- ###
class Basic_class:

    ### Initialization
    def __init__(
            self,
            device : int = -1,
            distance_threshold : float = 4.
    ):

        #torch.cuda.empty_cache()
    
        self.device = device
        if self.device >= 0:
            torch.cuda.set_device(device)


        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        # self.quantization_config = QuantoConfig(weights="int8")
        # self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", device_map=f"cuda:{self.device}", quantization_config=self.quantization_config)
        # self.model = self.model.eval()
        self.tokenizer = []
        self.model = []

        # Get correlation matrix
        lines = open('corr_matrix_zerodiag.txt', 'r').readlines()
        self.corr_probs = [[float(num) for num in line.split(' ')] for line in lines]
        self.corr_probs = np.array(self.corr_probs)

        # Get probability of mutation for a the sites
        lines = open('site_probability_zerodiag.txt', 'r').readlines()
        self.probs_site = [[float(num) for num in line.split(' ')] for line in lines]
        self.probs_site = np.array(self.probs_site)[:, 0]
        
        self.distance_threshold = distance_threshold

        print('Basic class, status:')
        print(f'model:              esmfold_v1')
        print(f'device:             {self.device}')
        print(f'distance threshold: {self.distance_threshold} [A]', '\n')


    ### Calculate contact map through esmfold model
    def calculate_contacts(self, sequence, method = 'explicit', return_plddt = False, return_trivial = False):
        
        self.inputs = self.tokenizer([f'{sequence}'], return_tensors="pt", add_special_tokens=False)['input_ids']
        self.inputs = self.inputs.cuda(f"cuda:{self.device}")
        
        with torch.no_grad():
            output = self.model(self.inputs)

        output = {k: v.to("cpu").numpy() for k, v in output.items()}

        plddt = output['plddt'][0, :, 1]
        plddt_mask = output['atom37_atom_exists'][0].astype(bool)        
        self._check_Ca(plddt_mask, sequence)

        if method == 'explicit':
            positions = output['positions'][-1, 0]
            positions_mask = output['atom14_atom_exists'][0].astype(bool)
            distance_matrix = self._calculate_distance_matrix(positions, positions_mask)
            contact_map = (distance_matrix < self.distance_threshold).astype(int)
        elif method == 'implicit':
            bins = np.append(0, np.linspace(2.3125, 21.6875, 63))
            contact_map = softmax(output['distogram_logits'], -1)[0]
            contact_map = contact_map[..., bins < 8].sum(-1)
            
        torch.cuda.empty_cache()
        del output
        
        if not return_trivial: 
            contact_map = self._eliminate_trivial_contacts(contact_map)
        if return_plddt: 
            return contact_map, plddt
        else:
            return contact_map



    ### Calculate distance matrix for given chain
    def _calculate_distance_matrix(self, positions, positions_mask):
        distance_matrix = np.zeros( (len(positions), len(positions)) )
        idxs = np.arange(0, len(positions))
        for row in idxs:
            for col in idxs[idxs > row]:
                residue_one = positions[row, positions_mask[row]]
                residue_two = positions[col, positions_mask[col]]
                distance_matrix[row, col] = self._calculate_residue_distance(residue_one, residue_two)
                distance_matrix[col, row] = distance_matrix[row, col]
        return distance_matrix


    ### Calculate residue distance (minimum distance between atoms for the given residues)
    @staticmethod
    @njit
    def _calculate_residue_distance(residue_one, residue_two):
        distances = []
        for xyz_one in residue_one:
            for xyz_two in residue_two:
                diff2_xyz = (xyz_one - xyz_two)**2
                distance = np.sqrt(diff2_xyz[0]+diff2_xyz[1]+diff2_xyz[2])
                distances.append(distance)
        return np.min(np.asarray(distances))


    ### Check for missing C-alphas in the chain
    def _check_Ca(self, plddt_mask, sequence):
        check = np.all(plddt_mask[:, 1])
        assert check, fr'Missing C-$\alpha$ for loaded sequence: {sequence}'

    
    ### Eliminate trivial contacts from the contact map
    @staticmethod
    @njit
    def _eliminate_trivial_contacts(contact_map):
        for row in range(len(contact_map)):
            contact_map[row, row] = 0
            if row > 0: contact_map[row, row - 1] = 0
            if row < len(contact_map) - 1: contact_map[row, row + 1] = 0
        return contact_map


    ### Set modules
    def set_device(self, device):
        self.device = device
        torch.cuda.set_device(device)

    def set_distance_threshold(self, distance_threshold):
        self.distance_threshold






### -------------------------------------- MUTATION ALGORITHM ------------------------------------- ###
class Mutation_class(Basic_class):

    ### Initialization
    def __init__(
            self,
            wt_sequence : str,
            ref_sequence : str = '',
            starting_sequence: str = '',
            metr_mutations : int = 100,
            eq_mutations : int = 0,
            T : float = 1.,
            k : float = 1.,
            seed : int = 0,
            unique_length : int = 10000,
            results_dir : str = 'results',
            restart_bool : bool = False,
            device : int = 0,
            distance_threshold : float = 4.,
            number_mutations : int = 5
    ):

        super().__init__(
                device = device,
                distance_threshold = distance_threshold
        )
        
        #self.stempering = self.InitSTempering()
        self.mask = Model_class(self.device, 0.7)

        self.n_mutations = number_mutations
        self.mutated_sites = []
        self.proposed_amino = []
        
        # Sequences
        self.wt_sequence = wt_sequence
        #self.wt_contacts = self.calculate_contacts(self.wt_sequence)
        self.wt_contacts = []

        if ref_sequence == '':
            self.ref_sequence = self.wt_sequence
            self.ref_contacts = self.wt_contacts.copy()
        else:
            if len(ref_sequence) == len(wt_sequence): 
                self.ref_sequence = ref_sequence
                #self.ref_contacts = self.calculate_contacts(self.ref_sequence)
            else: 
                raise ValueError("Mutation_class.__init__(): starting sequence ref_sequence must have the same length of the wild-type sequence.")
        self.ref_array = np.array(list(self.ref_sequence))

        if starting_sequence == '':
            self.starting_sequence = self.ref_sequence
            self.starting_contacts = self.ref_contacts.copy()
        else:
            if len(ref_sequence) == len(wt_sequence): 
                self.starting_sequence = starting_sequence
                #self.starting_contacts = self.calculate_contacts(self.starting_sequence)
            else: 
                raise ValueError("Mutation_class.__init__(): starting sequence starting_sequence must have the same length of the wild-type sequence.")
        self.sequence_length = len(self.wt_sequence)

        # Distance definitions
        self.distmatrix = pd.read_csv('inputs/DistPAM1.csv')
        self.distmatrix = self.distmatrix.drop(columns = ['Unnamed: 0'])
        self.residues = tuple(self.distmatrix.columns)
        self.distmatrix = np.array(self.distmatrix)

        # Parameters
        if metr_mutations > 0: self.metr_mutations = metr_mutations
        else: raise ValueError("Mutation_class.__init__(): metr_mutations must be positive.")

        if eq_mutations >= 0: self.eq_mutations = eq_mutations
        else: raise ValueError("Mutation_class.__init__(): eq_mutations can't be negative.")

        if T > 0.: self.T = T
        else: raise ValueError("Mutation_class.__init__(): T must be positive.")

        if k >= 0.: self.k = k
        else: raise ValueError("Mutation_class.__init__(): k can't be negative.")

        self.seed = seed
        np.random.seed(self.seed)

        if unique_length >= 0: self.unique_length = unique_length
        else: raise ValueError("Mutation_class.__init__(): unique_length can't be negative.")

        # Initialization
        self._get_id()
        self._check_directory(results_dir)
        self.set_restart_bool(restart_bool)



    ### Prepare simulation id
    def _get_id(self):
        T_str = str(self.T)
        k_str = str(self.k)
        s_str = str(self.seed)
        self.file_id = f'T{T_str}_k{k_str}_s{s_str}'



    ### Check for directory to store simulation mutants and data
    def _check_directory(self, results_dir):
        if results_dir[-1] != '/' and results_dir[:3] != './' and results_dir[:4] != '../':
            self.results_dir = results_dir
        else:
            if results_dir[-1] == '/':
                self.results_dir = results_dir[:-1]
            if results_dir[:3] == './':
                self.results_dir = results_dir[3:]
            if results_dir[:4] == '../':
                self.results_dir = results_dir[4:]
        
        path = self.results_dir.split('/')
        actual_dir = '..'
        for idx, new_dir in enumerate(path):
            if idx > 0:
                actual_dir = actual_dir + '/' + path[idx - 1]
            onlydirs = [d for d in listdir(f'{actual_dir}') if isdir(f'{actual_dir}/{d}')]
            if (new_dir in onlydirs) == False: 
                os.mkdir(f'{actual_dir}/{new_dir}')



    ### Reset parameters for new simulation
    def _reset(self):
        self.last_sequence = self.starting_sequence
        self.last_eff_energy = self.calculate_effective_energy(self.starting_contacts)
        self.last_ddG = 0
        self.last_PAM1_distance, self.last_Hamm_distance = self.get_distances(self.starting_sequence)
        self.generation = 0
        self.accepted_mutations = 0
        self.max_distance = self.last_PAM1_distance
        
        self.unique_sequences = np.array([self.starting_sequence], dtype = str)
        self.unique_data = np.array([[self.last_eff_energy, self.last_ddG, self.last_PAM1_distance, self.last_Hamm_distance]], dtype = float)

        paths = [f'../{self.results_dir}/mutants_{self.file_id}.dat', f'../{self.results_dir}/data_{self.file_id}.dat', f'../{self.results_dir}/status_{self.file_id}.txt']
        onlyfiles = [f'../{self.results_dir}/{f}' for f in listdir(f'../{self.results_dir}') if isfile(f'../{self.results_dir}/{f}')]

        for path in paths:
            if path in onlyfiles:
                call(['rm', path])



    ### Restart the previous simulation
    def _restart(self):
        # Find files
        paths = [f'../{self.results_dir}/mutants_{self.file_id}.dat', f'../{self.results_dir}/data_{self.file_id}.dat']
        onlyfiles = [f'../{self.results_dir}/{f}' for f in listdir(f'../{self.results_dir}') if isfile(f'../{self.results_dir}/{f}')]
        check = np.all( [path in onlyfiles for path in paths] )
        
        if check:
            # Discard incomplete data
            with open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'r') as mutants_file:
                muts_lines = mutants_file.readlines()
                muts_num = len(muts_lines)

            with open(f'../{self.results_dir}/data_{self.file_id}.dat', 'r') as data_file:
                data_lines = data_file.readlines()
                data_num = len(data_lines)

            if muts_num < data_num:
                with open(f'../{self.results_dir}/data_{self.file_id}.dat', 'w') as data_file:
                    for line in data_lines[:muts_num]: 
                        print(line, end = '', file = data_file)

            elif muts_num > data_num:
                with open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'w') as mutants_file:
                    for line in muts_lines[:data_num]: 
                        print(line, end = '', file = mutants_file)

            # Last sequence residues and contacts, and unique mutations
            with open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'r') as mutants_file:
                lines = mutants_file.readlines()
            
            last_line = lines[-1].split('\t')
            self.last_sequence = last_line[1]
            if self.last_sequence[-1] == '\n': self.last_sequence = self.last_sequence[:-1]
            
            sequences = np.array([line.split('\t')[1][:-1] for line in lines], dtype = str)
            self.unique_sequences = np.unique(sequences)
            if len(self.unique_sequences) > self.unique_length:
                self.unique_sequences = self.unique_sequences[(len(self.unique_sequences) - self.unique_length):]

            # Last sequence data, and unique mutations data
            with open(f'../{self.results_dir}/data_{self.file_id}.dat', 'r') as data_file:
                lines = np.array(data_file.readlines(), dtype = str)

            last_line = lines[-1].split('\t')
            self.generation = int( last_line[0] )
            self.last_eff_energy = float( last_line[1] )
            self.last_ddG = float( last_line[2] )
            self.last_PAM1_distance = float( last_line[3] )
            self.last_Hamm_distance = int( float(last_line[4]) )
            self.accepted_mutations = int( float(last_line[5]) * self.generation )
            self.T = float( last_line[6] )
            self.k = float( last_line[7] )

            distances = [float( line.split('\t')[3] ) for line in lines]
            self.max_distance = np.min(distances)

            obs_idxs = [1, 2, 3, 4] # i.e. Effective energy, ddG, PAM1 distance, Hamming distance
            masks = [(sequences == unique_sequence) for unique_sequence in self.unique_sequences]
            self.unique_data = np.array([[0.] * len(obs_idxs)], dtype = float)
            for mask in masks:
                data = np.array(lines[mask][0].split('\t'), dtype = float)[obs_idxs]
                self.unique_data = np.append(self.unique_data, [data], axis = 0)
            self.unique_data = self.unique_data[1:]

            assert len(self.unique_sequences) == len(self.unique_data), 'Mismatch between the unique lists.'
            
        else:
            self._reset()



    ### Calculate Hamming distance and PAM1 distance from the reference sequence (ref_sequence)
    def get_distances(self, mt_sequence):
        mt_array = np.array(list(mt_sequence))
        new_residues_idxs = np.where(self.ref_array != mt_array)[0]

        # Hamming distance
        Hamm_distance = len(new_residues_idxs) / self.sequence_length

        # PAM1 distance
        old_residues = self.ref_array[new_residues_idxs]
        new_residues = mt_array[new_residues_idxs]
        PAM1_distance = 0.
        for old_residue, new_residue in zip(old_residues, new_residues):
            old_idx = self.residues.index(old_residue)
            new_idx = self.residues.index(new_residue)
            PAM1_distance += self.distmatrix[new_idx, old_idx]
        PAM1_distance = PAM1_distance / self.sequence_length
        
        return PAM1_distance, Hamm_distance

    ### Produce multiple-mutations of the last metropolis sequence using mask prediction
    def multiple_mutation(self):
        self.mutated_sites = []
        self.proposed_amino = []

        positions = np.random.randint(153, size=(self.n_mutations))

        mt_sequence = self.last_sequence
        
        for n in range(self.n_mutations):
            probabilities = self.mask.get_ddG_distribution(mt_sequence, positions[n])
            probabilities = [sum(probabilities[:i+1]) for i in range(20)]
            probabilities[-1] = 1.
            random = np.random.rand()
            residue = self.residues[np.where(np.array(probabilities)>random)[0][0]]
            self.proposed_amino.append(residue)
            self.mutated_sites.append(positions[n])
            mt_sequence = mt_sequence[:positions[n]] + residue + mt_sequence[(positions[n] + 1):]
        
        return mt_sequence 

    ### Calculate effective as number of modified contacts divided by the number of the wild-type protein contacts
    def calculate_effective_energy(self, mt_contacts):
        # Modified contacts fraction
        mod_diff = abs(mt_contacts - self.wt_contacts)
        norm = np.sum(mt_contacts) + np.sum(self.wt_contacts)
        eff_en = np.sum(mod_diff) / norm
        return eff_en



    ### Calculate ddG
    def calculate_ddG(self):
        pass


    ### Metropolis algorithm
    def metropolis(self, equilibration = False, print_start = True):
        # Preparing the simulation
        if equilibration:
            mutations = self.eq_mutations
            mutants_file = open(f'../{self.results_dir}/eq_mutants_{self.file_id}.dat', 'w')
            data_file = open(f'../{self.results_dir}/eq_data_{self.file_id}.dat', 'w')
            info_file = open(f'../{self.results_dir}/info_{self.file_id}.dat', 'a')
        else:
            mutations = self.metr_mutations
            mutants_file = open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'a')
            data_file = open(f'../{self.results_dir}/data_{self.file_id}.dat', 'a')
            info_file = open(f'../{self.results_dir}/info_{self.file_id}.dat', 'w')

        if print_start:
            print(f'{self.generation}\t{self.last_sequence}', file = mutants_file)
            if self.generation == 0:
                print(f'{self.generation}\t{format(self.last_eff_energy, ".15f")}\t{format(self.last_ddG, ".15f")}\t{format(self.last_PAM1_distance, ".15f")}\t{format(self.last_Hamm_distance, ".15f")}\t{self.accepted_mutations}\t{format(self.max_distance, ".15f")}\t{self.T}\t{self.k}\t{self.sequence_length}', file = data_file)
            else:
                print(f'{self.generation}\t{format(self.last_eff_energy, ".15f")}\t{format(self.last_ddG, ".15f")}\t{format(self.last_PAM1_distance, ".15f")}\t{format(self.last_Hamm_distance, ".15f")}\t{self.accepted_mutations / self.generation}\t{format(self.max_distance, ".15f")}\t{self.T}\t{self.k}\t{self.sequence_length}', file = data_file)
        
        self.last_ddG = 0
        # Metropolis
        for imut in range(mutations):
            # Mutant generation
            self.generation += 1
            
            position = np.random.randint(len(self.last_sequence))
            old_amino = self.last_sequence[position]
            new_amino = self.residues[np.random.randint(len(self.residues))]
            mt_sequence = mt_sequence[:position] + new_amino + mt_sequence[(position + 1):]
            #mt_sequence = self.multiple_mutation()

            # Observables
            mask = self.unique_sequences == mt_sequence
            assert np.sum(mask.astype(int)) <= 1, "Too many 'unique' sequences equal to the same mutant."
            if np.any(mask):
                assert self.unique_sequences[mask][0] == mt_sequence, 'Wrong mask.'
                eff_energy = self.unique_data[mask, 0][0]
                ddG = self.unique_data[mask, 1][0]
                PAM1_distance = self.unique_data[mask, 2][0]
                Hamm_distance = self.unique_data[mask, 3][0]
            else:
                #mt_contacts = self.calculate_contacts(mt_sequence)
                #eff_energy = self.calculate_effective_energy(mt_contacts)
                ddG = -self.mask.predict_single_mutation(self.last_sequence, position, old_amino, new_amino)
                PAM1_distance, Hamm_distance = self.get_distances(mt_sequence)

                self.unique_sequences = np.append(self.unique_sequences, mt_sequence)
                self.unique_data = np.append(self.unique_data, [[eff_energy, ddG, PAM1_distance, Hamm_distance]], axis = 0)
                assert len(self.unique_sequences) == len(self.unique_data), "Length of unique sequences and unique data must coincide."
                if len(self.unique_sequences) > self.unique_length:
                    self.unique_sequences = self.unique_sequences[1:]
                    self.unique_data = self.unique_data[1:]

            # Update lists
            weight = (ddG) / self.T
            if PAM1_distance > self.max_distance: weight += (self.k / 2.) * ((PAM1_distance - self.max_distance) ** 2. - (self.last_PAM1_distance - self.max_distance) ** 2.)
            
            p = np.random.rand()
            if p <= np.exp(-weight):
                self.last_sequence = mt_sequence
                #self.last_eff_energy = eff_energy
                self.last_ddG += ddG
                self.last_PAM1_distance = PAM1_distance
                self.last_Hamm_distance = Hamm_distance
                self.accepted_mutations += 1
                if PAM1_distance < self.max_distance: self.max_distance = PAM1_distance
            
            #self.STempering(imut)
            #self.T = self.stempering.contents.temp[self.stempering.contents.itemp]

            #torch.cuda.empty_cache()

            # Save data
            print(f'{self.generation}\t{self.mutated_sites}\t{self.proposed_amino}\t{eff_energy}\t{self.last_eff_energy}', file = info_file)
            print(f'{self.generation}\t{self.last_sequence}', file = mutants_file)
            print(f'{self.generation}\t{format(self.last_eff_energy, ".15f")}\t{format(self.last_ddG, ".15f")}\t{format(self.last_PAM1_distance, ".15f")}\t{format(self.last_Hamm_distance, ".15f")}\t{self.accepted_mutations / self.generation}\t{format(self.max_distance, ".15f")}\t{self.T}\t{self.k}\t{self.sequence_length}', file = data_file)
        
        #print(f'Last sequence: {self.last_sequence}')
        # Close data files
        mutants_file.close()
        data_file.close()



    ### Print status
    def print_status(self):
        print(f'Mutation algorithm protein:')
        print(f'Wild-type sequence: {self.wt_sequence}')
        
        if self.ref_sequence != self.wt_sequence: 
            print(f'Reference sequence: {self.ref_sequence}')
        else: 
            print(f'Reference sequence: wild-type sequence')
        
        if self.starting_sequence != self.wt_sequence and self.starting_sequence != self.ref_sequence:
            print(f'Starting sequence:  {self.starting_sequence}\n')
        else:
            if self.starting_sequence == self.wt_sequence:
                print(f'Starting sequence:  wild-type sequence\n')
            else:
                print(f'Starting sequence:  reference sequence\n')

        print(f'Mutation algorithm parameters:')
        print(f'metropolis mutations:    {self.metr_mutations}')
        print(f'equilibration mutations: {self.eq_mutations}')
        print(f'temperature:             {self.T}')
        print(f'k:                       {self.k}')
        print(f'seed:                    {self.seed}')
        print(f'unique length:           {self.unique_length}')
        print(f'number of mutations:     {self.n_mutations}')
        print(f'results directory:       ../{self.results_dir}')
        print(f'restart:                 {self.restart_bool}\n')



    ### Print last mutation
    def print_last_mutation(self, print_file = sys.stdout):
        if print_file != sys.stdout:
            print_file = open(print_file, 'a')

        print(f'Generation:  {self.generation}', file = print_file)
        print(f'Wild tipe:   {self.wt_sequence}', file = print_file)
        
        if self.ref_sequence != self.wt_sequence:
            print(f'Reference sequence: {self.ref_sequence}', file = print_file)
        else:
            print(f'Reference sequence: wild-type sequence', file = print_file)
        
        if self.starting_sequence != self.wt_sequence and self.starting_sequence != self.ref_sequence:
            print(f'Starting sequence:  {self.starting_sequence}\n', file = print_file)
        else:
            if self.starting_sequence == self.wt_sequence:
                print(f'Starting sequence:  wild-type sequence\n', file = print_file)      
            else:
                print(f'Starting sequence:  reference sequence\n', file = print_file)
        
        print(f'Last mutant: {self.last_sequence}', file = print_file)
        print(f'Effective energy: {self.last_eff_energy}', file = print_file)
        print(f'ddG:              {self.last_ddG}', file = print_file)
        print(f'PAM1 distance:    {self.last_PAM1_distance}', file = print_file)
        print(f'Hamming distance: {self.last_Hamm_distance}', file = print_file)
        print(f'Max distance:     {self.max_distance}\n', file = print_file)

        if print_file != sys.stdout:
            print_file.close()



    ### Set modules
    def set_wt_sequence(self, wt_sequence : str):
        self.wt_sequence = wt_sequence
        #self.wt_contacts = self.calculate_contacts(self.wt_sequence)
        self.sequence_length = len(self.wt_sequence)
        if len(self.ref_sequence) != self.sequence_length:
            self.ref_sequence = self.wt_sequence
            self.ref_contacts = self.wt_contacts.copy()
        self._reset()

    def set_ref_sequence(self, ref_sequence : str):
        if len(ref_sequence) == len(self.wt_sequence):
            self.ref_sequence = ref_sequence
            self.ref_array = np.array(list(self.ref_sequence))
            #self.ref_contacts = self.calculate_contacts(ref_sequence)
        else: 
            raise ValueError("Mutation_class.set_ref_sequence(): starting sequence ref_sequence must have the same length of the wild-type sequence.")

    def set_starting_sequence(self, starting_sequence : str):
        if len(starting_sequence) == len(self.wt_sequence):
            self.starting_sequence = starting_sequence
            #self.starting_contacts = self.calculate_contacts(starting_sequence)
        else: 
            raise ValueError("Mutation_class.set_starting_sequence(): starting sequence starting_sequence must have the same length of the wild-type sequence.")

    def set_metr_mutations(self, metr_mutations : int): 
        if metr_mutations > 0: self.metr_mutations = metr_mutations
        else: raise ValueError("Mutation_class.set_metr_mutations(): metr_mutations must be positive.")

    def set_eq_mutations(self, eq_mutations : int):
        if eq_mutations >= 0: self.eq_mutations = eq_mutations
        else: raise ValueError("Mutation_class.set_eq_mutations(): eq_mutations can't be negative.")

    def set_T(self, T : float):
        if T > 0.: self.T = T
        else: raise ValueError("Mutation_class.__init__(): T must be positive.")
        self._get_id()

    def set_k(self, k : float):
        if k >= 0.: self.k = k
        else: raise ValueError("Mutation_class.set_k(): k can't be negative.")
        self._get_id()

    def set_seed(self, seed : int):
        self.seed = seed
        np.random.seed(self.seed)
        self._get_id()

    def set_unique_length(self, unique_length : int):
        if unique_length >= 0: self.unique_length = unique_length
        else: raise ValueError("Mutation_class.__init__(): unique_length can't be negative.")
        self.restart_bool()

    def set_results_dir(self, results_dir : str):
        self.results_dir = results_dir

    def set_restart_bool(self, restart_bool : bool):
        self.restart_bool = restart_bool
        if self.restart_bool: self._restart()
        else: self._reset()
