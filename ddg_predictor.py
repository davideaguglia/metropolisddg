import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import Bio
from Bio import SeqIO
import Bio.PDB as PDB
from transformers import AutoTokenizer, EsmForMaskedLM, AutoModelForMaskedLM
import torch
from scipy import stats
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from xgboost import XGBRegressor

class Model_class:

    ### Initialization
    def __init__(self, device : int = -1, temperature : float = 1.):
    
        self.tokenizer_mask = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        self.model_mask = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
        if self.device >= 0:
            self.model_mask.cuda(f'cuda:{device}')
        self.model_mask = self.model_mask.eval()

        self.RF = pickle.load(open('models/RF_model.sav', 'rb'))
        self.XG = pickle.load(open('models/XG_model.sav', 'rb'))
        self.KN = pickle.load(open('models/KN_model.sav', 'rb'))

        self.residues = list("ARNDCQEGHILKMFPSTWYV")
        self.device = device
        self.T = temperature

    ### Calculate prediction using the mean of the three models
    def predict(self, x):
        out = self.RF.predict(x)
        out += self.XG.predict(xgb.DMatrix(x))
        out += self.KN.predict(x)

        return out/3

    def get_ESM_prob(self, sequence, position):
        input_ids = self.tokenizer_mask.encode(f'{sequence}', return_tensors="pt")
        if self.device >= 0:
            input_ids = input_ids.cuda(f"cuda:{self.device}")
        # !!! input_ids has two more element, the starting and the final tokens
        position +=1
        inputs_mask_clone = input_ids.clone()
        inputs_mask_clone[0, position] = self.tokenizer_mask.mask_token_id        
        
        with torch.no_grad():
            logits = self.model_mask(inputs_mask_clone).logits
        logits = logits.cpu()
        
        prob = torch.nn.functional.softmax(logits[0, position], dim=0)
        probabilities = np.zeros(len(self.residues))
        for j, amino in enumerate(self.residues):
            probabilities[j] = prob[self.tokenizer_mask.convert_tokens_to_ids(amino)].item()

        return probabilities
        

    def get_ml_input(self, probabilities, old_amino):
        output = np.zeros((len(self.residues), 2, len(self.residues)))
        old_idx = np.where(np.array(self.residues) == old_amino)[0] 
        
        for i, new_amino in enumerate(self.residues):
            output[i][:][0] = probabilities

            new_idx = np.where(np.array(self.residues) == new_amino)[0]

            output[i][1][new_idx] += 1
            output[i][1][old_idx] += -1
        
        output = output.reshape(len(self.residues), -1)
        return output
    
    def get_ddG_distribution(self, sequence, position):
        old_amino = self.residues[np.where(np.array(self.residues) == sequence[position])[0][0]]
        ddG = np.zeros(len(self.residues))

        probabilities = self.get_ESM_prob(sequence, position)
        
        ddG = self.predict(self.get_ml_input(probabilities, old_amino))
        ddG = np.exp(-ddG/self.T)
        return ddG/sum(ddG)
    
    def predict_single_mutation(self, sequence, position, wt_amino, new_amino):
        output = np.zeros((2, len(self.residues)))
        input_ids = self.tokenizer_mask.encode(f'{sequence}', return_tensors="pt")

        # !!! input_ids has two more element, the starting and the final tokens
        position +=1
        inputs_mask_clone = input_ids.clone()
        inputs_mask_clone[0, position] = self.tokenizer_mask.mask_token_id
        
        with torch.no_grad():
            logits = self.model_mask(inputs_mask_clone).logits
        logits = logits.cpu()

        prob = torch.nn.functional.softmax(logits[0, position], dim=0)
        probabilities = np.zeros(len(self.residues))

        for j, amino in enumerate(self.residues):
            probabilities[j] = prob[self.tokenizer_mask.convert_tokens_to_ids(amino)].item()
        output[:][0] = probabilities

        old_idx = np.where(np.array(self.residues) == wt_amino)[0]
        new_idx = np.where(np.array(self.residues) == new_amino)[0]

        output[1][new_idx] += 1
        output[1][old_idx] += -1
        
        return self.predict(output.reshape(1, -1))



