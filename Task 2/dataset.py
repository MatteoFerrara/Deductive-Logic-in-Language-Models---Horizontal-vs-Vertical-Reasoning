

# This file is based on `dataset.py` from the original repository:
# https://github.com/abhay-sheshadri/backward-chaining-circuits
# It has been modified for the purposes of this project.

import numpy as np
import torch
from torch.utils.data import Dataset

from viz import parse_example


class GraphDataset_Splitted_Compact(Dataset):
    
    def __init__(self, n_states, file_name,dataset_depth_file_name=None):
        # Create a list of vocab
        number_tokens = sorted([str(i) for i in range(n_states)], key=lambda x: len(x), reverse=True)
        self.n_states = n_states
        self.max_seq_length = n_states * 4 + 2

        self.idx2tokens = [",", ":", "|", ">", "@"] + number_tokens
                
        self.tokens2idx = {token: idx for idx, token in enumerate(self.idx2tokens)}
        self.pad_token = self.tokens2idx[","]
        self.start_token = 1
        # Open up dataset file and load+tokenize strings
        self.X = []
        self.masks = []

        self.depths = None    
        if dataset_depth_file_name is not None:
            self.depths = []
            with open(dataset_depth_file_name, "r") as f:
                for line in f.readlines():
                    self.depths.append(line.rstrip())
        
        with open(file_name, "r") as f:
            for line in f.readlines():
                first_gt_after_colon = line.index('>', line.index(':'))
                before = line[:first_gt_after_colon]
                after = line[first_gt_after_colon+1:-1].replace('>', ',')
                after = ','.join(after.split(',')[:-1])
                line = before + "@" + after + '\n'

                # Tokenize string
                tokens = self.tokenize(line.rstrip())
                
                #remove from tokens the '>'
                tokens = [t for t in tokens if t != self.tokens2idx[">"]]

                #remove from tokens the commas that are after the '@'
                at_index = tokens.index(self.tokens2idx["@"])
                tokens = [t for i,t in enumerate(tokens) if not (i > at_index and t == self.tokens2idx[","])]

                padding_length = self.max_seq_length - len(tokens)
                tokens = np.array(tokens)
                tokens = np.pad(tokens, ((0, padding_length),), mode='constant', constant_values=0)

                self.X.append(tokens)
                # Create a mask
                index_tensor = np.arange(tokens.shape[0])
                mask = np.zeros_like(tokens, dtype=bool)
                #if reverse_output:
                start_idx = np.where(tokens == 4)[0].item()  # find index of '@'
                mask[start_idx+1:len(tokens)+1] = True
                # else:
                #     start_idx = np.where(tokens == 1)[0].item() # find index of ':'
                #     mask[start_idx+2:len(tokens)+2] = True

                self.masks.append(mask)

        # Stack arrays
        self.X = torch.from_numpy(np.stack(self.X))
        self.masks = torch.from_numpy(np.stack(self.masks))
        
    def tokenize(self, text):
        # Convert to token list
        tokens = []
        i = 0
        while i < len(text):
            for idx, word in enumerate(self.idx2tokens):
                if text.startswith(word, i):
                    tokens.append(idx)
                    i += len(word)
                    break
            else:
                i += 1
        # Convert to fixed length numpy array
        tokens_arr = np.array(tokens)
        
        return tokens_arr

    def untokenize(self, tokens):
        substrings = [self.idx2tokens[idx] for idx in tokens]
        return "".join(substrings).rstrip(",")
    
    def visualize_example(self, index):
        string = self.untokenize(self[index][0])
        parse_example(string)

    def __getitem__(self, index):
        if self.depths is None:
            return self.X[index], self.masks[index]
        else:
            return self.X[index], self.masks[index], self.depths[index]
        
    def __len__(self):
        return len(self.X)

class GraphDataset_Without_CoT(Dataset):
    
    def __init__(self, n_states, file_name):
        # Create a list of vocab
        number_tokens = sorted([str(i) for i in range(n_states)], key=lambda x: len(x), reverse=True)
        self.n_states = n_states
        self.idx2tokens = [",", ":", "|", "Y", "N"] + [f">{t}" for t in number_tokens] + number_tokens
        self.tokens2idx = {token: idx for idx, token in enumerate(self.idx2tokens)}
        #self.max_seq_length = n_states * 4 + 2
        self.max_seq_length = (n_states-2) * 3 + 4
        self.pad_token = self.tokens2idx[","]
        self.start_token = 1
        # Open up dataset file and load+tokenize strings
        self.X = []
        self.masks = []
        
        with open(file_name, "r") as f:
            for line in f.readlines():
                # Tokenize string
                tokens = self.tokenize(line.rstrip())
                self.X.append(tokens)
                # Run checks
                assert self.untokenize(self.X[-1]) == line.rstrip()
                assert len(self.X[-1]) <= self.max_seq_length
                # Create a mask
                start_idx = np.where(tokens == 1)[0].item()
                index_tensor = np.arange(tokens.shape[0])
                mask = np.zeros_like(tokens, dtype=bool)
                mask[start_idx+2:len(tokens)+2] = True
                self.masks.append(mask)
        # Stack arrays
        self.X = torch.from_numpy(np.stack(self.X))
        self.masks = torch.from_numpy(np.stack(self.masks))
        
    def tokenize(self, text):
        # Convert to token list
        tokens = []
        i = 0
        while i < len(text):
            for idx, word in enumerate(self.idx2tokens):
                if text.startswith(word, i):
                    tokens.append(idx)
                    i += len(word)
                    break
            else:
                i += 1
        # Convert to fixed length numpy array
        tokens_arr = np.array(tokens)
        padding_length = self.max_seq_length - len(tokens)
        tokens_arr = np.pad(tokens_arr, ((0, padding_length),), mode='constant', constant_values=0)
        return tokens_arr

    def untokenize(self, tokens):
        substrings = [self.idx2tokens[idx] for idx in tokens]
        return "".join(substrings).rstrip(",")
    
    def visualize_example(self, index):
        string = self.untokenize(self[index][0])
        parse_example(string)

    def __getitem__(self, index):
        return self.X[index], self.masks[index]

    def __len__(self):
        return len(self.X)
