import torch
import random
import string

# select V distinct random letters from the first A letters of the alphabet
def select_literals(V, A):
    if V > A:
        raise ValueError("V cannot be greater than A as you cannot select more distinct letters than available.")
    if A > 26:
        raise ValueError("A cannot be greater than 26 as there are only 26 uppercase letters.")
    return random.sample(string.ascii_uppercase[:A], V)

def literals_to_implications(input_list):
    if len(input_list) < 2:
        return []
    return [f"{input_list[i]}>{input_list[i+1]}" for i in range(len(input_list) - 1)]

def shuffle_implications(input_list):
    shuffled_list = input_list[:]
    random.shuffle(shuffled_list)
    return shuffled_list

def exchange_literals_in_implication(s):
    return s[-1] + s[1:-1] + s[0]

def replace_literal_in_implication(implication, new_literal, replace_head=True):
    if replace_head:
        return new_literal + implication[1:]
    else:
        return implication[:-1] + new_literal

def generate_fixed_length_positive_sample(literals, alphabet, cot=True, up_lo_case_equal=False):
    literals = select_literals(literals, alphabet)
    implications = literals_to_implications(literals)
    
    if up_lo_case_equal:
        implications_for_prompt = [imp.lower() if random.choice([True, False]) else imp for imp in implications]
    else:
        implications_for_prompt = implications

    # In a positive sample, the horn clauses allows to reach the last literal from the first one (from a single path)
    # Literals: ['C', 'M', 'L', 'F']  Implications: ['C>M', 'M>L', 'L>F']
    # Prompt: 'C>M,L>F,M>L|C>F'  Response: 'C>M,M>L,L>F-1'    
    shuffled_implications_for_prompt = shuffle_implications(implications_for_prompt)
    positive_prompt = ','.join(shuffled_implications_for_prompt)+'|'+literals[0]+'>'+literals[-1]
    
    if cot:
        positive_response = ','.join(implications)+'-1'
    else:
        positive_response = '1'
    
    return positive_prompt, positive_response

def generate_variable_length_positive_sample(literals, alphabet, cot=True, up_lo_case_equal=False):
    underscore_pos = 0  # initialize to a value that forces the while loop to run at least once
    while underscore_pos == 0:  # chain break point is at the beginning of the response
        negative_prompt, negative_response=generate_negative_sample_with_new_literal(literals, alphabet, cot=True, up_lo_case_equal=up_lo_case_equal)
        underscore_pos = negative_response.find('_')
    if underscore_pos == -1:  # chain break point is at the end of the response, there's no underscore
        underscore_pos = len(negative_response)-1
    new_literal = negative_response[underscore_pos-2]
    positive_prompt = negative_prompt[:-1] + new_literal 
    if cot:
        positive_response = negative_response[:-1] + '1'  
    else:
        positive_response = '1'  
    return positive_prompt, positive_response

def generate_negative_sample_with_inverted_implication(literals, alphabet, cot=True, up_lo_case_equal=False):
    literals = select_literals(literals, alphabet)
    implications = literals_to_implications(literals)
    
    if up_lo_case_equal:
        implications_for_prompt = [imp.lower() if random.choice([True, False]) else imp for imp in implications]
    else:
        implications_for_prompt = implications

    # In a negative sample, one of the implication is inverted to prevent reaching the last literal from the first one 
    # Literals:['A', 'H', 'G', 'B']  Implications:  ['A>H', 'H>G', 'G>B']
    # Prompt: 'A>H,B>G,H>G|A>B'   Response: 'A>H,H>G,___-0'
    chain_break_point = random.randint(0, len(implications_for_prompt)-1)
    broken_implications_for_prompt = implications_for_prompt.copy()
    broken_implications_for_prompt[chain_break_point] = exchange_literals_in_implication(broken_implications_for_prompt[chain_break_point])
    shuffled_implications_for_prompt=shuffle_implications(broken_implications_for_prompt)
    negative_prompt = ','.join(shuffled_implications_for_prompt)+'|'+literals[0]+'>'+literals[-1]
    
    if cot:
        partial_chain = ','.join(implications[:chain_break_point])
        n_padding = len(implications)-chain_break_point
        if chain_break_point > 0: partial_chain += ','
        partial_chain += ','.join(['___' for i in range(n_padding)])   # add n_padding strings of the form '___' to the partial chain
        negative_response = partial_chain+'-0'
    else:
        negative_response = '0'

    return negative_prompt, negative_response

def generate_negative_sample_with_new_literal(literals, alphabet, cot=True, up_lo_case_equal=False):
    literals = select_literals(literals+1, alphabet)
    new_literal = literals[-1]  # last literal is not used in the implications
    literals = literals[:-1]  # last literal is not used in the implications
    implications = literals_to_implications(literals)  # last literal is not used in the implications
    
    if up_lo_case_equal:
        implications_for_prompt = [imp.lower() if random.choice([True, False]) else imp for imp in implications]
        new_literal_for_prompt = new_literal.lower() if random.choice([True, False]) else new_literal
    else:
        implications_for_prompt = implications
        new_literal_for_prompt = new_literal

    chain_break_point = random.randint(0, len(implications_for_prompt)-1)
    replace_head = random.choice([True, False])
    broken_implications_for_prompt = implications_for_prompt.copy()
    broken_implications_for_prompt[chain_break_point] = replace_literal_in_implication(broken_implications_for_prompt[chain_break_point], new_literal_for_prompt, replace_head)
    shuffled_implications_for_prompt=shuffle_implications(broken_implications_for_prompt)
    negative_prompt = ','.join(shuffled_implications_for_prompt)+'|'+literals[0]+'>'+literals[-1]
    
    if cot:
        if replace_head:
            partial_chain = ','.join(implications[:chain_break_point])
            n_padding = len(implications)-chain_break_point
            if chain_break_point > 0: partial_chain += ','
        else:
            partial_chain = ','.join(implications[:chain_break_point+1])
            partial_chain = partial_chain[:-1] + new_literal
            n_padding = len(implications)-chain_break_point-1
            if chain_break_point < len(implications)-1: partial_chain += ','
        partial_chain += ','.join(['___' for i in range(n_padding)])   # add n_padding strings of the form '___' to the partial chain
        negative_response = partial_chain+'-0'
    else:
        negative_response = '0'

    return negative_prompt, negative_response

def generate_sample(literals, alphabet, positive,positive_sample_with_fixed_length,negative_sample_with_new_literal, cot, up_lo_case_equal):
    if positive:
        if positive_sample_with_fixed_length:
            return generate_fixed_length_positive_sample(literals, alphabet, cot, up_lo_case_equal)
        else:
            return generate_variable_length_positive_sample(literals, alphabet, cot, up_lo_case_equal)
    else:
        if not negative_sample_with_new_literal:
            return generate_negative_sample_with_inverted_implication(literals, alphabet, cot, up_lo_case_equal)
        else:
            return generate_negative_sample_with_new_literal(literals, alphabet, cot, up_lo_case_equal)

def shuffle_in_unison(dataset,init_seed=None):
    new_dataset = []
    len_dataset = dataset[0].size(0)
    if init_seed is not None:
        torch.manual_seed(init_seed)
    rand_indx = torch.randperm(len_dataset)
    for x in dataset:
        new_dataset.append(x[rand_indx])
    return new_dataset

# dictionary creation, encoding and deconding functions
#   Note: all tokens are single chars
def create_dics(alphabet,up_lo_case_equal=False):
    special_chars = '@>,|_-01'   # '@' is start of string token: it MUST be the first character (idx = 0)
    literals_chars = string.ascii_uppercase[:alphabet]

    if up_lo_case_equal:
        # if upper and lower case letters are considered equal, we add the lowercase letters to the dictionary
        literals_chars += string.ascii_lowercase[:alphabet]

    all_chars = special_chars + literals_chars
    stoi = { ch:i for i,ch in enumerate(all_chars) }
    itos = { i:ch for i,ch in enumerate(all_chars) }
    vocab_size = len(stoi)
    return vocab_size, stoi, itos

def encode(s,stoi):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l,itos):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

def random_split_dataset(src_data,tgt_data,validation_perc=0.25,init_seed=None):
    dataset_size=src_data.size(0)
    validation_size = int(dataset_size * validation_perc)
    training_size = dataset_size - validation_size
    src_data, tgt_data = shuffle_in_unison((src_data, tgt_data),init_seed)
    src_data_train = src_data[0:training_size]
    tgt_data_train = tgt_data[0:training_size]
    src_data_val = src_data[training_size:]
    tgt_data_val = tgt_data[training_size:]
    return src_data_train, tgt_data_train, src_data_val, tgt_data_val

def create_dataset(literals = 4, alphabet = 15, dataset_size = 10, positive_sample_with_fixed_length=True, negative_sample_with_new_literal=False, encfunc = None, cot=True,up_lo_case_equal=False):

    # if (init_seed is not None):
    #     random.seed(init_seed)
        
    input_seq_length = 4*literals - 1       # C>M,L>F,M>L|C>F

    if cot:
        output_seq_length = 4*(literals-1) + 2  # @C>M,M>L,L>F-1      add prefix @ as start of string token
    else:
        output_seq_length = 2                   # @1      add prefix @ as start of string token
    
    # sting representation of the dataset
    tset_input_str = []
    tset_output_str = []
    unique_prompts = set()

    # torch arrays of integers
    src_data = torch.full((dataset_size, input_seq_length), 0, dtype = torch.int32)   
    tgt_data = torch.full((dataset_size, output_seq_length), 0, dtype = torch.int32)   
    # generate dataset
    idx = 0
    collisions = 0
    for is_positive_sample in [True, False]:
        for i in range(dataset_size//2):
            prompt, response = generate_sample(literals,
                                                alphabet,
                                                positive=is_positive_sample,
                                                positive_sample_with_fixed_length=positive_sample_with_fixed_length,
                                                negative_sample_with_new_literal=negative_sample_with_new_literal,
                                                cot=cot,
                                                up_lo_case_equal=up_lo_case_equal)
            while prompt in unique_prompts:
                prompt, response = generate_sample(literals,
                                                   alphabet,
                                                   positive=is_positive_sample,
                                                   positive_sample_with_fixed_length=positive_sample_with_fixed_length,
                                                   negative_sample_with_new_literal=negative_sample_with_new_literal,
                                                   cot=cot,
                                                   up_lo_case_equal=up_lo_case_equal)
                collisions += 1
            unique_prompts.add(prompt)                
            tset_input_str.append(prompt)
            tset_output_str.append('@'+response)
            if encfunc is not None:
                src_data[idx] = torch.tensor(encode(prompt,encfunc))
                tgt_data[idx] = torch.tensor(encode('@'+response,encfunc))
            idx += 1
    print('Dataset created, size:',src_data.size(0), ', collisions:' ,collisions)
    return src_data,tgt_data,tset_input_str,tset_output_str

def print_sample_dataset(ds_len = 10):
    _,_,tset_in_str,tset_out_str = create_dataset(6, 20, ds_len, negative_sample_with_new_literal=True, encfunc = None, init_seed = 1234, cot=True, up_lo_case_equal=False)
    for i in range(len(tset_in_str)):
        print(tset_in_str[i], ' -> ', tset_out_str[i])

