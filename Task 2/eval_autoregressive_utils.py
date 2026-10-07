
import torch
import numpy as np

from tqdm import tqdm
from torch.nn import functional as F

def generate(model, prompt, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = model(prompt)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            #idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = torch.argmax(probs, dim=1,keepdim=True)
            # append sampled index to the running sequence and continue
            prompt = torch.cat((prompt, idx_next), dim=1)

        return prompt

def results_accuracy(preds_idx, target, sequences,wrong_seq_preds=None):
    num_token_correct = torch.sum(preds_idx == target)
    token_accuracy = num_token_correct.item() / len(target)
    seq_preds_idx = preds_idx.view(sequences, -1)
    seq_targets = target.view(sequences, -1)
    seq_len = seq_targets.size(1)
    seq_token_correct = torch.sum(seq_preds_idx == seq_targets, 1)
    num_seq_correct = (seq_token_correct == seq_len).sum()
    seq_accuracy = num_seq_correct / sequences
    #seq_accuracy = (seq_token_correct == seq_len).sum() / sequences

    if wrong_seq_preds is not None and num_seq_correct<sequences:
        for i in range(sequences):
            if seq_token_correct[i] < seq_len:
                wrong_seq_preds.append(i)

    return token_accuracy, seq_accuracy

def eval_accuracy_autoregressive(model,data_loader,eval_batch_size,wrong_sequences=None):
    token_accuracy = 0
    seq_accuracy = 0
    iterations = 0

    max_in_seq_length = None
    max_out_seq_length = None
    prompt = None

    pbar = tqdm(total=len(data_loader))
    for _, (tokens, mask) in enumerate(data_loader):
        if prompt is None:
            max_in_seq_length = torch.nonzero(mask[0], as_tuple=True)[0][0].item()
            max_out_seq_length = mask.size(1) - max_in_seq_length
            prompt = torch.full((eval_batch_size, max_in_seq_length), 0)
            prompt = prompt.to("cuda")

        prompt[:tokens.size(0),:max_in_seq_length] = tokens[:,:max_in_seq_length]
        generated = generate(model,prompt[:tokens.size(0)], max_out_seq_length)
        output = generated[:,max_in_seq_length:].contiguous().view(-1)
        target = tokens[:,max_in_seq_length:].contiguous().view(-1)
        output = output.to("cuda")
        target = target.to("cuda")
        if wrong_sequences is not None:
            wrong_seq_preds = []
        else:
            wrong_seq_preds = None
        it_token_accuracy, it_seq_accuracy = results_accuracy(output, target, tokens.size(0),wrong_seq_preds)
        token_accuracy += it_token_accuracy*tokens.size(0)
        seq_accuracy += it_seq_accuracy*tokens.size(0)
        iterations += tokens.size(0)

        if wrong_sequences is not None and len(wrong_seq_preds)>0:
           for i in range(len(wrong_seq_preds)):
               wrong_sequences.append(tokens[wrong_seq_preds[i], :].cpu())

        pbar.set_description(f"Token% {token_accuracy/iterations*100:4.2f}, Sequence% {seq_accuracy/iterations*100:4.2f}")
        pbar.update(1)

    pbar.close()

    return token_accuracy/iterations, seq_accuracy/iterations

def eval_accuracy_autoregressive_with_depth_info(model,data_loader,eval_batch_size):
    token_accuracy = 0
    seq_accuracy = 0
    iterations = 0

    max_in_seq_length = None
    max_out_seq_length = None
    prompt = None

    depth_seq_correct_count = {}
    depth_seq_count = {}
    for depth in range(1,16):
        id= str(depth)
        depth_seq_correct_count[id] = 0
        depth_seq_count[id] = 0

    pbar = tqdm(total=len(data_loader))
    for _, (tokens, masks,depths) in enumerate(data_loader):
        if prompt is None:
            max_in_seq_length = torch.nonzero(masks[0], as_tuple=True)[0][0].item()
            max_out_seq_length = masks.size(1) - max_in_seq_length
            prompt = torch.full((eval_batch_size, max_in_seq_length), 0)
            prompt = prompt.to("cuda")

        prompt[:tokens.size(0),:max_in_seq_length] = tokens[:,:max_in_seq_length]
        generated = generate(model,prompt[:tokens.size(0)], max_out_seq_length)
        output = generated[:,max_in_seq_length:].contiguous().view(-1)
        target = tokens[:,max_in_seq_length:].contiguous().view(-1)
        output = output.to("cuda")
        target = target.to("cuda")
        it_token_accuracy, it_seq_accuracy = results_accuracy(output, target, tokens.size(0))
        token_accuracy += it_token_accuracy*tokens.size(0)
        seq_accuracy += it_seq_accuracy*tokens.size(0)
        iterations += tokens.size(0)

        seq_preds_idx = output.view(tokens.size(0), -1)
        seq_targets = target.view(tokens.size(0), -1)
        seq_len = seq_targets.size(1)
        seq_token_correct = torch.sum(seq_preds_idx == seq_targets, 1)
        seq_correct = seq_token_correct == seq_len

        for i, depth in enumerate(depths):
            depth_seq_correct_count[depth] += seq_correct[i].item()
            depth_seq_count[depth] += 1

        pbar.set_description(f"Token% {token_accuracy/iterations*100:4.2f}, Sequence% {seq_accuracy/iterations*100:4.2f}")
        pbar.update(1)

    pbar.close()

    return token_accuracy/iterations, seq_accuracy/iterations, depth_seq_correct_count, depth_seq_count

def eval_token_accuracy_by_depth(model,data_loader):
    max_in_seq_length = None
    
    token_dist_from_leaf = np.zeros((15,))
    token_dist_from_leaf_count = np.zeros((15,))

    pbar = tqdm(total=len(data_loader))
    for _, (tokens, masks,depths) in enumerate(data_loader):
        if max_in_seq_length is None:
            max_in_seq_length = torch.nonzero(masks[0], as_tuple=True)[0][0].item()

        tokens = tokens.cuda().to(torch.long)
        inputs = tokens[:, :-1]
        output_mask = masks[:, 1:].cuda()
        targets = tokens[:, 1:][output_mask]

        output_logits = model(inputs)[output_mask]
        outputs = output_logits.argmax(-1)
        #outputs = torch.zeros_like(targets)

        outputs = outputs.view(tokens.size(0), -1)
        targets = targets.view(tokens.size(0), -1)

        comparison = outputs == targets
        
        for i, depth in enumerate(depths):
            depth_int = int(depth)
            for j in range(0,depth_int):
                if comparison[i][j].item():
                    token_dist_from_leaf[depth_int-j-1]+=1
                token_dist_from_leaf_count[depth_int-j-1]+=1

        pbar.update(1)

    pbar.close()

    return token_dist_from_leaf, token_dist_from_leaf_count