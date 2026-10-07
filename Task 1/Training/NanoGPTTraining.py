
import os
import math
import numpy as np
import torch

from model import GPTConfig, GPT
from logic_data import create_dics, create_dataset, random_split_dataset, shuffle_in_unison
from EnvUtilities import setup_environment

def results_accuracy(preds_idx, target, sequences):
    num_token_correct = torch.sum(preds_idx == target)
    token_accuracy = num_token_correct.item() / len(target)
    seq_preds_idx = preds_idx.view(sequences, -1)
    seq_targets = target.view(sequences, -1)
    seq_len = seq_targets.size(1)
    seq_token_correct = torch.sum(seq_preds_idx == seq_targets, 1) 
    seq_accuracy = (seq_token_correct == seq_len).sum() / sequences
    return token_accuracy, seq_accuracy

def eval_accuracy_autoregressive(data, eval_batch_size,max_in_seq_length,max_out_seq_length,model,device):
    data_size = data.size(0)
    assert(data_size % eval_batch_size == 0)
    iterations = data_size // eval_batch_size
    token_accuracy = 0
    seq_accuracy = 0

    prompt = torch.full((eval_batch_size, max_in_seq_length+1), 0)   # token 0 = <start of sequence>
    prompt = prompt.to(device)
    
    for iter in range(iterations):
        prompt[:,:max_in_seq_length+1] = data[iter*eval_batch_size : (iter+1)*eval_batch_size,:max_in_seq_length+1]    # prefix + <start of sequence>
        generated = model.generate(prompt, max_out_seq_length-1, temperature=1.0, top_k=1)
        output = generated[:,max_in_seq_length+1:].contiguous().view(-1)
        target = data[iter*eval_batch_size : (iter+1)*eval_batch_size,max_in_seq_length+1:].contiguous().view(-1)
        it_token_accuracy, it_seq_accuracy = results_accuracy(output, target, eval_batch_size)
        token_accuracy += it_token_accuracy
        seq_accuracy += it_seq_accuracy
    return token_accuracy/iterations, seq_accuracy/iterations

# learning rate decay scheduler (cosine with warmup)
def get_lr(it,learning_rate,warmup_iters,lr_decay_iters,min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def nanogpt_training(run_count, epochs, mb_size, out_folder_path = None, dataset_config = None, model_config = None):
    
    #Env params
    use_gpu = True
    init_seed = 127
    #---

    if dataset_config is None:
        # Default dataset config params
        literals = 6
        alphabet = 20
        dataset_size = 4096
        validation_perc = 0.25
        cot=True
        positive_sample_with_fixed_length = False
        negative_sample_with_new_literal = True
        up_lo_case_equal =False
    else:
        literals = dataset_config['literals']
        alphabet = dataset_config['alphabet']
        dataset_size = dataset_config['dataset_size']
        validation_perc = dataset_config['validation_perc']
        cot=dataset_config['cot']
        positive_sample_with_fixed_length = False
        negative_sample_with_new_literal = True
        up_lo_case_equal =False

    if model_config is None:
        # Default model config params
        n_layer = 6
        n_head = 8
        n_embd = 64
        dropout = 0.1
        bias = False # do we use bias inside LayerNorm and Linear layers?
        use_mlp = True
        use_flash_attention = False # False default (use flash attention or not, requires PyTorch >= 2.0)
    else:
        n_layer = model_config['n_layer']
        n_head = model_config['n_head']
        n_embd = model_config['n_embd']
        dropout = model_config['dropout']
        bias = model_config['bias']
        use_mlp = model_config['use_mlp']
        use_flash_attention = False # False default (use flash attention or not, requires PyTorch >= 2.0)

    # Optimizer params
    learning_rate = 1e-3 # with baby networks can afford to go a bit higher # 1e-3 ?  (con 5e-3 va cmq)
    # learning rate decay settings
    decay_lr = False # whether to decay the learning rate
    lr_decay_iters = 2000 # make equal to max_iters usually
    warmup_iters = 100 # not super necessary potentially
    min_lr = 1e-4 # learning_rate / 10 usually

    beta1 = 0.9
    beta2 = 0.98 # make a bit bigger because number of tokens per iter is small  # 0.99 ?
    weight_decay = 1e-1
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    #---

    env = setup_environment(use_gpu = use_gpu, init_seed = init_seed)
    device = env['device']

    my_vocab, stoi, _ = create_dics(alphabet,up_lo_case_equal)
        
    src_data, tgt_data,_,_=create_dataset(literals,
                                        alphabet,
                                        dataset_size,
                                        positive_sample_with_fixed_length,
                                        negative_sample_with_new_literal,
                                        stoi,
                                        cot=cot,
                                        up_lo_case_equal=up_lo_case_equal)
    
    max_in_seq_length = src_data.size(1)
    max_out_seq_length = tgt_data.size(1)

    src_data_train, tgt_data_train, src_data_val, tgt_data_val = random_split_dataset(src_data, tgt_data,validation_perc=validation_perc, init_seed=None)
    
    # concatenate src_data_train and tgt_data_train in a single np array along axis 1
    train_data = np.concatenate((src_data_train, tgt_data_train), axis=1)
    val_data = np.concatenate((src_data_val, tgt_data_val), axis=1)

    # move datasets to device
    train_data = torch.from_numpy(train_data).to(device)
    val_data = torch.from_numpy(val_data).to(device)

    train_size = train_data.size(0)
    iterations = train_size // mb_size

    run_loss=np.empty((run_count,epochs))
    run_train_seq_acc=np.empty((run_count,epochs))
    run_val_seq_acc=np.empty((run_count,epochs))

    out_file_name='{0}Pos_{1}Neg_{2}CoT_{3}UlC_DB{4}_ValP{5}_mb{6}_Ep{7}_AtL{8}_He{9}_Emb{10}_{11}MLP_{12}FlAt_InitS{13}'.format('FixLen' if positive_sample_with_fixed_length else 'VarLen',
                                                                                    'NewLit' if negative_sample_with_new_literal else 'InvIm',
                                                                                    '' if cot else 'No',
                                                                                    '' if up_lo_case_equal else 'No',
                                                                                    dataset_size,
                                                                                    int(validation_perc*100),
                                                                                    mb_size,
                                                                                    epochs,
                                                                                    n_layer,
                                                                                    n_head,
                                                                                    n_embd,
                                                                                    '' if use_mlp else 'No',
                                                                                    '' if use_flash_attention else 'No',
                                                                                    init_seed)

    for run in range(run_count):
        

        # model init (from scratch)
        model_args = dict(n_layer=n_layer,
                            n_head=n_head,
                            n_embd=n_embd,
                            block_size=max_in_seq_length + max_out_seq_length,
                            bias=bias,
                            vocab_size=my_vocab,
                            dropout=dropout,
                            use_mlp=use_mlp,
                            use_flash_attention=use_flash_attention)   
        
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.to(device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(env['dtype'] == 'float16'))

        # optimizer
        optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), env['device_type'])
        checkpoint = None # free up memory

        # compile the model
        if env['compile']:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0 (not yet for Windows)

        print(f"RUN: {run+1}")
        for epoch in range(epochs):
            ep_train_data = shuffle_in_unison((train_data,))[0]
            ep_loss=0

            model.train()
            for iter in range(iterations):
                # current batch
                it_data = ep_train_data[iter*mb_size : (iter+1)*mb_size]
                X = it_data[:,:-1]
                Y = it_data[:,1:].to(torch.long)   # Y is shifted right by one compared to X
                Y[:,:max_in_seq_length] = -1    # the prefix part is not a target (and does not contribute to loss), so we mask it. 
                                                # if prefix is not masked, much slower convergence 
                
                # determine and set the learning rate for this iteration
                global_iter = epoch * iterations + iter
                lr = get_lr(global_iter,learning_rate,warmup_iters,lr_decay_iters,min_lr) if decay_lr else learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # forward backward update and using the GradScaler if data type is float16
                # under autocast only the forward pass; backward pass uses the same time defined in forward pass
                with env['ctx']:
                    logits, loss = model(X, Y)
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
                # clip the gradient
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
                ep_loss+=loss
                print(".",end="")
                
            ep_loss/=iterations

            model.eval()
            with env['ctx']:
                train_token_acc, train_seq_acc = eval_accuracy_autoregressive(ep_train_data, mb_size, max_in_seq_length,max_out_seq_length,model,device)
                token_val_acc, seq_val_acc = eval_accuracy_autoregressive(val_data, mb_size, max_in_seq_length,max_out_seq_length,model,device)

            run_loss[run,epoch]=ep_loss
            run_train_seq_acc[run,epoch]=train_seq_acc
            run_val_seq_acc[run,epoch]=seq_val_acc

            print("")
            print(f"Run {run+1} Epoch {epoch+1}, Train Loss {ep_loss:5.4f}")
            print(f"TRAIN Token% {train_token_acc*100:4.2f}, Sequence% {train_seq_acc*100:4.2f}")
            print(f"VALID Token% {token_val_acc*100:4.2f}, Sequence% {seq_val_acc*100:4.2f}")

        if out_folder_path is not None and os.path.exists(out_folder_path):
            out_run_file_name='{0}{1}'.format(out_file_name,
                                              '_r{0}'.format(run+1) if run_count > 1 else '')

            checkpoint = { 'model': model.state_dict(),
                           'model_args': model_args 
                        }
            torch.save(checkpoint, os.path.join(out_folder_path,'{0}.mdl'.format(out_run_file_name)))

    return run_train_seq_acc,run_val_seq_acc