
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.nn import functional as F
from model_with_hooks import GPTConfig, GPT
from logic_data import encode, decode, decode_multi

# FUNCTIONS

def load_model(model_checkpoint_file_path, device):
    print("Loading model from: ", model_checkpoint_file_path)
    checkpoint = torch.load(model_checkpoint_file_path)
    state_dict = checkpoint['model']
    checkpoint_model_args = checkpoint['model_args']
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, gptconf

def generate(model, literals,cot, stoi, itos, prompt_str, device, ctx):
    prompt_length = 4*literals    # C>M,L>F,M>L|C>F@      add prefix @ as start of string token
    
    if cot:
        output_length = 4*(literals-1) + 1  # C>M,M>L,L>F-1     
    else:
        output_length = 1  # 1     

    # prepare prompt
    prompts = torch.full((1, prompt_length), 0)      # minibatch of size 1
    prompts = prompts.to(device)
    prompts[0] = torch.tensor(encode(prompt_str,stoi))

    # run generation
    model.eval()
    with torch.no_grad():
        with ctx:
            y = model.generate(prompts, output_length, temperature=1, top_k=1)
            return decode(y[0].tolist(),itos)

def to_tensor_prompts(prompt_str, stoi, device):
    prompt_length = len(prompt_str) 
    prompts = torch.full((1, prompt_length), 0)      # minibatch of size 1
    prompts = prompts.to(device)
    prompts[0] = torch.tensor(encode(prompt_str,stoi))

    return prompts

def add_out_to_prompt(prompts,logits):
    temperature=1
    top_k=1

    logits = logits[0][:, -1, :] / temperature
    # optionally crop the logits to only the top k options
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1)
    # append sampled index to the running sequence and continue
    out = torch.cat((prompts, idx_next), dim=1)

    return out

def to_str_tokens(str,separator_str="__"):
    str_tokens=[]
    for i in range(len(str)):
        str_tokens.append(separator_str+str[i]+separator_str)
    return str_tokens

def create_outputs(output_letters, level_y, node_width=0.015, node_height=0.05, x_start=0.1, x_end=0.9):
    outputs = []
    num_outputs = len(output_letters)
    if num_outputs >1:
        x_positions = [x_start + i*((x_end - x_start)/(num_outputs - 1)) for i in range(num_outputs)]
    else:
        x_positions = [x_start]
    for letter, x in zip(output_letters, x_positions):
        outputs.append(
            {
            'x': x,
            'y': level_y,
            'width': node_width,
            'height': node_height,
            'label': letter
            })
    return outputs

def draw_outputs(ax, outputs):
    for out in outputs:
        rect = patches.Rectangle((out['x'], out['y']),
                                out['width'],
                                out['height'],
                                facecolor="skyblue",
                                edgecolor="black",
                                zorder=10)
        ax.add_patch(rect)
        ax.text(out['x'] + out['width'] / 2,
                out['y'] + out['height'] / 2,
                out['label'],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
                zorder=11)


def create_nodes(levels, node_letters, level_y,node_subs=None, node_width=0.015, node_height=0.05, x_start=0.1, x_end=0.9):
    """
    Creates nodes for each level.
    Each node is keyed as 'Letter_level' and stores its position and size.
    """
    nodes = []
    num_nodes = len(node_letters)
    x_positions = [x_start + i*((x_end - x_start)/(num_nodes - 1)) for i in range(num_nodes)]
    for level in levels:
        level_nodes=[]
        for i in range(num_nodes):
            sub=None
            height= node_height
            if node_subs is not None and level in node_subs:
                if isinstance(node_subs[level][i], str):
                    sub = [s for s in node_subs[level][i]]
                elif isinstance(node_subs[level][i], list):
                    sub = node_subs[level][i]
                height=node_height*(len(sub)+1)

            level_nodes.append(
                {
                'x': x_positions[i],
                'y': level_y[level],
                'width': node_width,
                'height': height,
                'label': node_letters[i],
                'sub' : sub,
                })
        nodes.append(level_nodes)
    return nodes

def draw_nodes(ax, nodes):
    """
    Draws nodes on a given axis.
    """
    for level_nodes in nodes:
        for node in level_nodes:
            rect = patches.Rectangle((node['x'], node['y']),
                                    node['width'],
                                    node['height'],
                                    facecolor="skyblue",
                                    edgecolor="black",
                                    zorder=10)
            ax.add_patch(rect)
            if node['sub'] is None:
                ax.text(node['x'] + node['width'] / 2,
                        node['y'] + node['height'] / 2,
                        node['label'],
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=12,
                        zorder=11)
            else:
                sub_count= len(node['sub'])
                ax.text(node['x'] + node['width'] / 2,
                        node['y'] + node['height'] *(sub_count+1)/ (sub_count+2),
                        node['label'],
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=12,
                        zorder=11)
                for i in range(sub_count):
                    ax.text(node['x'] + node['width'] / 2,
                            node['y'] + node['height'] *(sub_count-i)/ (sub_count+2),
                            node['sub'][i],
                            horizontalalignment="center",
                            verticalalignment="center",
                            fontsize=8,
                            zorder=11)

def draw_flows(ax, nodes,source_level,target_level, sources,targets,values,w_thr=0,target_flows_excluded=None, line_width_multiplier=6):
    """
    Draws flows as arrows. For vertical flows, the arrow goes from the top center of
    the source node to the bottom center of the destination node.
    """
    for i in range(len(sources)):
        src = sources[i]
        tgt = targets[i]
        weight = values[i]
        if target_flows_excluded is None or tgt not in target_flows_excluded:
            if weight >= w_thr:
                src_node = nodes[source_level][src]
                tgt_node = nodes[target_level][tgt]
                start_x = src_node['x'] + src_node['width'] / 2
                start_y = src_node['y'] + src_node['height']  # top of source node
                end_x = tgt_node['x'] + tgt_node['width'] / 2
                end_y = tgt_node['y']  # bottom of target node
                ax.annotate("",
                            xy=(end_x, end_y), xycoords='data',
                            xytext=(start_x, start_y), textcoords='data',
                            arrowprops=dict(arrowstyle="-",
                                            color="gray",
                                            lw=weight * line_width_multiplier,
                                            connectionstyle="arc3,rad=0.0"))

def draw_flows_with_arrow(ax, nodes, source_level, target_level, sources, targets, values, w_thr=0, target_flows_excluded=None, line_width_multiplier=6, alpha=0.7):
    """
    Draws flows as arrows using ax.arrow with transparency and assigns different colors to each arrow.
    For vertical flows, the arrow goes from the top center of the source node to the bottom center of the destination node.

    Parameters:
        alpha (float): Transparency value for the arrows (0.0 is fully transparent, 1.0 is opaque).
    """
    # Extended list of colors
    colors = [
        'red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 
        'brown', 'pink', 'olive', 'navy', 'teal', 'gold', 'indigo', 'coral', 
        'lavender', 'turquoise', 'salmon', 'chocolate'
    ]
    
    for i in range(len(sources)):
        src = sources[i]
        tgt = targets[i]
        weight = values[i]
        if target_flows_excluded is None or tgt not in target_flows_excluded:
            if weight >= w_thr:
                src_node = nodes[source_level][src]
                tgt_node = nodes[target_level][tgt]
                start_x = src_node['x'] + src_node['width'] / 2
                start_y = src_node['y'] + src_node['height']  # top of source node
                end_x = tgt_node['x'] + tgt_node['width'] / 2
                end_y = tgt_node['y']  # bottom of target node

                dx = end_x - start_x
                dy = end_y - start_y

                # Cycle through the extended colors list.
                color = colors[i % len(colors)]
                
                ax.arrow(start_x, start_y, dx, dy,
                         length_includes_head=True,
                         head_width=0.01,
                         head_length=0.01,
                         fc=color,
                         ec=color,
                         linewidth=weight * line_width_multiplier,
                         alpha=alpha)
                
def draw_curved_flows(ax, nodes, source_level, target_level, sources, targets, values,
                      w_thr=0, target_flows_excluded=None, line_width_multiplier=6, alpha=0.7, curvature=0.2):
    """
    Draws flows as curved arrows using ax.annotate to simulate Sankey diagram flows.
    For vertical flows, the arrow originates at the top center of the source node and
    curves toward the bottom center of the destination node.

    Parameters:
        curvature (float): Base magnitude for the curvature of each arrow; arrows alternate the curvature direction.
        alpha (float): Transparency for the arrows (0.0 is fully transparent, 1.0 is opaque).
    """
    # Extended list of colors
    colors = [
        'red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 
        'brown', 'pink', 'olive', 'navy', 'teal', 'gold', 'indigo', 'coral', 
        'lavender', 'turquoise', 'salmon', 'chocolate'
    ]
    
    for i in range(len(sources)):
        src = sources[i]
        tgt = targets[i]
        weight = values[i]
        if target_flows_excluded is None or tgt not in target_flows_excluded:
            if weight >= w_thr:
                src_node = nodes[source_level][src]
                tgt_node = nodes[target_level][tgt]
                start_x = src_node['x'] + src_node['width'] / 2
                start_y = src_node['y'] + src_node['height']  # top of source node
                end_x = tgt_node['x'] + tgt_node['width'] / 2
                end_y = tgt_node['y']  # bottom of target node

                # Alternate curvature direction for variety
                rad = curvature if i % 2 == 0 else -curvature

                ax.annotate("",
                            xy=(end_x, end_y), xycoords='data',
                            xytext=(start_x, start_y), textcoords='data',
                            arrowprops=dict(arrowstyle="->",
                                            color=colors[i % len(colors)],
                                            lw=weight * line_width_multiplier,
                                            alpha=alpha,
                                            connectionstyle=f"arc3,rad={rad}"))

def draw_lines(ax, nodes, source_level, target_level, sources, targets, values,source_k_letters=None,source_v_letters=None,target_q_letters=None,
               w_thr=0, target_flows_excluded=None, line_width_multiplier=6, alpha=0.7,gray_lines=True):
    # Extended list of colors
    colors = [
        'red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow',
        'brown', 'pink', 'olive', 'navy', 'teal', 'gold', 'indigo', 'coral',
        'lavender', 'turquoise', 'salmon', 'chocolate'
    ]
    
    for i in range(len(sources)):
        src = sources[i]
        tgt = targets[i]
        weight = values[i]
        if target_flows_excluded is None or tgt not in target_flows_excluded:
            if weight >= w_thr:
                src_node = nodes[source_level][src]
                tgt_node = nodes[target_level][tgt]
                start_x = src_node['x'] + src_node['width'] / 2
                start_y = src_node['y'] + src_node['height']  # top of source node
                end_x = tgt_node['x'] + tgt_node['width'] / 2
                end_y = tgt_node['y']  # bottom of target node

                # Cycle through the extended colors list.
                color = colors[i % len(colors)] if not gray_lines else 'gray'
                
                ax.plot([start_x, end_x], [start_y, end_y], color=color,
                        linewidth=weight * line_width_multiplier, alpha=alpha)
                
                if source_k_letters is not None and source_v_letters is not None and target_q_letters is not None:
                    # Draw the letters at the start and end of the line
                    ax.text(start_x, start_y + 0.01, f"{source_k_letters[src]} ; {source_v_letters[src]}", fontsize=10, ha='center', va='bottom', color='red')
                    ax.text(end_x, end_y - 0.03, target_q_letters[tgt], fontsize=10, ha='center', va='bottom', color='red')

def from_logits_to_chars(logits,itos):
    v, _ = torch.topk(logits, min(1, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    indices = torch.multinomial(probs, num_samples=1).flatten().tolist()
    chars=decode(indices,itos)

    return chars,indices

def from_logits_to_top_k_chars(logits,itos,top_k):
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1).cpu().numpy()

    sorted_indices = np.argsort(probs, axis=1)[:, ::-1]

    sel_indices = sorted_indices[:, :top_k]

    chars=decode_multi([sel_indices[row_idx] for row_idx in range(sorted_indices.shape[0])],itos)

    return chars,sel_indices

def compute_avg_attention_layer_patterns(n_head,n_layer,data,model,enc_func,dec_func,device):
    avg_attention_layer_patterns={}
    for i in range(n_layer):
        avg_attention_layer_patterns[i]=np.zeros((n_head,data.shape[1]-1, data.shape[1]-1))
        
    model.return_all_logits = True
    for i in range(data.shape[0]):
        prompt_str=decode(data[i,:-1],dec_func)
        prompt_tensor=to_tensor_prompts(prompt_str, enc_func, device)

        logits,cache=model.run_with_cache(prompt_tensor,remove_batch_dim=True)

        for j in range(n_layer):
            key = f"transformer.h.{j}.attn.hook_pattern"
            avg_attention_layer_patterns[j]=avg_attention_layer_patterns[j]+cache[key].cpu().numpy()

    for i in range(n_layer):
        avg_attention_layer_patterns[i]= avg_attention_layer_patterns[i] / data.shape[0]

    return avg_attention_layer_patterns

def prepare_data_flows_from_attention_matrices(n_layer,n_head,attention_matrices):
    attention_layer_sources={}
    attention_layer_targets={}
    attention_layer_values={}

    for i in range(n_layer):
        attention_layer_sources[i]={}
        attention_layer_targets[i]={}
        attention_layer_values[i]={}
        for j in range(n_head):
            attention_layer_sources[i][j]=[]
            attention_layer_targets[i][j]=[]
            attention_layer_values[i][j]=[]
            
            attention_matrix=attention_matrices[i][j]
            for r in range(attention_matrix.shape[0]):
                for c in range(r+1):
                    if attention_matrix[r][c] > 0:
                        attention_layer_targets[i][j].append(r)
                        attention_layer_sources[i][j].append(c)
                        attention_layer_values[i][j].append(attention_matrix[r][c])
    
    return attention_layer_sources, attention_layer_targets, attention_layer_values

def check_and_visualize_embedding_orthogonality(model, itos, prompt_str_h):
    """
    Computes and prints the mean and maximum absolute cosine similarity between the token embeddings (wte)
    and positional embeddings (wpe) and visualizes the full cosine similarity matrix.
    The y-axis is labeled using the itos mapping (tokens from the vocabulary) and the
    x-axis is labeled with the tokens (chars) from prompt_str_h.
    Low values indicate that the two embedding spaces span different subspaces.
    """
    with torch.no_grad():
        # Get the embedding matrices
        wte = model.transformer.wte.weight  # shape: (vocab_size, n_embd)
        wpe = model.transformer.wpe.weight    # shape: (block_size, n_embd)
        
        # Normalize each row to compute cosine similarities
        wte_norm = wte / (wte.norm(dim=1, keepdim=True) + 1e-8)
        wpe_norm = wpe / (wpe.norm(dim=1, keepdim=True) + 1e-8)
        
        # Compute the similarity matrix; shape: (vocab_size, block_size)
        cosine_sim = torch.matmul(wte_norm, wpe_norm.t())
        
        # Compute statistics
        mean_sim = cosine_sim.abs().mean().item()
        max_sim = cosine_sim.abs().max().item()
        
        print(f"Mean absolute cosine similarity: {mean_sim:.4f}")
        print(f"Max absolute cosine similarity: {max_sim:.4f}")
        
        # Prepare axis labels:
        # For y-axis, use tokens from the vocabulary (itos mapping)
        vocab_size = wte.shape[0]
        token_labels = [itos.get(i, str(i)) for i in range(vocab_size)]
        
        # For x-axis, use the tokens (chars) from prompt_str_h.
        # The positional embedding matrix has dim 'block_size'
        block_size = wpe.shape[0]
        prompt_tokens = list(prompt_str_h)
        # If prompt_str_h has fewer tokens than block_size, pad with empty strings;
        # if more, truncate.
        if len(prompt_tokens) < block_size:
            prompt_tokens += [''] * (block_size - len(prompt_tokens))
        else:
            prompt_tokens = prompt_tokens[:block_size]
        
        # Visualize the cosine similarity matrix
        plt.figure(figsize=(12, 8))
        im = plt.imshow(cosine_sim.cpu().numpy(), aspect='auto', cmap='viridis')
        plt.colorbar(im)
        plt.xlabel("Tokens from prompt")
        plt.ylabel("Token (from vocabulary)")
        plt.title("Cosine Similarity between Token and Positional Embeddings")
        plt.xticks(ticks=range(block_size), labels=prompt_tokens, rotation=90, fontsize=6)
        plt.yticks(ticks=range(vocab_size), labels=token_labels, fontsize=6)
        plt.show()

def truncated_pseudoinverse(W, thr_sk=0.9):
    """
    Computes a truncated pseudoinverse of a matrix W using its SVD.
    
    Only the minimal set of singular values whose cumulative sum is at least 
    thr_sk (e.g., 0.9 represents 90%) of the total sum are inverted.
    
    Parameters:
        W (np.ndarray): The input matrix.
        thr_sk (float): The cumulative percentage threshold (between 0 and 1).
    
    Returns:
        np.ndarray: The truncated pseudoinverse of W.
    """
    # Compute the SVD of W
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    
    # Compute cumulative sum and determine number of singular values to retain
    total_sum = np.sum(S)
    cum_sum = np.cumsum(S)
    n = np.searchsorted(cum_sum, thr_sk * total_sum) + 1  # +1 because searchsorted is zero-indexed
    
    # Create an inverse diagonal vector: invert only the top-n components
    S_inv_truncated = np.zeros_like(S)
    S_inv_truncated[:n] = 1 / S[:n]
    
    # Form the truncated pseudoinverse
    W_inv_truncated = Vt.T @ np.diag(S_inv_truncated) @ U.T
    return W_inv_truncated, n

def draw_sankey(labels,output_labels,l1_top_k_subs,attention_data,res_stream_data,all_label_indices,weight_thr,target_flows, gray_lines=True):
    node_width = 0.015
    
    # Define levels and node labels.
    levels = [0, 1, 2]
    node_letters = labels

    # Define vertical positions for each level.
    level_y = {0: 0.1, 1:0.45, 2:0.8}

    node_subs={}
    node_subs[1] = l1_top_k_subs

    # Create nodes.
    nodes = create_nodes(levels, node_letters, level_y,node_subs=node_subs,node_width=node_width)

    pipe_char_x_coord=nodes[0][19]['x']
    at_char_x_coord=nodes[0][23]['x']

    outputs = create_outputs(output_labels,0.9,x_start=at_char_x_coord)

    # Create a figure.
    fig = plt.figure(figsize=(25, 8))
    ax = fig.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    attention_sources, attention_targets, attention_values  = attention_data

    draw_lines(ax, nodes,0,1, attention_sources[0][0],attention_targets[0][0],attention_values[0][0],
               w_thr=weight_thr,target_flows_excluded=None,gray_lines=gray_lines)

    target_flows_excluded_12=[item for item in all_label_indices if item not in target_flows]

    if res_stream_data is not None:
        target_Q_letters_12,sources_K_letters_12,source_V_letters_12 = res_stream_data
    else:
        target_Q_letters_12,sources_K_letters_12,source_V_letters_12 =None,None,None

    draw_lines(ax, nodes,1,2, attention_sources[1][0],attention_targets[1][0],attention_values[1][0],
                source_k_letters=sources_K_letters_12,source_v_letters=source_V_letters_12,target_q_letters=target_Q_letters_12,
                w_thr=weight_thr,target_flows_excluded=target_flows_excluded_12, gray_lines=gray_lines, line_width_multiplier=6)

    # Draw a fixed dashed vertical line at x position corresponding to node 20
    ax.axvline(x=pipe_char_x_coord+node_width/2, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=at_char_x_coord+node_width/2, color='black', linestyle='--', linewidth=1)

    # Adda fixed (Rules) text at x = 0.23, y = 0.08
    ax.text((nodes[0][0]['x']+nodes[0][18]['x'])/2, 0.05, 'Given rules', fontsize=16, ha='center', va='center', color='black')
    ax.text((pipe_char_x_coord+at_char_x_coord)/2, 0.05, 'Query', fontsize=16, ha='center', va='center', color='black')
    ax.text((at_char_x_coord+nodes[0][-1]['x'])/2, 0.99, 'Generated output', fontsize=16, ha='center', va='center', color='black')

    # Draw nodes.
    draw_nodes(ax, nodes)

    draw_outputs(ax, outputs)

    plt.show()

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
    
    outputs=np.empty((data_size,max_out_seq_length-1),dtype=int)
    for iter in range(iterations):
        prompt[:,:max_in_seq_length+1] = data[iter*eval_batch_size : (iter+1)*eval_batch_size,:max_in_seq_length+1]    # prefix + <start of sequence>
        generated = model.generate(prompt, max_out_seq_length-1, temperature=1.0, top_k=1)
        output = generated[:,max_in_seq_length+1:].contiguous().view(-1)
        target = data[iter*eval_batch_size : (iter+1)*eval_batch_size,max_in_seq_length+1:].contiguous().view(-1)
        it_token_accuracy, it_seq_accuracy = results_accuracy(output, target, eval_batch_size)
        token_accuracy += it_token_accuracy
        seq_accuracy += it_seq_accuracy
        outputs[iter*eval_batch_size : (iter+1)*eval_batch_size,:]=generated[:,max_in_seq_length+1:].cpu().numpy()
    return token_accuracy/iterations, seq_accuracy/iterations,outputs

def compute_avg_copy_positions(data, model, enc_func, dec_func, device, vocab_size, source_indices, target_indices):
    avg_positions = np.zeros(len(source_indices))

    model.return_all_logits = True
    for i in range(data.shape[0]):
        prompt_str = decode(data[i, :-1], dec_func)
        prompt_tensor = to_tensor_prompts(prompt_str, enc_func, device)

        _, cache = model.run_with_cache(prompt_tensor, remove_batch_dim=True)

        key = "hook_post_li_emb.0"
        post_l0_emb = cache[key]
        post_l0_logits = model.lm_head(post_l0_emb)
        l1_top_k_subs, _ = from_logits_to_top_k_chars(post_l0_logits, dec_func, vocab_size)

        source_chars = [prompt_str[j] for j in source_indices]
        target_sublists = [l1_top_k_subs[j] for j in target_indices]

        for k, (char, sublist) in enumerate(zip(source_chars, target_sublists)):
            if char not in sublist:
                raise ValueError(f"Source char '{char}' not found in target sublist at prompt {i}")
            avg_positions[k] += sublist.index(char)

    avg_positions /= data.shape[0]
    return avg_positions

def compute_avg_induction_attention(data, model, enc_func, dec_func, device, target_indices, get_source_index_func):
    avg_attention = np.zeros(len(target_indices))
    counts = np.zeros(len(target_indices))

    model.return_all_logits = True
    for i in range(data.shape[0]):
        prompt_str = decode(data[i, :-1], dec_func)
        prompt_tensor = to_tensor_prompts(prompt_str, enc_func, device)

        _, cache = model.run_with_cache(prompt_tensor, remove_batch_dim=True)

        key = "transformer.h.1.attn.hook_pattern"
        attention_matrix = cache[key][0].cpu().numpy()

        source_indices = [get_source_index_func(prompt_str, t) for t in target_indices]

        for k, (target_idx, source_idx) in enumerate(zip(target_indices, source_indices)):
            if source_idx is not None:
                avg_attention[k] += attention_matrix[target_idx][source_idx]
                counts[k] += 1

    valid = counts > 0
    avg_attention[valid] /= counts[valid]
    return avg_attention, counts

def get_corresponding_source_index_rule_completion(prompt,target_index):
    if prompt[target_index]!='>':
        return None
    
    target_char=prompt[target_index+1]

    search_substring='>'+target_char
    search_area=prompt[:23]
    source_index=search_area.find(search_substring)
    if source_index==-1:
        raise ValueError(f"Substring '{search_substring}' not found in search area '{search_area}'")
    return source_index+1

def get_corresponding_source_index_rule_chaining(prompt,target_index):
    if prompt[target_index]!=',' or prompt[target_index+1]=='_':
        return None
    
    target_char=prompt[target_index+1]

    search_substring=target_char+'>'
    search_area=prompt[:23]
    source_index=search_area.find(search_substring)
    if source_index==-1:
        raise ValueError(f"Substring '{search_substring}' not found in search area '{search_area}'")
    return source_index

def compute_avg_qkv_positions(data, model, enc_func, dec_func, device, vocab_size,
                               target_indices, get_source_index_func, layer=1,
                               thr_sk_Q=0.7, thr_sk_K=0.99, thr_sk_V=0.7):
    avg_Q_positions = np.zeros(len(target_indices))
    avg_K_positions = np.zeros(len(target_indices))
    avg_V_positions = np.zeros(len(target_indices))
    counts = np.zeros(len(target_indices))

    n_embd = model.transformer.h[layer].attn.c_attn.weight.shape[1]
    W_Q = model.transformer.h[layer].attn.c_attn.weight.split(n_embd, dim=0)[0].T
    W_K = model.transformer.h[layer].attn.c_attn.weight.split(n_embd, dim=0)[1].T
    W_V = model.transformer.h[layer].attn.c_attn.weight.split(n_embd, dim=0)[2].T

    W_Q_inv, _ = truncated_pseudoinverse(W_Q.cpu().numpy(), thr_sk=thr_sk_Q)
    W_K_inv, _ = truncated_pseudoinverse(W_K.cpu().numpy(), thr_sk=thr_sk_K)
    W_V_inv, _ = truncated_pseudoinverse(W_V.cpu().numpy(), thr_sk=thr_sk_V)

    model.return_all_logits = True
    for i in range(data.shape[0]):
        prompt_str = decode(data[i, :-1], dec_func)
        prompt_tensor = to_tensor_prompts(prompt_str, enc_func, device)

        _, cache = model.run_with_cache(prompt_tensor, remove_batch_dim=True)

        cache_key = "hook_post_li_emb.0" if layer >= 1 else "hook_pre_l0_emb"
        res_stream = cache[cache_key]
        res_stream = model.transformer.h[layer].ln_1(res_stream)

        res_stream_Q = res_stream @ W_Q @ torch.tensor(W_Q_inv, device=device)
        res_stream_K = res_stream @ W_K @ torch.tensor(W_K_inv, device=device)
        res_stream_V = res_stream @ W_V @ torch.tensor(W_V_inv, device=device)

        Q_letters, _ = from_logits_to_top_k_chars(model.lm_head(res_stream_Q), dec_func, vocab_size)
        K_letters, _ = from_logits_to_top_k_chars(model.lm_head(res_stream_K), dec_func, vocab_size)
        V_letters, _ = from_logits_to_top_k_chars(model.lm_head(res_stream_V), dec_func, vocab_size)

        source_indices = [get_source_index_func(prompt_str, t) for t in target_indices]

        for k, (target_idx, source_idx) in enumerate(zip(target_indices, source_indices)):
            if source_idx is None:
                continue
            # Q at target encodes the first char of the current output rule
            expected_Q_char = prompt_str[target_idx - 1]
            # K at source encodes the first char of the matching premise rule
            expected_K_char = prompt_str[target_idx - 1] 
            # V at source encodes the second char (the one being looked up)
            expected_V_char = prompt_str[source_idx]

            if expected_Q_char not in Q_letters[target_idx]:
                raise ValueError(f"Expected Q char '{expected_Q_char}' not found at position {target_idx} in prompt {i}")
            if expected_K_char not in K_letters[source_idx]:
                raise ValueError(f"Expected K char '{expected_K_char}' not found at position {source_idx} in prompt {i}")
            if expected_V_char not in V_letters[source_idx]:
                raise ValueError(f"Expected V char '{expected_V_char}' not found at position {source_idx} in prompt {i}")

            avg_Q_positions[k] += Q_letters[target_idx].index(expected_Q_char)
            avg_K_positions[k] += K_letters[source_idx].index(expected_K_char)
            avg_V_positions[k] += V_letters[source_idx].index(expected_V_char)
            counts[k] += 1

    valid = counts > 0
    avg_Q_positions[valid] /= counts[valid]
    avg_K_positions[valid] /= counts[valid]
    avg_V_positions[valid] /= counts[valid]

    return avg_Q_positions, avg_K_positions, avg_V_positions, counts

# ---