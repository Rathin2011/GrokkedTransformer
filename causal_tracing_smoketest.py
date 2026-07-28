"""
Mechanical smoke test for causal_tracing_composition.py: confirms the
patching code actually executes (loads model, builds counterfactuals,
patches+re-executes remaining layers, returns ranks) without erroring.
Not meant to find evidence of composition -- run only against a checkpoint
already known to show ~0% test_inferred_iid, so no signal is expected here.

Differences from causal_tracing_composition.py (kept minimal, on purpose):
  - device pinned to cuda:0 instead of the hardcoded cuda:5 (crashes on any
    node without 6 visible GPUs).
  - only runs ONE checkpoint (passed via --checkpoint) instead of every
    checkpoint in the directory.
  - only samples N_SMOKE_EXAMPLES=5 trials instead of 300, for speed.
Everything else (rank computation, patching loop, counterfactual
construction) is copied verbatim from the original script.
"""
import numpy as np
import json
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import torch
import torch.nn.functional as F
import random
import argparse

N_SMOKE_EXAMPLES = 5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--model_dir", default=None, type=str, required=True)
    parser.add_argument("--checkpoint", default=None, type=str, required=True)
    parser.add_argument("--num_layer", default=8, type=int)
    parser.add_argument("--wd", default=0.1, type=float)
    args = parser.parse_args()
    dataset, model_dir = args.dataset, args.model_dir

    directory = os.path.join(model_dir, "{}_{}_{}".format(dataset, args.wd, args.num_layer))
    device = torch.device('cuda:0')

    all_atomic = set()
    atomic_dict = dict()
    with open("data/{}/train.json".format(dataset)) as f:
        train_items = json.load(f)
    for item in tqdm(train_items, desc="indexing train.json"):
        temp = item['target_text'].strip("><").split("><")
        if len(temp) != 4:
            continue
        h, r, t = temp[:3]
        atomic_dict[(h, r)] = t
        all_atomic.add((h, r, t))

    id_atomic = set()
    for item in tqdm(train_items, desc="indexing id_atomic"):
        temp = item['target_text'].strip("><").split("><")
        if len(temp) == 4:
            continue
        h, r1, r2, t = temp[:4]
        b = atomic_dict[(h, r1)]
        assert atomic_dict[(b, r2)] == t
        id_atomic.add((h, r1, b))
        id_atomic.add((b, r2, t))

    h2rt_train = dict()
    for (h, r, t) in id_atomic:
        h2rt_train.setdefault(h, []).append((r, t))

    with open("data/{}/test.json".format(dataset)) as f:
        pred_data = json.load(f)
    d = dict()
    for item in pred_data:
        d.setdefault(item['type'], []).append(item)

    def return_rank(hd, word_embedding, token, token_list=None):
        logits_ = torch.matmul(hd, word_embedding.T)
        rank = []
        for j in range(len(logits_)):
            log = logits_[j].cpu().numpy()
            temp = [[i, log[i]] for i in range(len(log))] if token_list is None else [[i, log[i]] for i in token_list]
            temp.sort(key=lambda var: var[1], reverse=True)
            rank.append([var[0] for var in temp].index(token))
        return rank

    np.random.seed(0)
    split = 'train_inferred'
    rand_inds = np.random.choice(len(d[split]), N_SMOKE_EXAMPLES, replace=False).tolist()
    target_layer = args.num_layer

    model_path = os.path.join(directory, args.checkpoint)
    print("Loading model from", model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    word_embedding = model.lm_head.weight.data
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    full_list = []
    for index in tqdm(rand_inds, desc="smoke-test trials"):
        random.seed(index)
        res_dict = dict()

        query = d[split][index]['input_text']
        h, n_r, r = query.strip("><").split("><")
        b = atomic_dict[(h, n_r)]
        t = atomic_dict[(b, r)]

        decoder_temp = tokenizer([query], return_tensors="pt", padding=True)
        decoder_input_ids = decoder_temp["input_ids"].to(device)
        decoder_attention_mask = decoder_temp["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, output_hidden_states=True)
        all_hidden_states = outputs['hidden_states']

        rank_before = return_rank(all_hidden_states[target_layer][0, :, :], word_embedding, tokenizer("<" + t + ">")['input_ids'][0])[-1]
        res_dict['rank_before'] = rank_before

        # perturb the head entity, patch at layer 1, re-execute to target_layer
        all_ = set()
        assert (h, n_r) in atomic_dict
        for head in h2rt_train:
            if (head, n_r) not in atomic_dict:
                all_.add(head)
                continue
            tail = atomic_dict[(head, n_r)]
            if (tail, r) not in atomic_dict or atomic_dict[(tail, r)] != t:
                all_.add(head)
        query_ctft = "<{}>".format(random.choice(list(all_)))

        decoder_temp = tokenizer([query_ctft], return_tensors="pt", padding=True)
        decoder_input_ids_ = decoder_temp["input_ids"].to(device)
        decoder_attention_mask_ = decoder_temp["attention_mask"].to(device)

        with torch.no_grad():
            outputs_ctft = model(input_ids=decoder_input_ids_, attention_mask=decoder_attention_mask_, output_hidden_states=True)
        all_hidden_states_ctft = outputs_ctft['hidden_states']

        layer_to_intervene = 1
        hidden_states = all_hidden_states[layer_to_intervene].clone()
        hidden_states_ctft = all_hidden_states_ctft[layer_to_intervene]
        hidden_states[0, 0, :] = hidden_states_ctft[0, 0, :]

        with torch.no_grad():
            for i in range(layer_to_intervene, target_layer):
                f_layer = model.transformer.h[i]
                residual = hidden_states
                hidden_states = f_layer.ln_1(hidden_states)
                attn_output = f_layer.attn(hidden_states)[0]
                hidden_states = attn_output + residual
                residual = hidden_states
                hidden_states = f_layer.ln_2(hidden_states)
                feed_forward_hidden_states = f_layer.mlp.c_proj(f_layer.mlp.act(f_layer.mlp.c_fc(hidden_states)))
                hidden_states = residual + feed_forward_hidden_states
            hidden_states = model.transformer.ln_f(hidden_states)
        rank_after = return_rank(hidden_states[0, :, :], word_embedding, tokenizer("<" + t + ">")['input_ids'][0])[-1]
        res_dict['h_1'] = rank_after

        full_list.append(res_dict)
        print(res_dict)

    print("SMOKE TEST OK -- ran", len(full_list), "trials without error")

if __name__ == '__main__':
    main()
