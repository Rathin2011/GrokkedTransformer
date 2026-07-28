"""Diagnostic: split per-example d_rel by whether the model's argmax was
correct or not, to check whether the mean d_rel is being dominated by
confidently-wrong predictions rather than genuinely tracking M-vs-G."""
import json
import os
import random

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import argparse

EPS = 1e-6


def kl_sym(p, q, floor=1e-12):
    p = np.clip(p, floor, None)
    q = np.clip(q, floor, None)
    p = p / p.sum()
    q = q / q.sum()
    return 0.5 * np.sum(p * np.log(p / q)) + 0.5 * np.sum(q * np.log(q / p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_dir", default="outputs")
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--num_layer", type=int, default=8)
    parser.add_argument("--n_examples", type=int, default=150)
    args = parser.parse_args()

    directory = os.path.join(args.model_dir, "{}_{}_{}".format(args.dataset, args.wd, args.num_layer))
    device = torch.device('cuda:0')

    with open(f"data/{args.dataset}/train.json") as f:
        train_items = json.load(f)
    atomic_dict = {}
    for item in train_items:
        temp = item['target_text'].strip("><").split("><")
        if len(temp) == 4:
            h, r, t = temp[:3]
            atomic_dict[(h, r)] = t

    with open(f"data/{args.dataset}/test.json") as f:
        test_items = json.load(f)
    by_type = {}
    for item in test_items:
        by_type.setdefault(item['type'], []).append(item)

    with open(f"data/{args.dataset}/vocab.json") as f:
        vocab_list = json.load(f)
    entity_tokens = [t for t in vocab_list if t.startswith("<e_")]
    entity_index = {e: i for i, e in enumerate(entity_tokens)}
    n_ent = len(entity_tokens)
    uniform_dist = np.ones(n_ent) / n_ent

    random.seed(0)
    pool = by_type['test_inferred_iid']
    n = min(args.n_examples, len(pool))
    sample = random.sample(pool, n)

    model_path = os.path.join(directory, args.checkpoint)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    entity_ids = np.array([tokenizer(e)['input_ids'][0] for e in entity_tokens], dtype=np.int64)

    correct_d_rels, incorrect_d_rels = [], []
    correct_true_probs, incorrect_true_probs = [], []
    correct_top_probs, incorrect_top_probs = [], []

    for item in tqdm(sample):
        query = item['input_text']
        h, r1, r2 = query.strip("><").split("><")
        b = atomic_dict[(h, r1)]
        t_true = atomic_dict[(b, r2)]

        enc = tokenizer([query], return_tensors="pt")
        input_ids = enc['input_ids'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1, :]
        probs_full = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        probs_over_entities = probs_full[entity_ids]
        probs_over_entities = probs_over_entities / probs_over_entities.sum()

        true_idx = entity_index[f"<{t_true}>"]
        true_prob = probs_over_entities[true_idx]
        top_idx = int(np.argmax(probs_over_entities))
        top_prob = probs_over_entities[top_idx]
        is_correct = (top_idx == true_idx)

        g_dist = np.full(n_ent, EPS / n_ent)
        g_dist[true_idx] += 1.0 - EPS

        kl_m = kl_sym(probs_over_entities, uniform_dist)
        kl_g = kl_sym(probs_over_entities, g_dist)
        d_rel = kl_m / (kl_m + kl_g + 1e-12)

        if is_correct:
            correct_d_rels.append(d_rel)
            correct_true_probs.append(true_prob)
            correct_top_probs.append(top_prob)
        else:
            incorrect_d_rels.append(d_rel)
            incorrect_true_probs.append(true_prob)
            incorrect_top_probs.append(top_prob)

    print(f"n_correct={len(correct_d_rels)}, n_incorrect={len(incorrect_d_rels)}")
    if correct_d_rels:
        print(f"CORRECT:   mean d_rel={np.mean(correct_d_rels):.4f}  mean P(true)={np.mean(correct_true_probs):.4f}  mean top_prob={np.mean(correct_top_probs):.4f}")
    if incorrect_d_rels:
        print(f"INCORRECT: mean d_rel={np.mean(incorrect_d_rels):.4f}  mean P(true)={np.mean(incorrect_true_probs):.6f}  mean top_prob={np.mean(incorrect_top_probs):.4f}")
    overall = correct_d_rels + incorrect_d_rels
    print(f"OVERALL mean d_rel={np.mean(overall):.4f}")
    if incorrect_d_rels:
        contribution_incorrect = len(incorrect_d_rels) * np.mean(incorrect_d_rels) / len(overall)
        contribution_correct = len(correct_d_rels) * np.mean(correct_d_rels) / len(overall) if correct_d_rels else 0.0
        print(f"Contribution to mean from incorrect examples: {contribution_incorrect:.4f}")
        print(f"Contribution to mean from correct examples:   {contribution_correct:.4f}")


if __name__ == '__main__':
    main()
