"""Exact M/G Bayes predictors for the composition task + KL-based d_rel,
evaluated on the test_inferred_iid split (held-out, never-trained-on
combinations, by construction).

On this split, both reference predictors are constant across examples:
  M (memorizing): uniform over all entities -- M only knows literally-
    memorized training triples, and every test_inferred_iid triple is, by
    construction, NOT in the training set, so M has zero information.
  G (generalizing): point mass on the true answer -- G has learned the two
    atomic hops correctly and doesn't care whether this exact combination
    was trained on.

G is Laplace-smoothed by a tiny epsilon (mixed with uniform) so symmetrized
KL against it stays finite (a pure point mass makes KL(model, G) infinite
whenever the model assigns any other outcome nonzero probability).

d_rel = KL_sym(model, M) / (KL_sym(model, M) + KL_sym(model, G))
  0 = model's distribution matches G (generalizing)
  1 = model's distribution matches M (memorizing)
Same convention as compositional_urns' d_rel_m.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

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
    parser.add_argument("--model_dir", default="outputs")
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--num_layer", type=int, default=8)
    parser.add_argument("--n_examples", type=int, default=150)
    parser.add_argument("--checkpoint", default=None, help="run only this one checkpoint")
    parser.add_argument("--save_path", required=True)
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
    # vocab.json entries are already bracketed, e.g. "<e_0>", not bare "e_0"
    entity_tokens = [t for t in vocab_list if t.startswith("<e_")]
    entity_index = {e: i for i, e in enumerate(entity_tokens)}
    n_ent = len(entity_tokens)
    assert n_ent > 0, "no entity tokens found in vocab.json -- check the prefix filter"
    uniform_dist = np.ones(n_ent) / n_ent

    random.seed(0)
    split = 'test_inferred_iid'
    pool = by_type[split]
    n = min(args.n_examples, len(pool))
    sample = random.sample(pool, n)

    all_checkpoints = sorted(
        [c for c in os.listdir(directory) if c.startswith("checkpoint")],
        key=lambda c: int(c.split("-")[1]),
    )
    if args.checkpoint is not None:
        all_checkpoints = [args.checkpoint]

    results = []
    for checkpoint in tqdm(all_checkpoints):
        model_path = os.path.join(directory, checkpoint)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        entity_ids = np.array([tokenizer(e)['input_ids'][0] for e in entity_tokens], dtype=np.int64)

        d_rels = []
        for item in sample:
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

            g_dist = np.full(n_ent, EPS / n_ent)
            g_dist[entity_index[f"<{t_true}>"]] += 1.0 - EPS

            kl_m = kl_sym(probs_over_entities, uniform_dist)
            kl_g = kl_sym(probs_over_entities, g_dist)
            d_rel = kl_m / (kl_m + kl_g + 1e-12)
            d_rels.append(d_rel)

        step = int(checkpoint.split("-")[1])
        mean_d_rel = float(np.mean(d_rels))
        results.append({"step": step, "d_rel_mean": mean_d_rel, "n": len(d_rels)})
        print(f"checkpoint={checkpoint} step={step} d_rel_mean={mean_d_rel:.4f}")
        del model
        torch.cuda.empty_cache()

    with open(args.save_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", args.save_path)


if __name__ == "__main__":
    main()
