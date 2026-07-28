"""d_G / d_M: two independent distances instead of a forced M-vs-G ratio.

  d_G = 1 - p_theta(t*)                         -- distance from confidently correct
  d_M = TV(p_theta, uniform) = 0.5 * sum_e |p_theta(e) - 1/|E||   -- distance from genuinely uniform

Evaluated on test_inferred_iid (held-out, never-trained-on combinations, by
construction), same split used throughout this sweep.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


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
    entity_tokens = [t for t in vocab_list if t.startswith("<e_")]
    entity_index = {e: i for i, e in enumerate(entity_tokens)}
    n_ent = len(entity_tokens)
    assert n_ent > 0

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

        d_Gs, d_Ms = [], []
        n_correct = 0
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

            true_idx = entity_index[f"<{t_true}>"]
            p_true = probs_over_entities[true_idx]
            d_G = 1.0 - p_true
            d_M = 0.5 * np.sum(np.abs(probs_over_entities - 1.0 / n_ent))

            d_Gs.append(d_G)
            d_Ms.append(d_M)
            if int(np.argmax(probs_over_entities)) == true_idx:
                n_correct += 1

        step = int(checkpoint.split("-")[1])
        mean_dG = float(np.mean(d_Gs))
        mean_dM = float(np.mean(d_Ms))
        results.append({
            "step": step,
            "d_G_mean": mean_dG,
            "d_M_mean": mean_dM,
            "accuracy": n_correct / len(sample),
            "n": len(sample),
        })
        print(f"checkpoint={checkpoint} step={step} d_G={mean_dG:.4f} d_M={mean_dM:.4f} acc={n_correct/len(sample):.4f}")
        del model
        torch.cuda.empty_cache()

    with open(args.save_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", args.save_path)


if __name__ == '__main__':
    main()
