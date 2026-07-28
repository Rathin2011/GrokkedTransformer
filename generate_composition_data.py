"""Standalone port of composition.ipynb's data generation, for batch/cluster use.

Logic is verbatim from the notebook (build_dicts/choose/split/form_items/
build_dataset) -- only difference is --phis lets you generate a subset of
the paper's six phi values instead of always generating all six, and
--num_entities/--num_relations/--out_degree are exposed as CLI args instead
of hardcoded, for running a smaller pilot before the full 2000/200 setup.
"""

import argparse
import json
import os
from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm


def build_dicts(entities):
    entity2ind = dict()
    ind2entity = []
    for i in range(len(entities)):
        entity = entities[i]
        if entity not in ind2entity:
            ind2entity.append(entity)
            entity2ind[entity] = len(ind2entity) - 1
    return ind2entity, entity2ind


def choose(arr, ratio_or_count):
    if isinstance(ratio_or_count, float):
        num = round(ratio_or_count * len(arr))
    elif isinstance(ratio_or_count, int):
        num = ratio_or_count
    else:
        assert False
    if num >= len(arr):
        return arr
    rand_inds = np.random.choice(len(arr), num, replace=False).tolist()
    return [arr[i] for i in rand_inds]


def split(arr, ratio_or_count):
    if isinstance(ratio_or_count, float):
        num = round(ratio_or_count * len(arr))
    elif isinstance(ratio_or_count, int):
        num = ratio_or_count
    else:
        assert False
    train, test = [], []
    rand_inds = np.random.choice(len(arr), num, replace=False).tolist()
    rand_inds = set(rand_inds)
    for i in tqdm(range(len(arr))):
        if i in rand_inds:
            train.append(arr[i])
        else:
            test.append(arr[i])
    return [train, test]


def form_items(c, t):
    input_text = "".join(c)
    target_text = input_text + "".join([t, "</a>"])
    return {"input_text": input_text, "target_text": target_text}


def build_dataset(num_entities, num_relations, out_degree=20, split_train_inferred=False):
    entities = ["<e_{}>".format(i) for i in range(num_entities)]
    ind2entity, entity2ind = build_dicts(entities)

    relations = ["<r_{}>".format(i) for i in range(num_relations)]
    ind2relation, relation2ind = build_dicts(relations)

    atomic_dict = dict()
    atomic_facts = []
    atomics = []

    for i in tqdm(range(num_entities)):
        num_rows = out_degree
        selected_rows = np.random.choice(num_relations, size=num_rows, replace=False).tolist()
        for row_idx in selected_rows:
            col_idx = np.random.randint(num_entities)
            h, r, t = ind2entity[i], ind2relation[row_idx], ind2entity[col_idx]
            atomic_facts.append(form_items([h, r], t))
            atomics.append((h, r, t))
            if h not in atomic_dict:
                atomic_dict[h] = []
            atomic_dict[h].append((r, t))

    if not split_train_inferred:
        inferred_facts = []
        for ent in tqdm(entities):
            for (r1, b) in atomic_dict[ent]:
                for (r2, t) in atomic_dict[b]:
                    inferred_facts.append(form_items([ent, r1, r2], t))
        return entities, relations, atomic_facts, inferred_facts

    OOD_ratio = 0.05
    OOD_facts, ID_facts = split(atomics, round(len(atomics) * OOD_ratio))
    OOD_facts, ID_facts = set(OOD_facts), set(ID_facts)

    id_atomic_facts = [form_items([h, r], t) for (h, r, t) in ID_facts]
    ood_atomic_facts = [form_items([h, r], t) for (h, r, t) in OOD_facts]

    train_inferred_facts, test_inferred_iid, test_inferred_ood = [], [], []
    for ent in tqdm(entities):
        for (r1, b) in atomic_dict[ent]:
            for (r2, t) in atomic_dict[b]:
                if (ent, r1, b) in OOD_facts or (b, r2, t) in OOD_facts:
                    if (ent, r1, b) in OOD_facts and (b, r2, t) in OOD_facts:
                        test_inferred_ood.append(form_items([ent, r1, r2], t))
                    continue
                if np.random.uniform() > 0.005:
                    train_inferred_facts.append(form_items([ent, r1, r2], t))
                else:
                    test_inferred_iid.append(form_items([ent, r1, r2], t))

    return (
        entities,
        relations,
        id_atomic_facts,
        ood_atomic_facts,
        train_inferred_facts,
        test_inferred_iid,
        test_inferred_ood,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_entities", type=int, default=2000)
    p.add_argument("--num_relations", type=int, default=200)
    p.add_argument("--out_degree", type=int, default=20)
    p.add_argument("--test_size", type=int, default=3000)
    p.add_argument("--phis", type=float, nargs="+", default=[3.6])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="data")
    args = p.parse_args()

    np.random.seed(args.seed)

    (
        train_entities,
        train_relations,
        id_atomic_facts,
        ood_atomic_facts,
        train_inferred_facts,
        test_inferred_iid,
        test_inferred_facts,
    ) = build_dataset(
        args.num_entities, args.num_relations, args.out_degree, split_train_inferred=True
    )

    vocab = list(train_entities) + list(train_relations)
    vocab = vocab + ["<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>"]
    assert len(vocab) == len(set(vocab))
    print("vocab size:", len(vocab))

    id_atomic_facts_ds = choose(id_atomic_facts, args.test_size)
    ood_atomic_facts_ds = choose(ood_atomic_facts, args.test_size)
    test_inferred_iid_ds = choose(test_inferred_iid, args.test_size)
    test_inferred_facts_ds_full = choose(test_inferred_facts, args.test_size)

    all_atomics = id_atomic_facts + ood_atomic_facts
    print("num atomics:", len(all_atomics), "num train_inferred available:", len(train_inferred_facts))

    for phi in args.phis:
        dataset_name = "composition.{}.{}.{}".format(args.num_entities, args.num_relations, phi)
        out_dir = os.path.join(args.out_dir, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        train_inferred_facts_ds = choose(train_inferred_facts, round(phi * len(id_atomic_facts)))

        probes = []
        for item in id_atomic_facts_ds:
            probes.append(deepcopy(item))
            probes[-1]["type"] = "id_atomic"
        for item in ood_atomic_facts_ds:
            probes.append(deepcopy(item))
            probes[-1]["type"] = "ood_atomic"
        for item in choose(train_inferred_facts_ds, args.test_size):
            probes.append(deepcopy(item))
            probes[-1]["type"] = "train_inferred"
        for item in test_inferred_iid_ds:
            probes.append(deepcopy(item))
            probes[-1]["type"] = "test_inferred_iid"
        for item in test_inferred_facts_ds_full:
            probes.append(deepcopy(item))
            probes[-1]["type"] = "test_inferred_ood"

        with open(os.path.join(out_dir, "train.json"), "w", encoding="utf-8") as f:
            json.dump(all_atomics + train_inferred_facts_ds, f)
        with open(os.path.join(out_dir, "valid.json"), "w", encoding="utf-8") as f:
            json.dump(test_inferred_facts_ds_full, f)
        with open(os.path.join(out_dir, "test.json"), "w", encoding="utf-8") as f:
            json.dump(probes, f)
        with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f)

        print(
            f"phi={phi}: wrote {out_dir} "
            f"(train={len(all_atomics) + len(train_inferred_facts_ds)}, "
            f"valid={len(test_inferred_facts_ds_full)}, test={len(probes)})"
        )


if __name__ == "__main__":
    main()
