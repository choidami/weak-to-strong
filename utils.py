from functools import partial
import numpy as np

from datasets import load_from_disk


def set_to_groundtruth(ex, idx, indices):
    if idx in indices:
        ex['soft_label'] = [1 - float(ex["gt_label"]), float(ex["gt_label"])]
        ex['use_gt'] = True
    else:
        ex['use_gt'] = False
    return ex


def maybe_balance_dataset(ds, indices, balance_method):
    print("BALANCING DATA!!!")
    print("pre-balance:", len(indices))
    subset = ds.select(indices)

    # Note that selecting ground-truth examples to be class-balanced is not really
    # possible unless we know the label in advance.
    # There are two options:
    # 1) Use all selected ground-truth examples even if they are class-imbalanced.
    # 2) Use a subset of ground-truth examples such that the classes are balanced.
    # 3) Up-sample the smaller class (this will result in duplicate data).
    if balance_method == "subset":
        class0_idxs = np.where(np.array(subset['gt_label']) == 0)[0]
        class1_idxs = np.where(np.array(subset['gt_label']) == 1)[0]
        min_len = min(len(class0_idxs), len(class1_idxs))
        class0_idxs = class0_idxs[:min_len]
        class1_idxs = class1_idxs[:min_len]
        all_idxs = list(class0_idxs) + list(class1_idxs)
        indices = [indices[idx] for idx in all_idxs]

        assert sum(ds.select(indices)['gt_label']) == len(indices) // 2
    elif balance_method == "upsample":
        class0_idxs = np.where(np.array(subset['gt_label']) == 0)[0]
        class1_idxs = np.where(np.array(subset['gt_label']) == 1)[0]
        if len(class0_idxs) > len(class1_idxs):
            extra = len(class0_idxs) - len(class1_idxs)
            class1_idxs = list(class1_idxs) + list(np.random.choice(class1_idxs, extra))
        elif len(class0_idxs) < len(class1_idxs):
            extra = len(class1_idxs) - len(class0_idxs)
            class0_idxs = list(class0_idxs) + list(np.random.choice(class0_idxs, extra))
        all_idxs = list(class0_idxs) + list(class1_idxs)
        indices = [indices[idx] for idx in all_idxs]

        assert sum(ds.select(indices)['gt_label']) == len(indices) // 2

    # Shuffle gt_indices.
    np.random.shuffle(indices)

    print("post-balance:", len(indices))
    return indices


def get_first_round_datasets(
    weak_labels_path,
    gt_ratio=0.0,
    balance_method=None,
    dset_type="uniform",
    gt_selection_strategy="random",
    choose_all_weak=False,
    seed=0
):
    np.random.seed(seed)

    ds = load_from_disk(weak_labels_path)

    if dset_type == "weak" and choose_all_weak:
        return ds, []
    
    if gt_ratio > 0.0:
        num_train = len(ds)
        num_gt = min(num_train, int(gt_ratio * num_train))

        if gt_selection_strategy == "wm_conf":
            gt_indices = np.argsort(np.stack(ds['soft_label']).max(-1))[: num_gt]
        elif gt_selection_strategy == "random":
            gt_indices = np.random.choice(num_train, num_gt, replace=False)
        else:
            raise ValueError()

        gt_indices = maybe_balance_dataset(ds, gt_indices, balance_method)

        # Set datapoints corresponding to gt_indices to have soft_label be ground-truth values. 
        ds = ds.map(partial(set_to_groundtruth, indices=gt_indices), with_indices=True)
        if dset_type == "uniform":
            pass
        elif dset_type == "gt":
            # Select only ground-truth datapoints.
            ds = ds.select(list(gt_indices))
        elif dset_type == "weak" and not choose_all_weak:
            # Select only weak datapoints.
            non_gt_indices = list(i for i in range(num_train) if i not in gt_indices)
            ds = ds.select(list(non_gt_indices))
        else:
            raise ValueError()
    else:
        gt_indices = []

    return ds, gt_indices


def get_second_round_datasets(
    weak_labels_path,
    gt_ratio,
    dset_type,
    balance_method=None,
    gt_selection_strategy="random",
    strong_labels_path=None,
    choose_all_weak=False,
    seed=0
):
    np.random.seed(seed)

    ds = load_from_disk(weak_labels_path)

    if dset_type == "weak" and choose_all_weak:
        return ds, []
    
    num_train = len(ds)
    num_gt = min(num_train, int(gt_ratio * num_train))
    if gt_selection_strategy == "wm_conf":
        gt_indices = np.argsort(np.stack(ds['soft_label']).max(-1))[: num_gt]
    elif gt_selection_strategy == "random":
        gt_indices = np.random.choice(num_train, num_gt, replace=False)
    elif gt_selection_strategy == "sm_conf":
        sm_ds = load_from_disk(strong_labels_path)
        gt_indices = np.argsort(np.stack(sm_ds['soft_label']).max(-1))[: num_gt]
    else:
        raise ValueError()
    
    gt_indices = maybe_balance_dataset(ds, gt_indices, balance_method)

    if dset_type == "gt":
        ds = ds.map(partial(set_to_groundtruth, indices=gt_indices), with_indices=True)
        ds = ds.select(list(gt_indices))
    elif dset_type == "weak" and not choose_all_weak:
        # Select only weak datapoints.
        non_gt_indices = list(i for i in range(len(ds)) if i not in gt_indices)
        ds = ds.select(list(non_gt_indices))
    else:
        raise ValueError()
    
    return ds, gt_indices
