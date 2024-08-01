import json
import os
import random
import subprocess
from functools import partial
from typing import Dict, List, Optional, Literal

import fire
import numpy as np
import torch
from datasets import load_dataset, load_from_disk

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset,
                                     tokenize_dataset)
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model
from utils import get_first_round_datasets, get_second_round_datasets

# NOTE learning rates are not particularly tuned, work somewhat reasonably at train batch size 32
MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
        minibatch_size_per_device=32,
    ),
    ModelConfig(
        name="gpt2-medium",
        default_lr=5e-5,
        eval_batch_size=32,
        minibatch_size_per_device=16,
    ),
    ModelConfig(
        name="gpt2-large",
        default_lr=1e-5,
        eval_batch_size=32,
        minibatch_size_per_device=4,
    ),
    ModelConfig(
        name="gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        # Should use model_parallel on V100s (note: ironically if you have a single V100 it should run,
        # but if you have multiple it won't run without model_parallel because of the overhead of data
        # parallel training).
        model_parallel=(
            # torch.cuda.get_device_properties(0).total_memory < 35e9
            torch.cuda.device_count() > 1
        ),
        minibatch_size_per_device=1,
    ),
    ModelConfig(
        name="Qwen/Qwen-1_8B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.get_device_properties(0).total_memory < 35e9
            and torch.cuda.device_count() > 1
        ),
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "5fde88dff770a7d036847211f5d9d9705f0caa69",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-7B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "d4efd21e866b9cb3466cb65b963933f5e98016d1",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-14B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this bf16 support and without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "8be2854218fea9054331e217fd26a06f3fd02004",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-72B",
        default_lr=1e-5,
        eval_batch_size=1,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without bf16 support and many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "fec78c0e3b3b10dd9f0ce775c34a686a3255a7d1",
        },
        # This model is really big, save space by using adafactor.
        # Note that even then it will take up ~60GB per GPU on an 8-GPU machine.
        default_optimizer="adafactor",
    ),
]
MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}


loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
}

VALID_LOSSES: List[str] = list(loss_dict.keys())

BLACKLISTED_ARGS = [
    "balance_method",
    "choose_all_weak",
    "first_dset_type",
    "first_gt_selection_strategy",
    "second_dset_type",
    "second_gt_selection_strategy",
    "second_loss",
    "second_linear_probe",
]


def get_config_foldername(config: dict) -> str:
    def shorten_key(key: str) -> str:
        return "".join(word[0] for word in key.split("_"))

    def shorten_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            value = value.split("/")[-1]
            if "_" in value:
                return "_".join(word[:4] for word in value.split("_"))
            else:
                return value
        elif isinstance(value, dict):
            return ""
        elif isinstance(value, list):
            return ""
        else:
            return str(value)

    return "-".join(
        f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items())
        if k not in BLACKLISTED_ARGS
    )
    
    
WEAK_MODEL_CONFIG_NAMES = {
    "boolq": {
        "gpt2": "bs=32-dn=boolq-em=0.1-e=2-ee=1000000-lp=0-l=xent-l=5e-05-ls=cosi_anne-mc=1024-ms=gpt2-nd=20000-ntd=10000-o=adam-s={seed}-twd=0-ws=30",
        "gpt2-medium": "bs=32-dn=boolq-em=0.1-e=2-ee=1000000-lp=0-l=xent-l=5e-05-ls=cosi_anne-mc=1024-ms=gpt2-medium-nd=20000-ntd=10000-o=adam-s={seed}-twd=0-ws=30",
        "gpt2-large": "bs=32-dn=boolq-em=0.1-e=2-ee=1000000-lp=0-l=xent-l=1e-05-ls=cosi_anne-mc=1024-ms=gpt2-large-nd=20000-ntd=10000-o=adam-s={seed}-twd=0-ws=30",
        "gpt2-xl": "bs=32-dn=boolq-em=0.1-e=2-ee=1000000-lp=0-l=xent-l=1e-05-ls=cosi_anne-mc=1024-ms=gpt2-xl-nd=20000-ntd=10000-o=adam-s={seed}-twd=0-ws=30",
    }
}


def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "sciq",
    loss: str = "xent",
    n_docs: int = 20000,
    n_test_docs: int = 10000,
    model_size: str = "gpt2",
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    epochs: int = 2,
    force_retrain: bool = False,
    seed: int = 0,
    minibatch_size_per_device: Optional[float] = None,
    train_with_dropout: bool = False,
    results_folder: str = "results",
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    warmup_steps: int = 30,
    end_mult: float = 0.1,
    # Note: you can pass either weak_model_size or weak_labels_path. If you pass
    # weak_model_size, we will guess the path to the weak labels based on the weak
    # model. If you pass weak_labels_path, we will use that path instead.
    # If you pass neither, we will train on ground truth.
    weak_model_size: Optional[str] = None,
    weak_labels_path: Optional[str] = None,
    sweep_subfolder: str = "default",
    # Set to a very large value so that by default we don't do any intermediate evals but
    # still do final evals (which requires eval_every to be set to a non-zero, non-None value)
    eval_every: int = 1000000,
    # Arguments for when we include ground-truth labels when doing w2s training.
    gt_ratio: float = 0.0,
    balance_method: Literal["subset", "upsample", None] = None,
    choose_all_weak: bool = False,
    first_gt_selection_strategy: Literal["random", "wm_conf", None] = None,
    first_dset_type: Literal["uniform", "gt", "weak", None] = None,
    second_dset_type: Literal["gt", "weak", None] = None,
    second_gt_selection_strategy: Literal["random", "wm_conf", "sm_conf", None] = None,
    second_loss: str = "xent",
    second_linear_probe: bool = False,
    second_batch_size: int = 32,
    second_lr_mult: float = 1.0,
    second_epochs: int = 2,
    second_lr_schedule: str = "cosine_anneal",
    # second_warmup_steps: int = 30,
    # second_end_mult: float = 0.1,
    pretrained_folder: str = None,
    eval_valid: bool = False,
):
    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1
    assert ds_name in VALID_DATASETS, f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    assert (
        weak_model_size is None or weak_labels_path is None
    ), "Can't pass both weak_model_size and weak_labels_path"
    if weak_model_size is None and weak_labels_path is None:
        train_weak_to_strong = False
    else:
        train_weak_to_strong = True
    model_config = MODELS_DICT[model_size]
    if model_config.minibatch_size_per_device > 1:
        minibatch_size_per_device = model_config.minibatch_size_per_device

    use_default_lr = False
    if lr is None:
        assert (
            batch_size == 32
        ), "Learning rates were tuned on batch size 32, you probably want to sweep LR if you are tuning batch size"
        lr = model_config.default_lr
        use_default_lr = True

    if optim is None:
        optim = model_config.default_optimizer

    # The commented out terms are the ones that should not change final results
    config = {
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "ds_name": ds_name,
        "loss": loss,
        "n_docs": n_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "lr": lr,
        "optim": optim,
        "epochs": epochs,
        # "force_retrain": force_retrain,
        "seed": seed,
        # "minibatch_size_per_device": minibatch_size_per_device,
        "train_with_dropout": train_with_dropout,
        # "results_folder": results_folder,
        "linear_probe": linear_probe,
        "lr_schedule": lr_schedule,
        "eval_every": eval_every,
        # "sweep_subfolder": sweep_subfolder,
        "balance_method": balance_method,
        "choose_all_weak": choose_all_weak,
        "first_dset_type": first_dset_type,
        "first_gt_selection_strategy": first_gt_selection_strategy,
        "second_dset_type": second_dset_type,
        "second_gt_selection_strategy": second_gt_selection_strategy,
        "second_loss": second_loss,
        "second_linear_probe": second_linear_probe,
    }
    if warmup_steps > 0:
        config["warmup_steps"] = warmup_steps
    if lr_schedule == "cosine_anneal" and end_mult > 0.0:
        config["end_mult"] = end_mult
    if train_weak_to_strong and second_dset_type:
        config["second_batch_size"] = second_batch_size
        config["second_lr_mult"] = second_lr_mult
        config["second_epochs"] = second_epochs
        config["second_lr_schedule"] = second_lr_schedule
        # if second_warmup_steps > 0:
        #     config["second_warmup_steps"] = second_warmup_steps
        # if second_lr_schedule == "cosine_anneal" and second_end_mult > 0.0:
        #     config["second_end_mult"] = second_end_mult

    if weak_model_size is not None:
        # weak_model_config = config.copy()
        # weak_model_config["model_size"] = weak_model_size
        # weak_model_config["loss"] = "xent"
        # if use_default_lr:
        #     weak_model_config["lr"] = MODELS_DICT[weak_model_size].default_lr

        # weak_model_config_name = get_config_foldername(weak_model_config)
        weak_model_config_name = WEAK_MODEL_CONFIG_NAMES[ds_name][weak_model_size].format(seed=seed)

        weak_labels_path = (
            results_folder + "/" + "ceils" + "/" + weak_model_config_name + "/first" + "/weak_labels"
        )

    eval_batch_size = model_config.eval_batch_size
    random.seed(seed)

    # Load dataset
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]

    split_data = train_dataset.train_test_split(test_size=0.5, seed=seed)
    train1_ds, train2_ds = split_data["train"], split_data["test"]
        
    if train_weak_to_strong:
        if not weak_labels_path.endswith("weak_labels"):
            weak_labels_path = weak_labels_path + "/weak_labels"
        
        train1_ds, gt_indices = get_first_round_datasets(
            weak_labels_path,
            gt_ratio,
            balance_method,
            first_dset_type,
            first_gt_selection_strategy,
            choose_all_weak,
            seed
        )
        # train1_ds = load_from_disk(weak_labels_path)

        if gt_ratio > 0.0:
            config["gt_ratio"] = gt_ratio

        weak_model_config = json.load(open(weak_labels_path.replace("weak_labels", "config.json")))
        config["weak_model_size"] = weak_model_config["model_size"]
        config_name = get_config_foldername(config)
        config["weak_model"] = weak_model_config
        
        config["gt_indices"] = list(gt_indices)
    else:
        print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))

        # Set sweep_subfolder to "ceils".
        sweep_subfolder = "ceils"
        
        config_name = get_config_foldername(config)

    base_save_path = os.path.join(results_folder, sweep_subfolder, config_name)
    first_save_path = os.path.join(results_folder, sweep_subfolder, config_name, "first")
    second_save_path = os.path.join(results_folder, sweep_subfolder, config_name, "second")

    # Skip run if the run has already been evaluated.
    if train_weak_to_strong:
        if second_dset_type is None:
            if os.path.exists(os.path.join(first_save_path, "results_summary.json")):
                return
        else:
            if os.path.exists(os.path.join(second_save_path, "results_summary.json")):
                return
    else:
        if os.path.exists(os.path.join(first_save_path, "results_summary.json")):
            return
    
    logger.configure(
        name="{sweep_subfolder}_{config_name}_{datetime_now}",
        save_path=first_save_path,
        sweep_subfolder=sweep_subfolder,
        config_name=config_name,
    )
    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)
    train1_ds = tokenize_dataset(train1_ds, tokenizer, max_ctx)
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)
    if train2_ds:
        train2_ds = tokenize_dataset(train2_ds, tokenizer, max_ctx)
    
    pretrained_path = None
    if pretrained_folder:
        pretrained_config = config.copy()
        for key in list(pretrained_config.keys()):
            if key.startswith("second_"):
                pretrained_config.pop(key)
        pretrained_config.pop("gt_indices", None)
        pretrained_config.pop("weak_model", None)
        pretrained_config.pop("gt_ratio", None)
        
        pretrained_config_name = get_config_foldername(pretrained_config)
        pretrained_path = os.path.join(results_folder, pretrained_folder, pretrained_config_name, "first")
        # import pdb;pdb.set_trace()
                

    loss_fn = loss_dict[loss]
    print(f"Training model model, size {model_size}")
    test_results, weak_ds = train_and_save_model(
        model_config,
        train1_ds,
        test_ds,
        inference_ds=train2_ds,#(train2_ds if not train_weak_to_strong or second_gt_selection_strategy == "sm_conf" else None),
        batch_size=batch_size,
        load_path=(pretrained_path if pretrained_path else first_save_path),
        save_path=first_save_path,
        loss_fn=loss_fn,
        lr=lr,
        epochs=epochs,
        force_retrain=force_retrain,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        train_with_dropout=train_with_dropout,
        linear_probe=linear_probe,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        end_mult=end_mult,
        optimizer_name=optim,
        eval_every=eval_every,
    )

    eval_ds_path = first_save_path + "/" + ("strong_labels" if train_weak_to_strong else "weak_labels")
    if weak_ds:
        weak_ds.save_to_disk(eval_ds_path)

    acc = np.mean([x["acc"] for x in test_results])
    res_dict = {"accuracy": acc}
    print("accuracy:", acc)

    with open(os.path.join(first_save_path, f"config.json"), "w") as f:
        json.dump(config, f, indent=2, default=int)

    with open(os.path.join(first_save_path, f"results_summary.json"), "w") as f:
        json.dump(res_dict, f, indent=2)
    
    
    # Optionally do second stage of training.
    if not train_weak_to_strong or second_dset_type is None:
        return

    train1_ds, gt_indices = get_second_round_datasets(
        weak_labels_path,
        gt_ratio,
        second_dset_type,
        balance_method,
        second_gt_selection_strategy,
        eval_ds_path,
        choose_all_weak,
        seed
    )
    
    logger.configure(
        name="{sweep_subfolder}_{config_name}_{datetime_now}",
        save_path=second_save_path,
        sweep_subfolder=sweep_subfolder,
        config_name=config_name,
    )
    # Tokenize datasets
    train1_ds = tokenize_dataset(train1_ds, tokenizer, max_ctx)
    valid_ds = tokenize_dataset(split_data["train"], tokenizer, max_ctx)
    second_warmup_steps = int(len(train1_ds) // second_batch_size * second_epochs * 0.1)

    loss_fn = loss_dict[second_loss]
    print(f"Training model model, size {model_size}")
    test_results, weak_ds = train_and_save_model(
        model_config,
        train1_ds,
        test_ds,
        inference_ds=None,
        additional_eval_ds=(valid_ds if eval_valid else None),
        batch_size=second_batch_size,
        load_path=first_save_path,
        save_path=second_save_path,
        loss_fn=loss_fn,
        lr=lr * second_lr_mult,
        epochs=second_epochs,
        force_retrain=True,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        train_with_dropout=train_with_dropout,
        linear_probe=second_linear_probe,
        lr_schedule=second_lr_schedule,
        warmup_steps=second_warmup_steps,
        end_mult=0.1,
        optimizer_name=optim,
        eval_every=eval_every,
    )

    if weak_ds is not None:
        weak_ds.save_to_disk(second_save_path + "/" + "strong_labels")

    acc = np.mean([x["acc"] for x in test_results])
    res_dict = {"accuracy": acc}
    print("accuracy:", acc)
    
    config["gt_indices"] = list(gt_indices)
    with open(os.path.join(second_save_path, f"config.json"), "w") as f:
        json.dump(config, f, indent=2, default=int)

    with open(os.path.join(second_save_path, f"results_summary.json"), "w") as f:
        json.dump(res_dict, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
