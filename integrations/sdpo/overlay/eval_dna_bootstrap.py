import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from scipy.stats import pearsonr
from tqdm import tqdm

import dataloader_gosai
import oracle
from eval import compare_kmer
from sdpo_gosai import DiffusionSDPO
from utils import set_seed


def _load_models(args):
    GlobalHydra.instance().clear()
    initialize(config_path=args.config_path, job_name="eval_dna_bootstrap", version_base=None)
    cfg = compose(config_name=args.config_name)
    OmegaConf.set_struct(cfg, False)
    cfg.ordering = {
        "policy": args.order_policy,
        "dprm_beta": args.dprm_beta,
        "dprm_warmup_steps": args.dprm_warmup_steps,
        "dprm_switch_steps": args.dprm_switch_steps,
        "dprm_ready_count": args.dprm_ready_count,
        "dprm_phase_bins": args.dprm_phase_bins,
        "dprm_conf_bins": args.dprm_conf_bins,
        "dprm_shortlist_size": args.dprm_shortlist_size,
        "dprm_ablation": args.dprm_ablation,
        "dprm_ablation_seed": args.dprm_ablation_seed,
    }
    ref_path = os.path.join(args.base_path, args.ref_model_path)
    model_path = os.path.join(args.base_path, args.model_path)
    ref_model = DiffusionSDPO.load_from_checkpoint(
        ref_path, config=cfg, beta=1.0, generator="rkl", strict=False).to("cuda").eval()
    model = DiffusionSDPO.load_from_checkpoint(
        ref_path, config=cfg, beta=1.0, strict=False).to("cuda").eval()
    model.load_state_dict(torch.load(model_path, map_location="cuda"), strict=False)
    # Evaluation is aligned to the trained ordering policy; without a Lightning
    # Trainer attached, force the ordering schedule past the switch point.
    model._manual_global_step = max(args.dprm_switch_steps, args.dprm_warmup_steps)
    return model, ref_model


def _ci(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_path", type=str, default="data_and_model/")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ref_model_path", type=str, default="mdlm/outputs_gosai/pretrained.ckpt")
    parser.add_argument("--config_path", type=str, default="configs_gosai")
    parser.add_argument("--config_name", type=str, default="config_sdpo_gosai.yaml")
    parser.add_argument("--num_samples_per_batch", type=int, default=64)
    parser.add_argument("--num_sample_batches", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--raw_output",
        type=str,
        default="",
        help="Compressed per-sequence and bootstrap records; defaults next to --output.",
    )
    parser.add_argument("--order_policy", type=str, default="baseline",
                        choices=["baseline", "progressive", "entropy", "dprm", "dprm_random"])
    parser.add_argument("--dprm_beta", type=float, default=1.0)
    parser.add_argument("--dprm_warmup_steps", type=int, default=100)
    parser.add_argument("--dprm_switch_steps", type=int, default=400)
    parser.add_argument("--dprm_ready_count", type=int, default=64)
    parser.add_argument("--dprm_phase_bins", type=int, default=1)
    parser.add_argument("--dprm_conf_bins", type=int, default=10)
    parser.add_argument("--dprm_shortlist_size", type=int, default=64)
    parser.add_argument("--dprm_ablation", type=str, default="normal",
                        choices=["normal", "shuffled_bucket", "gate_only", "count_only"])
    parser.add_argument("--dprm_ablation_seed", type=int, default=20260611)
    args = parser.parse_args()

    set_seed(args.seed, use_cuda=True)
    model, ref_model = _load_models(args)
    model.reset_ordering_trace()

    all_raw, all_seqs = [], []
    for _ in tqdm(range(args.num_sample_batches), desc="sample"):
        samples = model._sample(eval_sp_size=args.num_samples_per_batch)
        all_raw.append(samples)
        all_seqs.extend(dataloader_gosai.batch_dna_detokenize(samples.detach().cpu().numpy()))
    all_raw = torch.cat(all_raw, dim=0)

    log_lik = ref_model.get_likelihood(all_raw, num_steps=128, n_samples=1).detach().cpu().numpy()
    preds = oracle.cal_gosai_pred_new(all_seqs, mode="eval")
    atac_preds = oracle.cal_atac_pred_new(all_seqs)
    atac_success = (atac_preds[:, 1] > 0.5).astype(np.float64)
    highexp = oracle.cal_highexp_kmers(return_clss=True)
    _, _, highexp_kmers_999, n_highexp_kmers_999, *_ = highexp

    rng = np.random.default_rng(args.seed)
    n = len(all_seqs)
    boot = {"hepg2_mean": [], "log_lik_mean": [], "atac_acc": [], "kmer_pearson": [], "total_metric": []}
    bootstrap_indices = np.empty((args.bootstrap, n), dtype=np.int32)
    for draw in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        bootstrap_indices[draw] = idx
        seqs = [all_seqs[i] for i in idx]
        hepg2 = preds[idx, 0].mean()
        atac = atac_success[idx].mean()
        kmers = oracle.count_kmers(seqs)
        kmer_p = compare_kmer(highexp_kmers_999, kmers, n_highexp_kmers_999, len(seqs))
        boot["hepg2_mean"].append(hepg2)
        boot["log_lik_mean"].append(log_lik[idx].mean())
        boot["atac_acc"].append(atac)
        boot["kmer_pearson"].append(kmer_p)
        boot["total_metric"].append(hepg2 * atac * kmer_p)

    metrics = {k: _ci(v) for k, v in boot.items()}
    metrics["n_samples"] = n
    metrics["order_policy"] = args.order_policy
    metrics["model_path"] = args.model_path
    metrics["ordering_coverage"] = model.dprm_coverage_summary()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    raw_output = args.raw_output or os.path.join(os.path.dirname(args.output), "eval_records.npz")
    np.savez_compressed(
        raw_output,
        sequences=np.asarray(all_seqs),
        token_ids=all_raw.detach().cpu().numpy(),
        hepg2=np.asarray(preds[:, 0], dtype=np.float64),
        atac_success=np.asarray(atac_success, dtype=np.float64),
        log_likelihood=np.asarray(log_lik, dtype=np.float64),
        bootstrap_indices=bootstrap_indices,
        **{f"bootstrap_{key}": np.asarray(values, dtype=np.float64) for key, values in boot.items()},
    )
    metrics["raw_records"] = os.path.basename(raw_output)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
