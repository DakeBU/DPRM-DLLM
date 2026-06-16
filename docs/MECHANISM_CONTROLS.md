# Mechanism Controls

These controls are meant to test whether DPRM is doing more than changing the
amount of exploration or compute. They are lightweight enough to port to most
host repositories.

## Reward-Blind Entropy Control

Use the same reveal budget, generation steps, shortlist size, task split, and
seed set as DPRM, but remove the bucket reward value. The cheap default score is:

```text
u_i(s_t) = 1 - max_v p_theta(v | s_t)
```

This is an uncertainty proxy, not a DPRM value. It asks whether the observed gain
comes merely from avoiding over-confident confidence ordering.

## Shuffled-Bucket Control

Keep counts, gates, beta, warmup, shortlist, and decode budget fixed. Randomly
permute nonempty bucket values before inference:

```text
R_hat(phi, b, a) -> R_hat(pi(phi, b, a))
```

If performance drops toward entropy-only or gate-only, the learned terminal
reward values carry useful information.

## Gate-Only And Count-Only Controls

Gate-only keeps the readiness scaffold but sets the reward value to zero:

```text
score_i = log p_i
```

Count-only keeps local readiness gates but replaces each bucket value with a
global mean value. These controls test whether readiness/count coverage alone is
responsible for the effect.

## EOT And Suffix-Anchor Diagnostics

For text-token diffusion in VQA or chat formats, log:

- candidate and selected end-of-turn token counts;
- selected EOT confidence;
- reveal-step histogram for EOT tokens;
- answer-position or suffix-anchor bins.

Direct EOT suppression and suffix-anchor confidence modulation are useful
diagnostic baselines. They should not be treated as DPRM components; they test
whether coarse phase/confidence buckets have enough structure for a task.

## Compute-Fair Reporting

For every mechanism control, report:

- identical task split and seed set;
- identical reveal budget and generation length;
- identical host model checkpoint and verifier/reward;
- NFE or wall-clock cost when the host changes the number of forward passes;
- per-example logs whenever paired bootstrap is possible.

The compact public summary is
[`statistics_outputs/mechanism_controls.csv`](../statistics_outputs/mechanism_controls.csv).
