# DPLM-2-Bit ordering ablation

All rows use the original DPLM-2 Bit evaluation protocol: CAMEO2022 forward folding and unconditional co-generation at lengths 100--500 with 50 samples per length.

| Model | Forward bb-RMSD ↓ | Forward bb-TM ↑ | Cogen ca-RMSD ↓ | Cogen bb-TM ↑ | Cogen pLDDT ↑ | Designable % ↑ |
|---|---:|---:|---:|---:|---:|---:|
| DPLM-2 Bit | 35.47 | 0.3071 | 66.73 | 0.4063 | 67.49 | 23.6 |
| Progressive DPLM-2 Bit | 29.43 | 0.3321 | 91.64 | 0.4621 | 76.51 | 40.4 |
| DPRM-DPLM-2 Bit | 29.43 | 0.3321 | 75.70 | 0.4605 | 68.77 | 40.0 |
| DPRM(random)-DPLM-2 Bit | 29.43 | 0.3321 | 80.23 | 0.4527 | 74.09 | 39.6 |

## Deltas against DPLM-2 Bit

Forward-folding deltas are paired over common CAMEO2022 targets. Co-generation deltas are independent bootstrap deltas over generated samples.

- `Progressive DPLM-2 Bit - DPLM-2 Bit` `bb_rmsd_to_gt_delta` = -6.0339 [-6.5329, -5.5400] via paired_bootstrap_targets.
- `Progressive DPLM-2 Bit - DPLM-2 Bit` `bb_tmscore_to_gt_delta` = 0.0250 [0.0186, 0.0317] via paired_bootstrap_targets.
- `Progressive DPLM-2 Bit - DPLM-2 Bit` `ca_rmsd_to_gt_delta` = -6.2900 [-6.7882, -5.7973] via paired_bootstrap_targets.
- `DPRM-DPLM-2 Bit - DPLM-2 Bit` `bb_rmsd_to_gt_delta` = -6.0339 [-6.5329, -5.5400] via paired_bootstrap_targets.
- `DPRM-DPLM-2 Bit - DPLM-2 Bit` `bb_tmscore_to_gt_delta` = 0.0250 [0.0186, 0.0317] via paired_bootstrap_targets.
- `DPRM-DPLM-2 Bit - DPLM-2 Bit` `ca_rmsd_to_gt_delta` = -6.2900 [-6.7882, -5.7973] via paired_bootstrap_targets.
- `DPRM(random)-DPLM-2 Bit - DPLM-2 Bit` `bb_rmsd_to_gt_delta` = -6.0339 [-6.5329, -5.5400] via paired_bootstrap_targets.
- `DPRM(random)-DPLM-2 Bit - DPLM-2 Bit` `bb_tmscore_to_gt_delta` = 0.0250 [0.0186, 0.0317] via paired_bootstrap_targets.
- `DPRM(random)-DPLM-2 Bit - DPLM-2 Bit` `ca_rmsd_to_gt_delta` = -6.2900 [-6.7882, -5.7973] via paired_bootstrap_targets.
- `Progressive DPLM-2 Bit - DPLM-2 Bit` `ca_rmsd_delta` = 24.9096 [18.7942, 31.0759] via independent_bootstrap_samples.
- `Progressive DPLM-2 Bit - DPLM-2 Bit` `bb_tmscore_delta` = 0.0558 [0.0348, 0.0766] via independent_bootstrap_samples.
- `Progressive DPLM-2 Bit - DPLM-2 Bit` `mean_plddt_delta` = 9.0138 [6.7059, 11.2450] via independent_bootstrap_samples.
- `Progressive DPLM-2 Bit - DPLM-2 Bit` `designable_delta` = 0.1680 [0.0880, 0.2480] via independent_bootstrap_samples.
- `DPRM-DPLM-2 Bit - DPLM-2 Bit` `ca_rmsd_delta` = 8.9734 [3.4166, 14.3586] via independent_bootstrap_samples.
- `DPRM-DPLM-2 Bit - DPLM-2 Bit` `bb_tmscore_delta` = 0.0542 [0.0343, 0.0738] via independent_bootstrap_samples.
- `DPRM-DPLM-2 Bit - DPLM-2 Bit` `mean_plddt_delta` = 1.2828 [-1.4293, 3.9442] via independent_bootstrap_samples.
- `DPRM-DPLM-2 Bit - DPLM-2 Bit` `designable_delta` = 0.1640 [0.0840, 0.2480] via independent_bootstrap_samples.
- `DPRM(random)-DPLM-2 Bit - DPLM-2 Bit` `ca_rmsd_delta` = 13.5035 [7.7790, 18.9419] via independent_bootstrap_samples.
- `DPRM(random)-DPLM-2 Bit - DPLM-2 Bit` `bb_tmscore_delta` = 0.0464 [0.0253, 0.0671] via independent_bootstrap_samples.
- `DPRM(random)-DPLM-2 Bit - DPLM-2 Bit` `mean_plddt_delta` = 6.6029 [4.2642, 8.8941] via independent_bootstrap_samples.
- `DPRM(random)-DPLM-2 Bit - DPLM-2 Bit` `designable_delta` = 0.1600 [0.0800, 0.2400] via independent_bootstrap_samples.