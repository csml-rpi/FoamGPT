# Per-case results

One CSV per experiment and one row per case and run. Columns: `experiment`, `run`, `family`, `variant`, `success` (1 = the solver log ends in `End` with the final time written and finite fields, within ten solver attempts), `failure_reason` (`max-loop-reached`: the ten attempts were used up; `early-exit`: the workflow stopped on malformed output; `non-finite-fields`), `last_loop`, `nmse` (lowest-is-best NMSE against the reference; empty when no evaluable solution), `credit` (field-fidelity credit: 1 if NMSE < 0.1, 0.5 if < 0.3, else 0; 0 for failed cases; empty where NMSE was not computed), `total_tokens` (prompt plus completion tokens of the case, where logged).

Field fidelity in the paper is the mean credit over the 33 cases of a run, averaged over runs. "Best of three" takes each case's highest credit across the three runs.

## Primary cluster (4 x A100-80GB)

| File | System | Paper |
|---|---|---|
| `E21_foamgpt_nemotron_shared.csv` | FoamGPT-Nemotron (shared adapter), 3 runs | Table 1, Table 2 (Shared), Tables 7-8, Figure 5 |
| `E23_foamgpt_gemma4_shared.csv` | FoamGPT-Gemma4 (shared adapter), 3 runs | Table 1, Table 8, Figure 5 |
| `E22_nemotron_nano_untuned.csv` | Nemotron-Nano, untuned, 3 runs | Table 1 |
| `E24_gemma4_untuned.csv` | Gemma4, untuned, 3 runs | Table 1 |
| `E08_split_stage_specific.csv` | Stage-specific split C, 3 runs | Table 2 |
| `E09_split_random_R1.csv`, `E10_split_random_R2.csv`, `E11_split_random_R3.csv` | Random splits R1-R3, 3 runs each | Table 2 |
| `E12_revert_parse_case_info.csv` ... `E20_revert_rewrite_files.csv` | Shared adapter with one call site on the base model, 1 run each | Figure 3 |
| `E25_advanced_foamgpt_nemotron_shared.csv` | Shared adapter on the 16 Advanced cases, 3 runs (scored offline; a case is solved when it has an evaluable solution) | Table 3 |

## Second cluster (4 x B200); success counts only runs with finite fields

| File | System | Paper |
|---|---|---|
| `E32_second_cluster_foamgpt_nemotron_shared.csv` | Shared adapter re-evaluated, 5 runs | App. F |
| `E33_nemotron_super_untuned.csv`, `E37_nemotron_super_untuned.csv` | Nemotron-Super 120B, untuned, 2 runs | Table 1 |
| `E46_qwen38_untuned.csv`, `E47_deepseek_v4_untuned.csv`, `E48_deepseek_v41_untuned.csv` | Qwen3.8 125B, DeepSeek-V4 284B, DeepSeek-V4.1 552B, untuned, 1 run each | Table 1 |
| `E53_gpt6_luna.csv`, `E54_gpt6_sol.csv` | GPT-6 Luna, GPT-6 Sol, 1 run each | Table 1 |
| `E38_scaling_025.csv` ... `E41_scaling_100.csv` | Shared adapter retrained on 25/50/75/100% of the training rollouts, 3 runs each | Figure 4 |
