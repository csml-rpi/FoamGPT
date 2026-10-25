# Data

| File | Records | Content |
|---|---|---|
| `teacher_traces.jsonl.gz` | 4,117 | Every LLM call made by the teacher (Claude Opus 4.6) in Foam-Agent rollouts, one rollout per case over the 126 cases (110 Basic, 16 Advanced) |
| `train.jsonl.gz` | 2,647 | Records of variants 1-7 of the 11 Basic families, the training set before the length filter |
| `test.jsonl.gz` | 1,050 | Records of the held-out variants 8-10; used only for held-out losses, never for training |
| `manifests/nemotron.jsonl` | 2,647 lines | One line per training record (same order as `train.jsonl.gz`): `sha256` of the record's line, `case_id`, `call_site`, `loop_iteration`, chat-templated length in tokens, `status`; 2,500 `kept`, 147 `dropped_over_length` |
| `manifests/gemma4.jsonl` | 2,647 lines | The same with the Gemma4 tokenizer; 2,499 `kept`, 148 `dropped_over_length` |
| `scaling/selection.json` | - | Seed, algorithm, and the nested selection of training rollouts for the scaling curve |
| `scaling/L025`, `L050`, `L075`, `L100` | - | `case_ids.txt`: the training rollouts of each level (19, 39, 58, 77 rollouts) |

Record fields: `case_id`, `step` (plan, write, review, rewrite), `substep` (the call site), `loop_iteration` (0 for the first pass, 1 and above inside the repair loop), `file_target`, `system_prompt`, `user_prompt`, `response` (the teacher's completion), `model`, `prompt_tokens`, `completion_tokens`, `timestamp`.

The length filter keeps a record when its chat-templated system, user, and assistant text is at most 16,384 tokens (`training/build_manifest.py`); every dropped record is an `error_analysis` call. `sha256` is computed over the record's line in `train.jsonl` (UTF-8, without the newline).

Terms: the responses are outputs of Claude Opus 4.6 obtained through Amazon Bedrock and are subject to Anthropic's terms of service and usage policy. The prompts come from Foam-Agent rollouts on public OpenFOAM tutorial cases and contain no personal data. Released for research use.
