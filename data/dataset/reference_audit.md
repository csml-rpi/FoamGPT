# Reference-set audit

Field fidelity compares each solved case with the reference solution downloaded into `data/dataset/` by NMSE: for each of the velocity, pressure, density, and temperature fields present in both solutions, sum((ref - gen)^2) / sum(ref^2), averaged over those fields. The reference is read at its final written time and the generated case at its closest available time; when cell counts differ the reference is interpolated onto the generated mesh. Credit is 1 if NMSE < 0.1, 0.5 if NMSE < 0.3, and 0 otherwise; failed cases score 0.

Findings that limit what the metric measures:

| Family | Finding | Effect |
|---|---|---|
| `wedge` | The prompts of variants 8, 9, and 10 are byte-identical, and the references carry the variant-1 inlet condition while the prompts sweep it | Every completed `wedge` run in the shared vs. split comparison has NMSE 0.398 (credit 0); counting the three identical prompts once lowers the shared Nemotron success rate from 47.5% to 44.1% |
| `BernardCells` | The references write pressure in absolute rather than gauge form | NMSE of order 1e8 for runs that solve the case |
| six of the 11 families | Fewer distinct references than prompts | Several variants are scored against the same reference |
| `counterFlowFlame2D`, `forwardStep`, `obliqueShock`, `shallowWaterWithSquareBump` | Per-variant references and no convention mismatch | On these four families every system in the shared vs. split comparison scores between 0.21 and 0.25, all of it from `obliqueShock` |

We therefore report field fidelity as a diagnostic against a fixed reference set and rank systems by execution success. Per-case NMSE values are in [`data/results/`](../results/README.md).
