# Chart map

| Artifact | Formula and artifact crosswalk | Checkpoint / split / direction | Form | Claim boundary |
| --- | --- | --- | --- | --- |
| `role_summary_{last,best_val}` | ID-1 `id/accuracy`; ID-2 `id/nll_raw`; GEO-5 `geometry/nc1_pinv`; GEO-10 `geometry/rankme_uncentered` | file-specific checkpoint; `id_test` or `id_train`; see Card 11 direction | Faceted seed dots and mean ± sample-SD interval | Descriptive n=3; no significance claim |
| `associations/<checkpoint>/<dataset>_<endpoint>` | geometry formula/artifact × OOD-1 AUROC or OOD-4 FPR; exact row mapping in the same-stem CSV | file-specific checkpoint and OOD dataset; AUROC up, FPR negated to performance-up | Dataset-specific 10-config Spearman heatmap | Exploratory association; no dataset pooling or causal claim |

Card 11 Methods wording for every formula/artifact pair is in [`tables/methods_crosswalk.md`](tables/methods_crosswalk.md).
