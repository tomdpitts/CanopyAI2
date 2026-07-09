# dapt/artifacts — layout & conventions

One subdir per experiment type; every script writes here by default. Filenames carry
`arm_capacity_fracNN_sSEED_cfghash` where applicable, so files never collide across
rounds — new rounds ADD files, they don't overwrite (pass --out to override).

| dir | contents | writer |
|---|---|---|
| `detection/` | multi-seed test summaries + paired gaps (`run_baseline`) | `dapt.run_baseline` |
| `selection/` | val checkpoint-selection rankings | `dapt.select_checkpoint` |
| `labeff/` | label-efficiency curves | `dapt.label_efficiency` |
| `semseg/` | area-F1 (box-fill pseudo-GT) summaries | `dapt.semseg` |
| `probes/` | single-probe runs: `<tag>.json` + `<tag>.pt` head weights | `dapt.train` (CLI) |
| `logs/` | run logs | ad hoc |

Naming for SSL-derived arms: `dapt_s<SSLseed>[<tag>]_i<iter>` (e.g. `dapt_s42_i999`,
`dapt_s42_shuf_i999`, `dapt_s43_p1k_i999`). Protocol tags: no tag = original 5k-iter
schedule; `_p1k` = short 1000-iter protocol; `_shuf` = shuffled-pool control.

Storage policy (large binaries):
- `dapt/ckpt/` holds ONLY winner HF checkpoints (val-selected). Full checkpoint
  trails are archived on the Modal volume (`dinov3-dapt-arid-vol`: DCP under
  `/out_s<seed><tag>/ckpt/`, teacher .pth under `/export/`) — re-download +
  `dapt.ssl.export_hf` to restore any arm.
- `dapt/cache/<arm>/` feature caches exist only for registered arms in active use;
  regenerable in ~25 s/arm via `dapt.cache_features`. Delete freely.

Results narrative: `dapt/REPORT.md`. Protocol + runbook: `dapt/ssl/SPEC.md`.
