# Stage 1: custom Q4 first (preliminary research snapshot)

This is a research-preview report, not a publication-grade benchmark release.
The custom Q4 artifact and its checksum are verified. Recovered speed,
perplexity, and prompt records are listed in `results/stage1/`; billing,
peak-memory, and the historical source commit remain unavailable and are not
reconstructed.

Completed on 2026-08-12 with one NVIDIA A100 SXM 80 GB workspace on VESSL.
The BF16 source was kept immutable. The custom quant was produced from the
converted BF16 GGUF with `llama.cpp` build `b10353`; Meta's published GGUF
files were evaluated only as external baselines.

## Artifact outcome

| Artifact | Bytes | GiB | Relative to BF16 |
|---|---:|---:|---:|
| BF16 GGUF | 55,725,514,112 | 51.90 | 100% |
| Custom `Q4_K_M` | 16,935,294,592 | 15.77 | 30.42% |

The custom Q4 is 38,790,219,520 bytes (36.13 GiB), or 69.58%, smaller than
the BF16 GGUF. Its SHA-256 is recorded in the preserved object-storage copy:
`dcd100005262563bdcef650b2de0b41f285570a5dc2760af63d546c224b2e21d`.

The Q4 model loaded and generated successfully. The BF16 smoke test also
exited successfully; the Q4 smoke output was confirmed during the subsequent
PTY-backed prompt run.

## Benchmark snapshot

All prompt records use one deterministic pass, seed 42, temperature 0, top-k
1, `--jinja`, and the same 10-prompt suite. Outputs were non-empty for all 40
records: 10 each for BF16, custom Q4, Meta's 17 GB build, and Meta's dynamic
build.

| Variant | Prefill tok/s | Decode tok/s | Held-out PPL |
|---|---:|---:|---:|
| BF16 | 3395.93 | 28.62 | 9.3473 |
| Custom `Q4_K_M` | 1440.91 | 52.95 | 9.5372 |
| Meta `kquant-17gb` | 1441.18 | 53.85 | 9.5390 |
| Meta dynamic | 1382.55 | 46.91 | 9.3619 |

The task suite includes reasoning, coding, tool-call JSON, multilingual text,
and general quantization questions. Only the arithmetic reasoning prompt has
an automatic exact-match check in this first tranche; the remaining task
records are qualitative outputs for later review. Peak memory was not
available because the container image did not include `/usr/bin/time -v`.

The prompt JSONL was persisted at VESSL object volume
`muse-glimmer-q4-stage1/stage1-final/` and remains available there. The
published speed and perplexity rows retain their original imported first-run
markers; they should be replaced by directly captured reruns in a later
publication-quality snapshot. The historical Stage 1 source was configured as
mutable `main`; its exact commit was not exported and is therefore unknown.
Future downloads are pinned to the immutable revision in
`configs/experiment.yaml`.

## Cost and scope

The second A100 workspace was terminated after export. The final VESSL
balance was 27.335 credits and the burn rate was 0. No Q5, Q6, importance
matrix, `mmproj`, or DFlash artifact was created in this tranche.

## Interpretation

Q4 achieved the primary milestone: a real, independently produced Muse
Glimmer quantized model that is small enough for substantially smaller-memory
systems than the BF16 source. The perplexity increase from BF16 to custom Q4
was 0.1899 on this small evaluation text, while the decode benchmark was
faster because the lower-weight model moves less data. These results are a
directional lab measurement, not a broad quality claim: the perplexity sample
is small, the task suite is small, and Meta's calibration recipe is unknown.

The next decision should be made from this snapshot. The highest-value next
experiment is calibrated `Q4_K_M` if the quality tradeoff is unacceptable;
otherwise create `Q5_K_M` for a quality/size comparison. Do not start another
GPU workspace until that choice is made.
