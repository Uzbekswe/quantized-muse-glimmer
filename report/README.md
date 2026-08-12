# Research report

The first GPU-backed tranche is documented in
[stage1-q4.md](stage1-q4.md). It answers:

1. How much smaller is each variant than the BF16 source?
2. What speed and memory tradeoffs appear on the same hardware?
3. Does importance-aware quantization improve perplexity or task behavior?
4. How close are the independently produced quants to Meta's published GGUF
   baselines?
5. Which claims are measured, and which remain unknown because Meta's exact
   calibration and internal recipe are not public?

Do not commit model weights or raw private prompts. Generated result files are
ignored by Git until a deliberate result snapshot is selected for publication.
