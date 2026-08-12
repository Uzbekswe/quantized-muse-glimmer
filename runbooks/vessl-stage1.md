# VESSL Stage 1: Q4 plus text baselines

This run creates and preserves one custom `Q4_K_M` Muse Glimmer GGUF, then
benchmarks it against BF16 and Meta's two official text GGUF baselines. It does
not create an importance matrix, download `mmproj`/DFlash, or create other
custom quantizations.

## Hard limits

- One A100 80 GB workspace only.
- Stop at 5.5 wall-clock hours or 10 credits, whichever comes first.
- Stop immediately after the Q4 copy and baseline report are verified.
- Terminate the workspace; do not leave it paused or running.

## Remote setup

```bash
git clone https://github.com/Uzbekswe/quantized-muse-glimmer.git
cd quantized-muse-glimmer
python -m pip install -e '.[dev]'
git clone --branch b10353 https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j "$(nproc)"
mkdir -p results
exec > >(tee -a results/command-log.txt) 2> >(tee -a results/failure-log.txt >&2)
date -u
vesslctl billing show --output json | tee results/billing-start.json
python -m scripts.preflight --llama-cpp-dir llama.cpp --output results/preflight.json
```

The workspace must provide at least 64 GiB system RAM, 150 GiB free disk, and
an A100-visible CUDA runtime. The object volume should be mounted at
`/shared/results`; do not mount it over the repository or `/root`.

## Stage commands

```bash
python -m scripts.prepare --execute --text-only
python -m scripts.convert --execute
python -m scripts.preflight --require-artifacts --output results/preflight-after-convert.json

# Verify BF16 before deleting the source checkpoint.
./llama.cpp/build/bin/llama-cli \
  -m artifacts/converted/Muse-Glimmer-30B-BF16.gguf \
  -ngl 99 -c 512 --jinja --single-turn -n 16 \
  -p 'Reply with exactly: BF16 smoke test passed.'

# After the BF16 smoke test, reclaim the source-checkpoint disk space.
python -m scripts.cleanup_source --execute

python -m scripts.quantize --execute --recipes q4_k_m --threads "$(nproc)"
python -m scripts.preflight --require-artifacts --output results/preflight-after-q4.json

./llama.cpp/build/bin/llama-cli \
  -m artifacts/quantized/Muse-Glimmer-30B-q4_k_m.gguf \
  -ngl 99 -c 512 --jinja --single-turn -n 16 \
  -p 'Reply with exactly: Q4 smoke test passed.'

python -m scripts.preserve --execute \
  --source artifacts/quantized/Muse-Glimmer-30B-q4_k_m.gguf \
  --destination /shared/results/models/Muse-Glimmer-30B-custom-Q4_K_M.gguf

python -m scripts.benchmark --execute --mode all
python -m scripts.report --input results/benchmarks.jsonl
python -m scripts.prepare --verify-only
cp -a results /shared/results/stage1-results
cp -a report/generated /shared/results/stage1-report
```

Record billing before and after the run:

```bash
date -u
vesslctl billing show --output json | tee results/billing-end.json
```

If any command stalls, storage becomes unsafe, the model fails to load, or the
billing ceiling is approached, stop the current command and terminate the
workspace. No optional recipe should be started in this stage.
