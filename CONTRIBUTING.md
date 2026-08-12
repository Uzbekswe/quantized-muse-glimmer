# Contributing

Contributions should improve reproducibility, teaching value, or benchmark
quality without placing model weights in Git.

Before opening a pull request:

- run `python -m pytest -q`;
- run the documented dry-run commands and Python compile check;
- preserve immutable model sources and record revisions, checksums, hardware,
  commands, and timestamps for new experiments;
- label imported, incomplete, or preliminary measurements clearly;
- do not commit GGUF, Safetensors, or other large model artifacts.

Quantization changes should include a focused configuration update, tests for
the command recipe, and a short report describing quality, speed, memory, and
size tradeoffs. New experiments must not overwrite an existing source or
published artifact.
