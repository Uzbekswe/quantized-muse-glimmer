# Stage 1 evidence bundle

This small committed bundle records the evidence that is safe to publish from
the first Q4 VESSL run. It intentionally separates verified artifact facts
from historical fields that were not exported before the workspace ended.

The custom Q4 file is preserved outside Git and is not included in this
repository. Its SHA-256 is recorded in `SHA256SUMS` and its metadata is in
`artifact-manifest.json`.

Recovered speed/perplexity records are included under `raw/` and retain their
original imported-measurement markers. Prompt records remain in VESSL object
storage because they contain verbose model output. Billing snapshots, peak
memory, and the historical source commit remain unavailable; no values were
reconstructed from memory or retroactively presented as direct measurements.
