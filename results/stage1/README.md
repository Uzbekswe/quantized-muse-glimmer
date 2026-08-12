# Stage 1 evidence bundle

This small committed bundle records the evidence that is safe to publish from
the first Q4 VESSL run. It intentionally separates verified artifact facts
from historical fields that were not exported before the workspace ended.

The custom Q4 file is preserved outside Git and is not included in this
repository. Its SHA-256 is recorded in `SHA256SUMS` and its metadata is in
`artifact-manifest.json`.

Unavailable raw speed/perplexity records and billing snapshots are marked as
unavailable in `provenance.json`; no values were reconstructed from memory or
retroactively presented as raw measurements.
