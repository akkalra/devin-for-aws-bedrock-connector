# sample_repo

Tiny batching utility with two seeded bugs (`pad_batch` pads to the shortest
sequence; `bucket_by_length` uses `<` instead of `<=`). The demo agent fixes them
using only Amazon Bedrock for inference. Run `python -m pytest -q` to check.
