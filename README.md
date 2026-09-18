<p align="center">
  <img src="assets/beaver.jpg" width="200" alt="BEAVER">
</p>

# BEAVER

**BEAVER**, a branch-aware, confidence-gated reasoner, is a perception-grounded,
structured chain-of-thought system for whole-slide pathology images. This
repository contains the **inference code** for our REG2026 challenge submission.

Built upon the [NARWHAL](https://github.com/icgi/NARWHAL) codebase, with task-specific modifications.

## Build and run

`base.Dockerfile` builds the feature-extraction base (`feature-extraction:v1`):
PyTorch, [TRIDENT](https://github.com/mahmoodlab/TRIDENT), `timm`, `openslide`.
The `Dockerfile` builds BEAVER on top.

```bash
docker build -f base.Dockerfile -t feature-extraction:v1 .
docker build -f Dockerfile -t beaver .
docker run --rm --gpus all --network none \
  --volume /path/to/input:/input:ro \
  --volume /path/to/output:/output \
  --volume /path/to/weights:/opt/ml/model:ro \
  beaver
```

## Example output

`example/sample_cot_output.json` is a chain-of-thought produced by the pipeline.
Each step is `{"question", "answer", "next_question"}`:

```json
[
  {"question": "What is the organ?", "answer": "Breast",
   "next_question": "Is there any abnormality present?"},
  {"question": "What is the procedure?", "answer": "Core needle biopsy",
   "next_question": "Is there any abnormality present?"}
]
```

## Citation

A REG2026 challenge paper describing this work is in preparation; the citation
will be added here once available.

## License

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
