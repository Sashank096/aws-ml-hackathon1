# Business Entity Resolution Pipeline

This is an offline, streaming entity-resolution pipeline for the challenge data.
It uses an on-disk SQLite index for multi-key blocking, a compact logistic-regression pair model, and a precision-oriented decision layer designed for macro F0.5.

## Project architecture

```text
notebooks/                         # exploration and experiment records
  01_eda.ipynb
  02_normalization.ipynb
  03_blocking.ipynb
  04_features.ipynb
  05_model_validation.ipynb
src/                               # reusable implementation
  config.py
  data_loader.py
  preprocessing.py
  blocking.py
  features.py
  model.py
  validation.py
  inference.py
  submission.py
  main.py                          # frozen end-to-end entry point
```

## EDA

Run `notebooks/01_eda.ipynb` for the read-only, chunked EDA. It uses `pd.read_csv(..., sep="\t")` and does not rewrite the raw TSV files.

## Run

From `student_resource/`:

```bash
python code/business_entity_resolution/src/main.py \
  --data dataset \
  --output output
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The final shipment should include `code/`, `output/`, the filled methodology document, and this README. Do not include the supplied raw dataset, generated SQLite index, Python caches, notebook checkpoints, or temporary files.

The SQLite index can be placed elsewhere with `--index`. It is generated from the supplied files and should not be included in the final ZIP unless desired.

## Design

- Name/address normalization is Unicode-aware and handles common legal and address abbreviations.
- Blocking uses exact normalized name, exact normalized address, first-name token, and postal-token keys within country.
- Candidate pairs are scored using exact-match, token-Jaccard, character similarity, postal overlap, and token-presence features.
- The pair model is trained from supplied ground truth positives and hard negatives drawn from the same blocking process.
- A high probability threshold plus independent-signal checks reduce false merges, which matters because F0.5 weights precision more than recall.
- No internet, external database, geocoder, or external enrichment is used.
