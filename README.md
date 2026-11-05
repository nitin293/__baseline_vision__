# Visual baseline evaluation

```
baseline_eval/
├── evaluate_baseline.py   # the single baselining script
├── infer_baseline.py      # run a saved checkpoint on images + score against labels
├── auxiliary_scripts/
│   └── download_models.py # pre-download + verify pre-trained weights (run this first)
├── plot_tsne.py           # t-SNE of saved embeddings
├── config.json            # predefined config (all models on)
├── requirements.txt
└── models/                # one script per PTM + the two downstream heads + pipeline
    ├── resnet.py vgg.py densenet.py efficientnet.py mobilenet.py vit.py   (torchvision)
    ├── dinov2.py clip.py                                                  (HuggingFace transformers)
    ├── fcn.py            # D -> 128 -> 256 -> 128 -> classes
    ├── cnn_maxpool.py    # Conv3x3(64)+MaxPool -> Conv3x3(128)+MaxPool -> Flatten -> Linear
    └── pipeline.py       # BaselinePipeline(backbone -> head), used for fine-tuning
```

## Run
```bash
pip install -r requirements.txt
python auxiliary_scripts/download_models.py            # once, while online (optional --cache_dir ./pretrained)
python evaluate_baseline.py --metadata meta.csv --config config.json \
       --filepath image_path --target label --out_dir results \
       --metrics accuracy eer loss f1_macro --monitor loss --device auto   # add --cache_dir ./pretrained if used above
python plot_tsne.py --embeddings_dir results/embeddings --out_dir results/tsne --splits test
```
* `--filepath` / `--target`: metadata column names. `--root_dir` prefixes image paths (relative paths are also tried relative to the CSV).
* Splits: by default, the main `--metadata` CSV is stratified into train/validation/test using `--val_size` and `--test_size` (default 15% each). Use `--split_col` for a preassigned train/val/test column. Alternatively, provide independent `--val_metadata` and/or `--test_metadata` CSVs; any held-out split not supplied is carved from the main metadata pool. With separate CSVs, `--metadata` is the training pool, and validation/test labels must belong to its training classes. Do not combine `--split_col` with separate validation/test CSVs.
* Example with automatic holdouts: omit `--val_metadata`/`--test_metadata`; the evaluator creates both splits from `--metadata`. Example with only an external test set: pass `--test_metadata test.csv`; validation is split from the main metadata using `--val_size`.
* `--metrics`: accuracy, balanced_accuracy, loss, eer, auc, f1_macro, f1_weighted, precision_macro, recall_macro.
  EER = binary EER, or macro one-vs-rest EER for multiclass. `--monitor` picks best epoch / early stopping (default `loss`).
* `--device auto|cpu|cuda|mps`.
* Saving flags (each has a `--no-...` form): `--save_checkpoints` (default **off**), `--save_embeddings`, `--save_confusion_matrix`,
  `--save_predictions`, `--save_plots` (default on).
* Progress bars (tqdm): overall runs, epochs with live loss/accuracy, batches (fine-tuning) and embedding extraction. `--no_progress` switches back to plain per-epoch log lines.
* `--cache_dir DIR`: where pre-trained weights live (same value as for `download_models.py`; default = the standard torch / HuggingFace caches).
* Other: `--models resnet18 vgg16` (subset), `--epochs N` (override), `--no_pretrained` (random init, offline smoke test),
  `--skip_existing`, `--fail_fast`, `--list_models`, `--list_layers resnet50`.

## Config (`config.json`)
Precedence per run: built-in defaults < `global` < `downstream.<head>` < model-level keys < `model.<mode>` < `model.<head>`.

| key | meaning |
|---|---|
| `enabled` | turn a model on/off |
| `mode` | `frozen` (extract embeddings once, train head), `finetune` (train backbone + head), or `both` |
| `downstream` | list of heads: `fcn`, `cnn_maxpool` |
| `embedding_layer` | `"final"` (head-less output) or any sub-module path, e.g. `"layer3"`, `"features.6"`, `"m.encoder.layer.8"` (see `--list_layers`). 4-D maps are global-avg-pooled, token sequences are mean-pooled |
| `pretrained` | use pre-trained weights (downloaded on first use) |
| `lr`, `backbone_lr` | head LR / backbone LR (fine-tune; defaults to `lr`) |
| `weight_decay`, `lr_scheduler` (`none/step/exponential/cosine`), `lr_decay`, `lr_step` | decay settings |
| `optimizer` (`adam/adamw/sgd`), `epochs`, `batch_size`, `patience`, `augment`, `num_workers` | training |
| `dropout`, `activation`, `norm` | head options |
| `normalize_embeddings` | `standardize` (z-score with train stats) or `none` (frozen mode) |

The CNN-maxpool head reshapes the (B, D) embedding into a zero-padded 1 x s x s map (s = ceil(sqrt(D))) before the convolutions.

## Outputs (`--out_dir`)
```
summary.csv / summary.json      one row per run (val_* and test_* for each metric)
config_used.json  split.csv
embeddings/<model>__frozen__<layer>/{train,val,test}.npz      (frozen: raw PTM embeddings)
embeddings/<run>/{train,val,test}.npz                         (finetune: embeddings of the tuned backbone)
runs/<model>__<mode>__<head>[__<layer>]/
    results.json history.csv training_curves.png predictions_test.csv
    confusion_matrix_test.{csv,png}   checkpoints/best.pt
```
Classification targets only. Adding a model = one new file in `models/` with `@register("name")`, imported in `models/__init__.py`.

## auxiliary_scripts/download_models.py
Downloads and verifies the weights of every enabled model in the config (`--models a b` for specific ones, `--all` for everything,
`--list` to just show sources, `--retries N`, `--cache_dir DIR`). Already-cached models are not downloaded again; the exit code is 1 if any model failed.

## infer_baseline.py
Runs a checkpoint saved with `--save_checkpoints` on any metadata CSV and, if `--target` is given, reports the metrics against the true labels
(`--metrics accuracy eer loss f1_macro ...`, same definitions as `evaluate_baseline.py`).
```bash
python infer_baseline.py --checkpoint results/runs/resnet18__finetune__fcn \
       --metadata meta.csv --filepath image_path --target label --out_dir infer_out
```
`--checkpoint` is a `best.pt` file or a run directory. Add `--split_col fold --split test` to score only one split, `--drop_unknown` to ignore labels the
model never saw, `--save_embeddings` to also dump backbone embeddings. Outputs: `predictions.csv` (pred, confidence, per-class probabilities, true, correct),
`metrics.json`, `confusion_matrix.{csv,png}`. Frozen checkpoints hold only the head, so their backbone is rebuilt from the pre-trained weights (use the same `--cache_dir`).


## GPU throughput settings

The default `config.json` is configured for a high-memory CUDA GPU (e.g., NVIDIA H200):
- Global image batch size: 128
- DataLoader workers: 16
- Pinned host memory, persistent workers, and prefetch factor 4 when workers are enabled
- CUDA BF16 autocast (automatically falls back to FP16 if BF16 is unsupported)
- Non-blocking host-to-device transfers
- Batched GPU-to-CPU transfers for extracted embeddings and predictions

Fine-tuning batch sizes are set separately to 64 for ResNet-18 and MobileNetV3-Small.
These are starting points, not guaranteed optima; benchmark batch sizes and worker counts on
the target storage system. Override globally with `--batch_size` and `--num_workers`.
Set `global.amp` to `false` or `global.amp_dtype` to `float32` to disable autocast.


## GPU-resident image cache for frozen CUDA extraction

For frozen PTM embedding extraction on CUDA, the evaluator now:
1. Resizes each source image once using the backbone input size and stores RGB pixels as uint8.
2. Writes the complete preprocessed dataset to a persistent `.pt` cache under `OUT_DIR/image_cache/`.
3. Loads that tensor onto the GPU once and reuses it across frozen PTMs with the same input size.
4. Selects each split by GPU indices, converts only each batch to FP32, applies the backbone-specific mean/std on GPU, and runs PTM inference.

This avoids repeated PIL decoding and CPU-to-GPU transfers for every frozen PTM. The cache is invalidated when image paths, file sizes, modification times, or input size change. The cache requires approximately 9.6 GiB for 63,726 RGB images resized to 224x224 (uint8); larger input sizes require proportionally more. Ensure sufficient local disk space for the `.pt` cache and sufficient GPU memory.

The cache is used for frozen CUDA embedding extraction. Fine-tuning continues to use the ordinary DataLoader so that training-time augmentation remains intact. On CPU/MPS, frozen extraction also retains the ordinary DataLoader path.

Set `--image_cache_dir /path/to/cache` to place the persistent `.pt` cache on another local disk.
