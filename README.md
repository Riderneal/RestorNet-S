# RestorNet-S - Structure-Preserving Joint Denoising & Super-Resolution

**KLA Track: AI-Based Restoration of Degraded Inspection Images**

RestorNet-S restores degraded semiconductor inspection images - noisy,
low-resolution captures that hide the exact defects an inspection system
needs to catch - into clean, full-resolution images, in a single forward
pass.

## Problem

In semiconductor manufacturing, microscopic inspection images must be
extremely sharp and clean, because a single pixel of noise or loss of
detail can hide a defect that causes a chip to fail. Real inspection
images are degraded by two compounding effects:

- **Speckle noise** - grainy, multiplicative noise that can push pixel
  intensities outside the original valid range.
- **Spatial resolution reduction** - downsampling that discards fine
  detail.

Most restoration methods treat denoising and super-resolution as two
separate steps, and generic models tend to smooth away the fine repeating
structures (DRAM columns, FinFET fins) that inspection depends on.
RestorNet-S treats both problems as **one inverse problem**, solved
jointly, with an explicit bias toward preserving periodic device
structure.

## How it works

1. **Physics-matched degradation model** - multiplicative speckle noise ->
   additive Gaussian noise -> bicubic downsampling, matching real
   inspection-camera degradation (`restornet/degradation.py`).
2. **Hierarchical residual encoder-decoder with channel attention** - a
   stack of Residual Dense Blocks (RDB), each with squeeze-and-excitation
   channel attention, plus a lightweight dilated-conv "periodicity" branch
   that widens the receptive field along repeating structures instead of
   smoothing them out (`restornet/model.py`).
3. **Sub-pixel (PixelShuffle) upsampling** performs the resolution
   recovery *and* denoising in the same forward pass; a bicubic-upsampled
   copy of the input is added back as a global residual so the network
   only has to learn the correction.
4. **Multi-scale hybrid loss**: L1 + multi-scale SSIM + Sobel edge-aware
   loss, so the model is pushed to get both pixel values and structural
   edges right (`restornet/losses.py`).

```
degraded image -> RestorNet-S -> restored image
   (low-res,        (single         (clean, full-
    noisy)         forward pass)     resolution)
```

## About the dataset

**The official KLA inspection-image dataset had not been released at the
time this repository was built.** To keep the full pipeline runnable and
verifiable end-to-end right now, the repo ships with:

- `restornet/dataset.py::SyntheticChipDataset` - a procedural generator
  that produces grayscale images with the *structural* properties this
  task cares about (periodic DRAM-like grids, FinFET-like parallel fins,
  and small defect blobs), with no proprietary data involved.
- `restornet/dataset.py::FolderDataset` - a drop-in loader for **real**
  images. Point `--dataset folder --data-dir <path>` at the real KLA
  dataset once it's available; no other code changes are required.
- `weights/restornet_s_final.pth` - a checkpoint trained on the synthetic
  dataset (17 epochs, see `weights/training_log.json`), included so the
  inference and evaluation scripts are runnable immediately. **Retrain on
  the real dataset before drawing conclusions about real-world
  performance** - see [Training](#training).
- `sample_outputs/` - degraded / restored / ground-truth triplets and a
  comparison grid, generated with `scripts/make_samples.py` on synthetic
  data for the same reason.

Everything downstream (training loop, inference script, evaluation
script, loss functions) is written against real image folders and is
dataset-agnostic - swapping in the KLA dataset is a one-flag change.

## Installation

Requires Python 3.10+.

```bash
git clone <this-repo-url>
cd restornet-s
pip install -r requirements.txt
```

Everything runs on CPU (as demonstrated in this repo); a CUDA GPU is
picked up automatically if available (`torch.cuda.is_available()`).

## Running inference

```bash
python scripts/inference.py \
    --input  path/to/degraded_images \
    --output path/to/restored_images \
    --weights weights/restornet_s_final.pth
```

- **Input**: a folder of grayscale images (`.png`, `.jpg`, `.jpeg`,
  `.tif`, `.tiff`, `.bmp`). Any resolution - the network is fully
  convolutional.
- **Output**: `--output` is created if it doesn't exist; each image is
  restored (denoised + super-resolved by the checkpoint's scale factor,
  x2 by default) and saved as `<name>_restored.png`.
- No manual edits are needed - every path is a CLI flag. For very large
  images that don't fit in memory at once, add `--tile 256` to run
  tiled inference with automatic blending.

Try it on the bundled samples:

```bash
python scripts/inference.py \
    --input sample_outputs/degraded \
    --output /tmp/restored_demo \
    --weights weights/restornet_s_final.pth
```

## Training

```bash
# Quick run on the built-in synthetic dataset (what produced the shipped checkpoint)
python scripts/train.py --dataset synthetic --epochs 30 --steps-per-epoch 40 \
    --batch-size 8 --patch-size 96 --scale 2 --out weights/restornet_s_final.pth

# Real data (once the KLA dataset is available)
python scripts/train.py --dataset folder \
    --data-dir path/to/train_images --val-dir path/to/val_images \
    --epochs 100 --batch-size 16 --patch-size 128 --scale 2 \
    --out weights/restornet_s_final.pth
```

Useful flags: `--speckle-sigma`, `--gaussian-sigma` (degradation
strength), `--base-channels`, `--n-rdb` (model capacity), `--resume
<ckpt>` (continue training from a checkpoint). A per-epoch loss curve is
written to `--log` (default `weights/training_log.json`).

## Evaluation

```bash
python scripts/evaluate.py \
    --restored     path/to/restored_images \
    --ground-truth path/to/ground_truth_images
```

Matches files by filename stem, then prints per-image and averaged
**SSIM**, **PSNR**, and **LPIPS** (requires `pip install lpips`; add
`--no-lpips` to skip it, e.g. offline). Example, on the bundled synthetic
samples (`sample_outputs/metrics.txt`):

```
image         PSNR (dB)      SSIM     LPIPS
-------------------------------------------
sample_00        26.387    0.9576       n/a
sample_01        24.042    0.9699       n/a
sample_02        21.670    0.9440       n/a
sample_03        26.197    0.9829       n/a
sample_04        26.065    0.9826       n/a
sample_05        23.800    0.9263       n/a
-------------------------------------------
AVERAGE          24.694    0.9605       n/a
```

(LPIPS omitted above only because this environment had no internet access
to fetch AlexNet weights - it runs normally with a network connection.)

## Sample outputs

See `sample_outputs/comparison_grid.png` for degraded -> restored -> ground
truth comparisons, and `sample_outputs/{degraded,restored,ground_truth}/`
for the individual images. Regenerate with:

```bash
python scripts/make_samples.py --weights weights/restornet_s_final.pth
```

## Repository layout

```
restornet-s/
|--- README.md
|--- requirements.txt
|--- restornet/              # importable package
|   |--- model.py             # RestorNet-S architecture
|   |--- degradation.py       # physics-based degradation engine
|   |--- losses.py            # L1 + MS-SSIM + edge-aware hybrid loss
|   `--- dataset.py           # FolderDataset (real) + SyntheticChipDataset
|--- scripts/
|   |--- train.py              # training loop (synthetic or real data)
|   |--- inference.py          # folder-in -> folder-out restoration
|   |--- evaluate.py           # SSIM / PSNR / LPIPS
|   `--- make_samples.py       # regenerates sample_outputs/
|--- weights/
|   |--- restornet_s_final.pth
|   `--- training_log.json
`--- sample_outputs/
    |--- degraded/  restored/  ground_truth/
    |--- comparison_grid.png
    `--- metrics.txt
```

## Target metrics (per the project plan)

SSIM >= 0.90-0.94, PSNR >= 30-34 dB on the official KLA validation/test
splits (in-distribution + out-of-distribution), with inference under a
few seconds per image, model size under 50 MB, and zero additional
hardware - pure software restoration deployable on existing inspection
tooling. Current numbers above are on **synthetic placeholder data** with
a lightly-trained checkpoint; they will be re-measured once training and
evaluation are run against the real dataset.

## License

MIT - see `LICENSE`.
