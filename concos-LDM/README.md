# concos-LDM

A self-contained extension of the ConCoS repo: **latent diffusion on CIFAR-10** with ConCoS-trained
VAEs vs. ordinary ones, class-conditional (classifier-free guidance) vs. unconditional diffusion.
PyTorch only; it does not import or modify the parent TensorFlow package (the ConCoS loss is ported
in [`concos_ldm/loss.py`](concos_ldm/loss.py)).

Environment: `conda activate torch-gpu` (extra packages: `lpips`, `pytorch-fid`, `umap-learn`).
Data, checkpoints and encoded latents live in `~/.cache/concos-ldm` (override with `--work` or
`CONCOS_LDM_WORK`) so they stay out of OneDrive. Figures and logs go to `outputs/`.

## Two model families

| | v1 "spatial" | v2 "flat" |
|---|---|---|
| VAE latent | 4×8×8 image code (256 numbers) | 64-d vector |
| VAE decoder | ResBlocks with GroupNorm | dense + transposed convs, no normalisation |
| VAE loss | L1 + LPIPS + 1e-5·KL (+ 0.5·ConCoS) | MSE + 1e-4·KL + 0.01·Euclidean contrast (+ 0.5·ConCoS) |
| VAEs | `plain`, `concos` | `plain` (Euclidean only), `concos` (ConCoS + Euclidean), `concosonly` (ConCoS only) |
| LDM | 17M UNet, 60k steps | 10M MLP with FiLM, 20k steps |
| Code | `run.py`, [`train.py`](concos_ldm/train.py) | `run_flat.py`, [`flat.py`](concos_ldm/flat.py) |
| Train time (RTX 4060 laptop) | ~5.5 h | ~22 min (+ ~11 min for `concosonly`) |

Both LDM families use a cosine schedule, v-prediction, EMA weights and deterministic DDIM sampling with
`v = v_uncond + w · (v_cond − v_uncond)` (w = 1 plain conditional, w = 0 unconditional). Conditional models
drop the label with p = 0.1. Latents are normalised with a per-dimension mean and one global scale.

```bash
python run.py all                                                     # v1: 2 VAEs + 4 LDMs
python run_flat.py all                                                # v2: plain + concos, 4 LDMs
python run_flat.py train --name concosonly --concos-weight 0.5 --euclid-weight 0
```

## Analysis scripts

| Script | What it makes |
|---|---|
| `generate.py` | 10×10 sample grids for the four v1 LDMs |
| `evaluate.py` | FID, precision/recall, classifier accuracy, CFG sweep, FID vs. DDIM steps (v1) |
| `radial_variation.py` | decode γ·z for generated latents: class kept vs. image change (v1) |
| `viz_trajectories.py` | DDIM trajectories in PCA/UMAP of the latents (`--family`, `--method`, `--fit-traj`) |
| `viz_compare.py` | PCA and UMAP side by side for every model (`--families`, `--vaes`, `--split test`) |
| `rotate_scale.py` | rotate generated latents towards another class and scale them (v2) |

## Findings (one training run per model)

1. **v1: ConCoS collapsed into an animal / vehicle split.** Given only the ten flat labels, the ConCoS
   VAE put all six animal classes on one direction and all four vehicles on the opposite one (class-centre
   cosines ±1.00; 92% agreement with animal/vehicle from unsupervised 2-means). The superclass was
   already decodable from the plain VAE (83%) but was not its main structure.
2. **v1: ConCoS did not improve diffusion.** FID was worse at every guidance scale (e.g. 49.6 vs 45.2 at
   w = 3, 1k samples), class accuracy under CFG the same, and much worse at 10 DDIM steps (60.1 vs 47.4).
3. **v1: scaling γ·z kept the class for both VAEs.** The GroupNorm decoder divides out most of the
   latent's magnitude, so this architecture cannot distinguish ConCoS from a plain VAE on that property.
4. **v2: ConCoS reached the ten-direction simplex** (loss −0.193 of an optimal −0.194; inter-class cosine
   −1/9). On training latents UMAP shows clean islands; on held-out test latents they merge, and a linear
   probe is no better than the Euclidean-only VAE (62% vs 63%). The clean islands are partly memorisation.
5. **v2: ConCoS-only vs Euclidean-only.** Under scaling, ConCoS latents keep their class more often
   (45% vs 28% at γ = 0.5, 49% vs 38% at γ = 2) and the image changes less. Rotating a latent towards
   another class's direction switches the class equally often for both (16% at full rotation, measured
   by a classifier that recognises only ~37% of these blurry samples).

Figures for all of this are in `outputs/v1_spatial/` and `outputs/flat/`.

## Layout

```
run.py, run_flat.py          training entry points (v1, v2)
concos_ldm/loss.py           ConCoS + Euclidean contrastive loss
concos_ldm/models.py         v1 VAE and UNet
concos_ldm/flat.py           v2 VAE, MLP denoiser, training and sampler
concos_ldm/nn.py             shared blocks (ResBlock, attention, timestep embedding)
concos_ldm/diffusion.py      v-prediction loss, DDIM + CFG sampler (with trajectories)
concos_ldm/train.py          v1 training stages and loaders
concos_ldm/sample.py         v1 Sampler
concos_ldm/metrics.py        FID, precision/recall, scoring ResNet-18
concos_ldm/report.py         evaluate.py's tables and plots
concos_ldm/common.py         paths, seeding, GPU-resident CIFAR-10
```
