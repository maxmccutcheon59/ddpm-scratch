# WRITEUP — ddpm-scratch

## Citation

Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.

Official paper: https://arxiv.org/abs/2006.11239

This repository is an independent educational / portfolio reimplementation by Max McCutcheon. It is **not** affiliated with the original authors.

## Method (what we implemented)

1. **Forward process** `q(x_t | x_0)` with a **linear β schedule** over `T` discrete steps, using the closed form  
   `x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε`, `ε ~ N(0,I)` (Ho et al. Eq. 4).
2. **Parameterization**: a small U-Net predicts noise `ε_θ(x_t, t)`.
3. **Training objective**: simplified loss  
   `L_simple = E_{t,x_0,ε} [ || ε − ε_θ(x_t, t) ||² ]` (Eq. 14).
4. **Sampling**: ancestral sampler (Algorithm 2), using the predicted `x_0` to form the reverse posterior mean, with σ_t² set to the true posterior variance β̃_t.

Code maps to paper symbols in `ddpm_scratch/schedule.py` and `ddpm_scratch/diffusion.py`.

## Experiment scale vs paper

| Setting | Ho et al. 2020 (typical) | This repo (default CPU) |
|--------|---------------------------|-------------------------|
| Data | CIFAR-10 32×32, LSUN, etc. | MNIST / Fashion-MNIST 28×28 |
| Timesteps `T` | 1000 | 200 (smoke: 50) |
| Architecture | Large U-Net, GroupNorm, attention | Compact U-Net, `base_channels=32`, no attention |
| Compute | Multi-GPU / long training | CPU-friendly, few epochs |
| Metrics | FID / IS reported in paper | Train MSE + qualitative grids only |

**We do not claim SOTA, paper-matching FID, or CIFAR-10 parity.** Digits/garments may look blurry or mode-collapsed after short CPU runs; that is expected at this scale.

## Honest results stance

- After a short `configs/smoke.yaml` run, loss should decrease over a few steps; samples are not meaningful.
- After `configs/cpu_mnist.yaml` (e.g. 5 CPU epochs), expect recognizable digit-like blobs for some seeds, not paper-quality samples.
- No FID numbers are fabricated or reported here without a measured evaluation script on a fixed protocol.

## Limitations

- No classifier-free guidance, DDIM, latent diffusion, or EMA weights in v0.1.0.
- No multi-GPU, mixed precision, or large-resolution support.
- Dataset download uses torchvision (see COMPLIANCE_NOTES.md for license notes).
- Research / educational code — not production generative infrastructure.

## Reproducibility

- Pin deps via `pip install -e ".[dev]"`; configs live under `configs/`.
- Seed is set in config (`seed: 0`) for torch CPU RNG; full bitwise reproducibility across platforms is not guaranteed.
