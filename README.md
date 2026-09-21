# ddpm-scratch

**From-scratch PyTorch reimplementation** of *Denoising Diffusion Probabilistic Models* (Ho et al., NeurIPS 2020).

Author: **Max McCutcheon** \<MaxMcCutcheon1@outlook.com\>

This is **not** a tutorial clone. The math (linear β schedule, `q(x_t|x_0)`, noise-prediction loss L_simple, ancestral sampler) is implemented directly from the paper equations in a small, readable codebase aimed at CPU-friendly MNIST / Fashion-MNIST experiments.

> **Honesty note:** Results here are **demo-scale** (small U-Net, T≪1000, few CPU epochs on 28×28 grayscale). They are **not** comparable to the paper’s CIFAR-10 / LSUN numbers. See [WRITEUP.md](WRITEUP.md).

## Features

- Linear noise schedule + closed-form forward process
- Compact time-conditioned U-Net (ε-prediction)
- Training with simplified MSE objective (Ho et al. Eq. 14)
- Ancestral reverse sampling (Algorithm 2)
- CPU-first YAML configs (zero assumed GPU / cloud spend)
- Pytest coverage of schedule math and sampling shapes
- CI: pytest + pip-audit + gitleaks

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Unit tests (no dataset download required for schedule/unet tests)
pytest -q

# Short smoke train (downloads MNIST once into ./data)
python scripts/train.py --config configs/smoke.yaml

# Full-ish CPU demo (still small vs paper)
python scripts/train.py --config configs/cpu_mnist.yaml

# Fashion-MNIST variant
python scripts/train.py --config configs/cpu_fashion.yaml

# Sample from a checkpoint
python scripts/sample.py --checkpoint checkpoints/epoch_005.pt --out samples/generated.png
```

Default device in configs is **`cpu`**. Override only if you choose to: `--device cuda`.

## Layout

```
ddpm_scratch/     # library: schedule, diffusion, unet, data
configs/          # cpu_mnist, cpu_fashion, smoke
scripts/          # train.py, sample.py
tests/            # schedule / diffusion / unet math tests
WRITEUP.md        # method, citation, honest limitations
SECURITY.md
COMPLIANCE_NOTES.md
```

## Citation (paper)

```bibtex
@inproceedings{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Security

See [SECURITY.md](SECURITY.md). Vulnerability reports: MaxMcCutcheon1@outlook.com.
